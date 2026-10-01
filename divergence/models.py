"""Statistical tests of the divergence-gap hypothesis.

H1 (volatility): |D| and/or D at T predict higher realized (idiosyncratic) volatility
    over T+1..T+h, beyond HAR volatility terms, momentum, size, news volume, the
    level of media tone and earnings timing.
H2 (drawdowns): D_pos (company rosier than media) predicts larger forward maximum
    drawdowns and a higher probability of an anomalous (abnormal, > k-sigma) drawdown.
"""
from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy import stats

log = logging.getLogger(__name__)

CONTROLS = ["log_rv_5", "log_rv_21", "log_rv_63", "ret_21", "ret_252_21", "beta",
            "log_dvol", "log_n_news", "earn_week"]

SPECS = {
    "S1_signed": ["D"],
    "S2_abs": ["D", "absD", "zN"],
    "S3_asym": ["D_pos", "D_neg", "zN"],
}


def targets_for(h: int) -> dict[str, str]:
    return {
        f"fwd_log_rv_{h}": "log realized vol",
        f"fwd_log_idio_{h}": "log idiosyncratic vol",
        f"fwd_dd_{h}": "max drawdown depth",
        f"fwd_abdd_{h}": "abnormal drawdown depth",
        f"anom_dd_{h}": "anomalous drawdown (LPM)",
    }


def _dk_bandwidth(h: int) -> int:
    # Weekly sampling with h-day targets overlaps ~h/5 periods.
    return max(2, math.ceil(h / 5) + 1)


def panel_ols(df: pd.DataFrame, y: str, x: list[str], h: int, controls=CONTROLS) -> pd.DataFrame:
    cols = list(dict.fromkeys(x + list(controls)))
    d = df[["ticker", "date", y] + cols].dropna().set_index(["ticker", "date"])
    if len(d) < 200:
        return pd.DataFrame()
    mod = PanelOLS(d[y], d[cols], entity_effects=True, time_effects=True, drop_absorbed=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dk = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=_dk_bandwidth(h))
        cl = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    rows = []
    for v in x:
        if v not in dk.params:
            continue
        rows.append(dict(y=y, var=v, coef=dk.params[v], se_dk=dk.std_errors[v], t_dk=dk.tstats[v],
                         p_dk=dk.pvalues[v], t_cl2=cl.tstats[v], p_cl2=cl.pvalues[v],
                         nobs=int(dk.nobs), n_firms=d.index.get_level_values(0).nunique(),
                         r2_within=dk.rsquared_within))
    return pd.DataFrame(rows)


def regression_table(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = []
    for h in horizons:
        for y in targets_for(h):
            for spec, x in SPECS.items():
                r = panel_ols(panel, y, x, h)
                if not r.empty:
                    out.append(r.assign(h=h, spec=spec))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def logit_drawdown(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """Logit of anomalous drawdown with year dummies, firm-clustered SEs."""
    y = f"anom_dd_{h}"
    x = ["D_pos", "D_neg", "zN"] + CONTROLS + ["vix"]
    d = panel[["ticker", "year", y] + x].dropna()
    if d[y].sum() < 30:
        return pd.DataFrame()
    X = pd.concat([d[x], pd.get_dummies(d["year"], prefix="y", drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    groups = pd.factorize(d["ticker"])[0]
    res = sm.Logit(d[y], X).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
    me = res.get_margeff(at="overall")
    me_s = pd.Series(me.margeff, index=X.columns[1:])
    rows = []
    for v in ["D_pos", "D_neg", "zN"]:
        rows.append(dict(y=y, h=h, var=v, coef=res.params[v], odds_ratio=np.exp(res.params[v]),
                         z=res.tvalues[v], p=res.pvalues[v], marg_eff=me_s[v],
                         base_rate=d[y].mean(), nobs=int(res.nobs), pseudo_r2=res.prsquared))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dumitrescu-Hurlin panel Granger causality
# ---------------------------------------------------------------------------

def dumitrescu_hurlin(panel: pd.DataFrame, y: str, x: str, lags: int) -> dict:
    """H0: x does not Granger-cause y for any firm (heterogeneous panel)."""
    W, Ts = [], []
    for _, g in panel.sort_values("date").groupby("ticker"):
        g = g[[y, x]].copy()
        cols = []
        for k in range(1, lags + 1):
            g[f"y_l{k}"], g[f"x_l{k}"] = g[y].shift(k), g[x].shift(k)
            cols += [f"y_l{k}", f"x_l{k}"]
        g = g.dropna()
        T = len(g)
        if T < 5 + 3 * lags:
            continue
        X = sm.add_constant(g[cols])
        res = sm.OLS(g[y], X).fit()
        R = np.zeros((lags, X.shape[1]))
        for i, k in enumerate(range(1, lags + 1)):
            R[i, list(X.columns).index(f"x_l{k}")] = 1
        W.append(float(np.squeeze(res.wald_test(R, scalar=True, use_f=False).statistic)))
        Ts.append(T)
    if not W:
        return {}
    N, K, T = len(W), lags, float(np.mean(Ts))
    wbar = float(np.mean(W))
    zbar = math.sqrt(N / (2 * K)) * (wbar - K)
    ztilde = math.sqrt(N / (2 * K) * (T - 2 * K - 5) / (T - K - 3)) * ((T - 2 * K - 3) / (T - 2 * K - 1) * wbar - K)
    return dict(y=y, x=x, lags=K, n_firms=N, avg_T=T, W_bar=wbar, Z_bar=zbar,
                p_Z_bar=2 * stats.norm.sf(abs(zbar)), Z_tilde=ztilde, p_Z_tilde=2 * stats.norm.sf(abs(ztilde)))


def granger_table(panel: pd.DataFrame, lags: int) -> pd.DataFrame:
    tests = [("log_rv_5", "D"), ("log_rv_5", "absD"), ("D", "log_rv_5"), ("absD", "log_rv_5"),
             ("log_rv_5", "zN"), ("log_rv_5", "zC")]
    return pd.DataFrame([r for r in (dumitrescu_hurlin(panel, y, x, lags) for y, x in tests) if r])


# ---------------------------------------------------------------------------
# Portfolio sorts
# ---------------------------------------------------------------------------

def _nw_mean(x: pd.Series, lags: int) -> tuple[float, float]:
    x = x.dropna()
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0])


def quintile_sorts(panel: pd.DataFrame, sort_var: str, h: int, n_q: int = 5) -> pd.DataFrame:
    tgts = [f"fwd_log_rv_{h}", f"fwd_log_idio_{h}", f"fwd_dd_{h}", f"anom_dd_{h}"]
    d = panel[["date", sort_var, "log_rv_63"] + tgts].dropna(subset=[sort_var]).copy()
    d["dvol"] = d[f"fwd_log_rv_{h}"] - d["log_rv_63"]   # vol change vs. trailing quarter
    tgts = tgts + ["dvol"]
    d = d[d.groupby("date")[sort_var].transform("count") >= 2 * n_q]
    d["q"] = d.groupby("date")[sort_var].transform(lambda s: pd.qcut(s.rank(method="first"), n_q, labels=False) + 1)
    by = d.groupby(["date", "q"])[tgts].mean().unstack("q")
    rows = []
    for t in tgts:
        row = {"sort": sort_var, "h": h, "target": t}
        for q in range(1, n_q + 1):
            row[f"Q{q}"] = by[(t, q)].mean()
        m, tstat = _nw_mean(by[(t, n_q)] - by[(t, 1)], lags=_dk_bandwidth(h))
        row.update({f"Q{n_q}-Q1": m, "t_NW": tstat})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Out-of-sample forecasting
# ---------------------------------------------------------------------------

def _design(d: pd.DataFrame, cols: list[str], firms: list[str]) -> np.ndarray:
    fe = (d["ticker"].to_numpy()[:, None] == np.array(firms)[None, :]).astype(float)
    return np.column_stack([d[cols].to_numpy(float), fe])


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    pos = y == 1
    if pos.sum() == 0 or (~pos).sum() == 0:
        return np.nan
    ranks = stats.rankdata(s)
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def oos_forecast(panel: pd.DataFrame, h: int, first_test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window, year-by-year OOS test with an h-day embargo.

    Models (all with firm intercepts): base = controls + zN; aug = base + D + absD.
    Continuous targets: OOS R^2 vs. base and Clark-West (2007) nested test.
    Anomalous drawdowns: logit AUC.
    """
    base = CONTROLS + ["zN"]
    aug = base + ["D", "absD"]
    embargo = pd.Timedelta(days=int(h * 7 / 5) + 7)
    years = sorted(y for y in panel["year"].unique() if y >= first_test_year)
    out = []
    for y in [f"fwd_log_rv_{h}", f"fwd_log_idio_{h}", f"fwd_dd_{h}"]:
        d = panel[["ticker", "date", "year", y] + aug].dropna()
        preds = []
        for yr in years:
            cut = pd.Timestamp(f"{yr}-01-01")
            tr, te = d[d["date"] < cut - embargo], d[d["year"] == yr]
            if len(tr) < 500 or te.empty:
                continue
            firms = sorted(tr["ticker"].unique())
            te = te[te["ticker"].isin(firms)]
            p = {}
            for name, cols in (("base", base), ("aug", aug)):
                mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
                tr_s, te_s = tr.copy(), te.copy()
                tr_s[cols], te_s[cols] = (tr[cols] - mu) / sd, (te[cols] - mu) / sd
                Xtr, Xte = _design(tr_s, cols, firms), _design(te_s, cols, firms)
                b, *_ = np.linalg.lstsq(Xtr, tr[y].to_numpy(), rcond=None)
                p[name] = Xte @ b
            preds.append(pd.DataFrame({"date": te["date"].to_numpy(), "y": te[y].to_numpy(),
                                       "f0": p["base"], "f1": p["aug"]}))
        if not preds:
            continue
        P = pd.concat(preds)
        e0, e1 = P["y"] - P["f0"], P["y"] - P["f1"]
        f = e0 ** 2 - (e1 ** 2 - (P["f0"] - P["f1"]) ** 2)
        cw_mean, cw_t = _nw_mean(f.groupby(P["date"]).mean(), lags=_dk_bandwidth(h))
        out.append(dict(y=y, h=h, n=len(P), mse_base=float((e0 ** 2).mean()), mse_aug=float((e1 ** 2).mean()),
                        r2_oos_vs_base=1 - float((e1 ** 2).sum() / (e0 ** 2).sum()),
                        clark_west=cw_mean, cw_t=cw_t, cw_p_one_sided=float(stats.norm.sf(cw_t))))

    # Classification of anomalous drawdowns.
    yv = f"anom_dd_{h}"
    d = panel[["ticker", "date", "year", yv, "vix"] + aug].dropna()
    cls = []
    for yr in years:
        cut = pd.Timestamp(f"{yr}-01-01")
        tr, te = d[d["date"] < cut - embargo], d[d["year"] == yr]
        if tr[yv].sum() < 30 or te.empty:
            continue
        row = {"year": yr, "n": len(te), "events": int(te[yv].sum())}
        for name, cols in (("base", base + ["vix"]), ("aug", aug + ["vix"])):
            mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
            try:
                m = sm.Logit(tr[yv], sm.add_constant((tr[cols] - mu) / sd)).fit(disp=0, maxiter=200)
                row[f"auc_{name}"] = _auc(te[yv].to_numpy(), m.predict(sm.add_constant((te[cols] - mu) / sd, has_constant="add")).to_numpy())
            except Exception as e:  # perfect separation etc.
                log.warning("OOS logit failed for %s: %s", yr, e)
                row[f"auc_{name}"] = np.nan
        cls.append(row)
    return pd.DataFrame(out), pd.DataFrame(cls)


# ---------------------------------------------------------------------------
# Placebo: permutation of D across firms within each period
# ---------------------------------------------------------------------------

def permutation_test(panel: pd.DataFrame, y: str, var: str, n: int, seed: int) -> dict:
    cols = ["D", "absD", "D_pos", "D_neg"]
    d = panel[["ticker", "date", y, "zN"] + cols + CONTROLS].dropna().reset_index(drop=True)
    spec = ["D_pos", "D_neg", "zN"] if var in ("D_pos", "D_neg") else ["D"]

    def coef(frame):
        x = list(dict.fromkeys(spec + CONTROLS))
        f = frame.set_index(["ticker", "date"])
        return PanelOLS(f[y], f[x], entity_effects=True, time_effects=True).fit().params[var]

    actual = coef(d)
    rng = np.random.default_rng(seed)
    perm = np.empty(n)
    for i in range(n):
        idx = d.groupby("date").indices
        order = np.arange(len(d))
        for rows in idx.values():
            order[rows] = rng.permutation(rows)
        dd = d.copy()
        dd[cols] = d[cols].to_numpy()[order]
        perm[i] = coef(dd)
    return dict(y=y, var=var, coef=actual, n_perm=n, perm_mean=perm.mean(), perm_sd=perm.std(),
                p_two_sided=float((np.abs(perm) >= abs(actual)).mean()))


# ---------------------------------------------------------------------------
# Event study on daily data
# ---------------------------------------------------------------------------

def event_study(panel: pd.DataFrame, daily: pd.DataFrame, horizon: int = 63, pct: float = 0.9) -> pd.DataFrame:
    """Average cumulative abnormal return and abnormal |return| after divergence events.

    Events: firm enters the top (company rosier) or bottom (media rosier) decile of
    the period's D distribution. Day 0 is the decision date; paths start at day 1.
    """
    d = panel[["ticker", "date", "D"]].dropna().sort_values(["ticker", "date"]).copy()
    d["rank"] = d.groupby("date")["D"].rank(pct=True)
    d["grp"] = np.select([d["rank"] >= pct, d["rank"] <= 1 - pct], ["corp_rosier", "media_rosier"], "middle")
    d["prev"] = d.groupby("ticker")["grp"].shift(1)
    ev = d[(d["grp"] != d["prev"])]
    daily = daily.sort_values(["ticker", "date"])
    paths = []
    for tk, g in daily.groupby("ticker"):
        ab = g["abret"].to_numpy()
        absab = np.abs(ab)
        pos = pd.Index(g["date"])
        for _, e in ev[ev["ticker"] == tk].iterrows():
            i = pos.get_indexer([e["date"]])[0]
            if i < 0 or i + horizon >= len(ab):
                continue
            seg = ab[i + 1 : i + 1 + horizon]
            if np.isnan(seg).any():
                continue
            paths.append((e["grp"], np.cumsum(seg), absab[i + 1 : i + 1 + horizon]))
    rows = []
    for grp in ("corp_rosier", "middle", "media_rosier"):
        P = [p for p in paths if p[0] == grp]
        if not P:
            continue
        car = np.stack([p[1] for p in P])
        aab = np.stack([p[2] for p in P])
        for k in range(horizon):
            rows.append(dict(group=grp, day=k + 1, n_events=len(P), car=car[:, k].mean(),
                             car_se=car[:, k].std(ddof=1) / np.sqrt(len(P)), abs_abret=aab[:, k].mean()))
    return pd.DataFrame(rows)


def run_all(panel: pd.DataFrame, daily: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    a = cfg["analysis"]
    H = cfg["targets"]["horizons"]
    h0 = a["primary_horizon"]
    log.info("Panel regressions")
    res = {"regressions": regression_table(panel, H)}
    log.info("Logit")
    res["logit"] = pd.concat([logit_drawdown(panel, h) for h in H], ignore_index=True)
    log.info("Granger")
    res["granger"] = granger_table(panel, a["granger_lags"])
    log.info("Sorts")
    res["sorts"] = pd.concat([quintile_sorts(panel, v, h) for v in ("D", "absD") for h in H], ignore_index=True)
    log.info("OOS")
    oos, oos_cls = zip(*(oos_forecast(panel, h, a["oos_first_test_year"]) for h in H))
    res["oos"], res["oos_auc"] = pd.concat(oos, ignore_index=True), pd.concat(oos_cls, keys=H, names=["h"]).reset_index(0)
    log.info("Permutation placebo")
    res["permutation"] = pd.DataFrame([
        permutation_test(panel, f"fwd_log_idio_{h0}", "D", a["n_permutations"], a["seed"]),
        permutation_test(panel, f"fwd_abdd_{h0}", "D_pos", a["n_permutations"], a["seed"]),
    ])
    log.info("Event study")
    res["event_study"] = event_study(panel, daily, horizon=max(H))
    return res
