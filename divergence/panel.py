"""Firm-period panel: sentiment streams, divergence spread, forward risk targets.

Timing convention (no look-ahead):
  * A document is usable at decision date T iff its availability date <= T,
    where availability is the first trading close at/after publication.
  * Features at T use data up to and including T's close.
  * Targets at T use returns over trading days T+1 ... T+h only.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .calendar import availability_date, period_ends, to_eastern

log = logging.getLogger(__name__)
ANN = 252


# ---------------------------------------------------------------------------
# Market data: targets and controls
# ---------------------------------------------------------------------------

def _fwd_sum(x: pd.Series, h: int) -> pd.Series:
    """sum_{s=t+1}^{t+h} x_s (NaN if the window is incomplete)."""
    return x.rolling(h).sum().shift(-h)


def _fwd_paths(logp: np.ndarray, logm: np.ndarray, beta: np.ndarray, h: int):
    """Forward max drawdown of the raw and market-adjusted (abnormal) log paths."""
    n = len(logp)
    mdd = np.full(n, np.nan)
    amdd = np.full(n, np.nan)
    if n <= h:
        return mdd, amdd
    wp = sliding_window_view(logp, h + 1)          # rows t: logp[t..t+h]
    wm = sliding_window_view(logm, h + 1)
    p = wp - wp[:, :1]
    a = p - beta[: len(p), None] * (wm - wm[:, :1])
    mdd[: len(p)] = np.min(p - np.maximum.accumulate(p, axis=1), axis=1)
    amdd[: len(p)] = np.min(a - np.maximum.accumulate(a, axis=1), axis=1)
    return mdd, amdd


def ticker_market_features(px: pd.DataFrame, mkt: pd.Series, vix: pd.Series, cfg: dict) -> pd.DataFrame:
    """Daily controls and forward targets for one ticker.

    px: indexed by date with columns close, volume. mkt: market log returns on the
    same calendar. vix: VIX close.
    """
    t = cfg["targets"]
    px = px.sort_index()
    r = np.log(px["close"]).diff()
    rm = mkt.reindex(px.index)
    df = pd.DataFrame(index=px.index)
    df["ret"] = r

    bw = t["beta_window"]
    beta = (r.rolling(bw, min_periods=bw // 2).cov(rm) / rm.rolling(bw, min_periods=bw // 2).var())
    df["beta"] = beta
    e = r - beta.shift(1) * rm                          # out-of-sample idiosyncratic return
    df["abret"] = e
    df["sigma_idio"] = e.rolling(t["idio_window"], min_periods=t["idio_window"] // 2).std()

    for w in (5, 21, 63):
        df[f"rv_{w}"] = np.sqrt(ANN * (r ** 2).rolling(w, min_periods=int(w * 0.8)).mean())
        df[f"log_rv_{w}"] = np.log(df[f"rv_{w}"].clip(lower=1e-4))
    df["ret_21"] = r.rolling(21).sum()
    df["ret_252_21"] = r.rolling(231).sum().shift(21)   # 12-1 momentum
    df["log_dvol"] = np.log((px["close"] * px["volume"]).rolling(21, min_periods=10).mean().clip(lower=1))
    df["vix"] = vix.reindex(px.index).ffill()

    logp = np.log(px["close"]).to_numpy()
    logm = rm.fillna(0).cumsum().to_numpy()
    b = beta.fillna(1.0).to_numpy()
    for h in t["horizons"]:
        s_rr = _fwd_sum(r ** 2, h)
        s_rm = _fwd_sum(r * rm, h)
        s_mm = _fwd_sum(rm ** 2, h)
        df[f"fwd_rv_{h}"] = np.sqrt(ANN / h * s_rr)
        df[f"fwd_log_rv_{h}"] = np.log(df[f"fwd_rv_{h}"].clip(lower=1e-4))
        idio_ss = (s_rr - 2 * beta * s_rm + beta ** 2 * s_mm).clip(lower=0)
        df[f"fwd_log_idio_{h}"] = np.log(np.sqrt(ANN / h * idio_ss).clip(lower=1e-4))
        df[f"fwd_ret_{h}"] = _fwd_sum(r, h)
        df[f"fwd_abret_{h}"] = _fwd_sum(r, h) - beta * _fwd_sum(rm, h)
        mdd, amdd = _fwd_paths(logp, logm, b, h)
        df[f"fwd_dd_{h}"] = -np.expm1(mdd)       # drawdown depth, positive = worse
        df[f"fwd_abdd_{h}"] = -amdd              # abnormal (market-adjusted) log drawdown depth
        thresh = -t["drawdown_k"] * df["sigma_idio"] * np.sqrt(h)
        flag = (pd.Series(amdd, index=df.index) < thresh).astype(float)  # amdd <= 0
        df[f"anom_dd_{h}"] = flag.where(~np.isnan(amdd) & thresh.notna())
    return df


def market_panel(prices: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    u = cfg["universe"]
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    cal = pd.DatetimeIndex(wide[u["market_ticker"]].dropna().index)
    mkt = np.log(wide[u["market_ticker"]]).diff().reindex(cal)
    vix = wide[u["vix_ticker"]] if u["vix_ticker"] in wide else pd.Series(np.nan, index=cal)
    frames = []
    for tk, g in prices[prices["ticker"].isin(u["tickers"])].groupby("ticker"):
        g = g.set_index("date")[["close", "volume"]]
        g = g[g.index.isin(cal)]
        f = ticker_market_features(g, mkt, vix, cfg)
        f["ticker"] = tk
        frames.append(f)
    daily = pd.concat(frames).rename_axis("date").reset_index()
    return daily, cal


# ---------------------------------------------------------------------------
# Sentiment streams
# ---------------------------------------------------------------------------

def attach_availability(docs: pd.DataFrame, ts_col: str, source_tz: str, cal: pd.DatetimeIndex) -> pd.DataFrame:
    docs = docs.copy()
    docs["ts_et"] = to_eastern(docs[ts_col], source_tz)
    docs["avail"] = availability_date(docs["ts_et"], cal)
    return docs.dropna(subset=["avail", "score"]).reset_index(drop=True)


def type_adjust(docs: pd.DataFrame, min_prior: int = 20) -> pd.DataFrame:
    """Subtract the expanding cross-firm mean score of each corporate doc type.

    MD&A and press releases have very different baseline tones; mixing raw scores
    would make C jump whenever a 10-K arrives. Only documents available strictly
    before a document's availability date enter its baseline.
    """
    docs = docs.sort_values("avail").copy()
    docs["score_adj"] = np.nan
    for _, g in docs.groupby("doc_type"):
        daily = g.groupby("avail")["score"].agg(["sum", "count"])
        prior = daily.cumsum().shift(1)
        mean = (prior["sum"] / prior["count"]).where(prior["count"] >= min_prior)
        docs.loc[g.index, "score_adj"] = g["score"].to_numpy() - g["avail"].map(mean).to_numpy()
    return docs.dropna(subset=["score_adj"])


def ew_stream(docs: pd.DataFrame, ends: pd.DatetimeIndex, value: str, window_days: int,
              halflife_days: float, prefix: str) -> pd.DataFrame:
    """Exponentially-weighted mean of document scores available in (T - window, T]."""
    out = []
    ends_ns = ends.values.astype("datetime64[ns]")
    for tk, g in docs.groupby("ticker"):
        g = g.sort_values("avail")
        a = g["avail"].values.astype("datetime64[ns]")
        v = g[value].to_numpy(float)
        hi = np.searchsorted(a, ends_ns, side="right")
        lo = np.searchsorted(a, ends_ns - np.timedelta64(window_days, "D"), side="right")
        level = np.full(len(ends), np.nan)
        count = np.zeros(len(ends))
        age = np.full(len(ends), np.nan)
        for j in range(len(ends)):
            if hi[j] > lo[j]:
                days = (ends_ns[j] - a[lo[j]:hi[j]]) / np.timedelta64(1, "D")
                w = 0.5 ** (days / halflife_days)
                level[j] = np.dot(w, v[lo[j]:hi[j]]) / w.sum()
                count[j] = hi[j] - lo[j]
                age[j] = days.min()
        out.append(pd.DataFrame({"ticker": tk, "date": ends, prefix: level,
                                 f"n_{prefix}": count, f"age_{prefix}": age}))
    if not out:
        return pd.DataFrame(columns=["ticker", "date", prefix, f"n_{prefix}", f"age_{prefix}"])
    return pd.concat(out, ignore_index=True)


def _xs_z(s: pd.Series, dates: pd.Series, min_n: int) -> pd.Series:
    g = s.groupby(dates)
    z = (s - g.transform("mean")) / g.transform("std")
    return z.where(g.transform("count") >= min_n)


def _firm_z(s: pd.Series, tickers: pd.Series, min_hist: int) -> pd.Series:
    g = s.groupby(tickers)
    mu = g.transform(lambda x: x.expanding(min_hist).mean().shift(1))
    sd = g.transform(lambda x: x.expanding(min_hist).std().shift(1))
    return (s - mu) / sd


def _xs_resid(y: pd.Series, x: pd.Series, dates: pd.Series, min_n: int) -> pd.Series:
    out = pd.Series(np.nan, index=y.index)
    for _, idx in y.groupby(dates).groups.items():
        yy, xx = y.loc[idx], x.loc[idx]
        ok = yy.notna() & xx.notna()
        if ok.sum() < min_n:
            continue
        X = np.column_stack([np.ones(ok.sum()), xx[ok]])
        coef, *_ = np.linalg.lstsq(X, yy[ok], rcond=None)
        res = yy[ok] - X @ coef
        out.loc[res.index] = res / res.std()
    return out


def add_divergence(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pc = cfg["panel"]
    p = panel.sort_values(["ticker", "date"]).copy()
    mn, mh = pc["min_firms_per_period"], pc["firm_min_history"]
    both = p["C"].notna() & p["N"].notna()
    C, N = p["C"].where(both), p["N"].where(both)

    p["D_raw"] = C - N
    p["zC_xs"], p["zN_xs"] = _xs_z(C, p["date"], mn), _xs_z(N, p["date"], mn)
    p["D_xs"] = p["zC_xs"] - p["zN_xs"]
    p["zC_firm"], p["zN_firm"] = _firm_z(C, p["ticker"], mh), _firm_z(N, p["ticker"], mh)
    p["D_firm"] = p["zC_firm"] - p["zN_firm"]
    p["D_resid"] = _xs_resid(p["zC_xs"], p["zN_xs"], p["date"], mn)

    mode = pc["standardize"]
    src = {"xs": "D_xs", "firm": "D_firm", "resid": "D_resid"}[mode]
    zsrc = "firm" if mode == "firm" else "xs"
    p["D"] = p[src]
    p["zC"], p["zN"] = p[f"zC_{zsrc}"], p[f"zN_{zsrc}"]
    p["absD"] = p["D"].abs()
    p["D_pos"] = p["D"].clip(lower=0)      # corporate rosier than media
    p["D_neg"] = (-p["D"]).clip(lower=0)   # media rosier than corporate
    p["dD"] = p.groupby("ticker")["D"].diff()
    return p


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_panel(corp: pd.DataFrame, news: pd.DataFrame, prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """corp: scored corporate docs (ticker, accepted_utc, doc_type, items, score).
    news: scored news (ticker, ts [UTC], score)."""
    pc, u = cfg["panel"], cfg["universe"]
    daily, cal = market_panel(prices, cfg)
    sample = cal[(cal >= pd.Timestamp(u["start"])) & (cal <= pd.Timestamp(u["end"]))]
    ends = period_ends(sample, pc["freq"])

    corp = attach_availability(corp[corp["doc_type"].isin(pc["corp_doc_types"])], "accepted_utc", "UTC", cal)
    corp = type_adjust(corp)
    news = attach_availability(news, "ts", "UTC", cal)

    C = ew_stream(corp, ends, "score_adj", pc["corp_max_age_days"], pc["corp_halflife_days"], "C")
    N = ew_stream(news, ends, "score", pc["news_window_days"], pc["news_halflife_days"], "N")
    N.loc[N["n_N"] < pc["min_news"], "N"] = np.nan
    pr = corp[corp["doc_type"] == "press_release"]
    C_pr = ew_stream(pr, ends, "score_adj", pc["corp_max_age_days"], pc["corp_halflife_days"], "C_pr")

    # Earnings-release week: an item-2.02 8-K became available in the last 7 days.
    earn = pr[pr["items"].fillna("").str.contains("2.02", regex=False)]
    E = ew_stream(earn.assign(one=1.0), ends, "one", 7, 1e9, "earn")[["ticker", "date", "n_earn"]]

    grid = pd.MultiIndex.from_product([u["tickers"], ends], names=["ticker", "date"]).to_frame(index=False)
    panel = grid
    for f in (C, N, C_pr[["ticker", "date", "C_pr"]], E):
        panel = panel.merge(f, on=["ticker", "date"], how="left")
    panel["earn_week"] = (panel["n_earn"].fillna(0) > 0).astype(float)
    panel["log_n_news"] = np.log1p(panel["n_N"].fillna(0))
    panel = panel.drop(columns=["n_earn"])
    panel = panel.merge(daily, on=["ticker", "date"], how="left")
    panel = add_divergence(panel, cfg)
    panel["year"] = panel["date"].dt.year
    log.info("Panel: %d firm-periods, %d with divergence", len(panel), panel["D"].notna().sum())
    return panel
