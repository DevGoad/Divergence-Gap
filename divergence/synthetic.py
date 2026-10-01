"""Synthetic world with a planted divergence effect, for pipeline validation.

Each firm has a latent weekly "hidden trouble" state h (AR(1)). Management spins
(corporate tone rises with h) while the media reports it (media tone falls with h),
so the divergence spread tracks h. Trouble raises volatility and negative-jump risk
over the *following* weeks. A correct pipeline must recover D -> future risk and
must not find it when the link is switched off (``effect=0``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_world(n_firms: int = 40, start: str = "2011-01-01", end: str = "2019-12-31",
               effect: float = 1.0, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    weeks = days.to_period("W-FRI")
    wk_codes, wk_index = pd.factorize(weeks)
    n_w = len(wk_index)

    rm = rng.normal(0.0003, 0.01, len(days))
    prices = [pd.DataFrame({"date": days, "ticker": "SPY", "close": 100 * np.exp(np.cumsum(rm)), "volume": 1e8})]
    vix = 15 + 5 * np.abs(np.sin(np.arange(len(days)) / 200)) + rng.normal(0, 1, len(days))
    prices.append(pd.DataFrame({"date": days, "ticker": "^VIX", "close": vix, "volume": 0.0}))

    corp_rows, news_rows = [], []
    for i in range(n_firms):
        tk = f"F{i:02d}"
        h = np.zeros(n_w)
        for w in range(1, n_w):
            h[w] = 0.85 * h[w - 1] + rng.normal(0, 0.55)
        # Risk responds to trouble with a one-week lag -> divergence *leads* volatility.
        h_lag = np.concatenate([[0], h[:-1]])[wk_codes]
        base = rng.uniform(0.012, 0.022)
        sig = base * np.exp(0.35 * effect * np.clip(h_lag, -2, 3))
        jump = rng.random(len(days)) < 0.002 * np.exp(1.2 * effect * np.clip(h_lag, -2, 3))
        r = rng.uniform(0.6, 1.4) * rm + sig * rng.standard_t(5, len(days)) / np.sqrt(5 / 3) - jump * rng.uniform(0.03, 0.1, len(days))
        prices.append(pd.DataFrame({"date": days, "ticker": tk, "close": 50 * np.exp(np.cumsum(r)),
                                    "volume": rng.lognormal(15, 0.3, len(days))}))

        corp_tone, media_tone = 0.25 + 0.30 * h, 0.05 - 0.30 * h
        # Corporate: press releases ~monthly, earnings 8-K + 10-Q quarterly.
        for w in range(n_w):
            wk_start = wk_index[w].start_time
            if rng.random() < 0.25:
                ts = wk_start + pd.Timedelta(days=int(rng.integers(0, 5)), hours=float(rng.uniform(7, 19)))
                earn = w % 13 == 0
                corp_rows.append(dict(ticker=tk, accepted_et=ts, doc_type="press_release",
                                      items="2.02,9.01" if earn else "8.01",
                                      score=np.clip(corp_tone[w] + rng.normal(0, 0.15), -1, 1)))
            if w % 13 == 1:
                ts = wk_start + pd.Timedelta(days=2, hours=17)
                corp_rows.append(dict(ticker=tk, accepted_et=ts, doc_type="mdna_10q", items="",
                                      score=np.clip(corp_tone[w] - 0.2 + rng.normal(0, 0.1), -1, 1)))
            for _ in range(rng.poisson(4)):
                ts = (wk_start + pd.Timedelta(days=float(rng.uniform(0, 7)))).tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
                news_rows.append(dict(ticker=tk, ts=ts, score=np.clip(media_tone[w] + rng.normal(0, 0.45), -1, 1)))

    corp = pd.DataFrame(corp_rows)
    corp["accepted_utc"] = corp.pop("accepted_et").dt.tz_localize(
        "America/New_York", ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC")
    return pd.concat(prices, ignore_index=True), corp, pd.DataFrame(news_rows)


def synthetic_config(cfg: dict, n_firms: int = 40) -> dict:
    cfg = dict(cfg)
    cfg["universe"] = dict(cfg["universe"], tickers=[f"F{i:02d}" for i in range(n_firms)],
                           start="2012-01-01", end="2019-06-30", market_ticker="SPY", vix_ticker="^VIX")
    return cfg
