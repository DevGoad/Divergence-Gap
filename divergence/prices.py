"""Daily prices from Yahoo Finance (adjusted closes and volume)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def fetch_prices(cfg: dict, raw_dir: Path) -> pd.DataFrame:
    """Long frame: date, ticker, close, volume. Includes the market proxy and VIX."""
    u = cfg["universe"]
    path = raw_dir / "prices.parquet"
    tickers = list(dict.fromkeys(u["tickers"] + [u["market_ticker"], u["vix_ticker"]]))
    if path.exists():
        cached = pd.read_parquet(path)
        if set(tickers) <= set(cached["ticker"].unique()):
            return cached
    start = (pd.Timestamp(u["start"]) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")  # warm-up for betas
    end = (pd.Timestamp(u["end"]) + pd.DateOffset(months=4)).strftime("%Y-%m-%d")      # forward targets
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True,
                      group_by="column", threads=True)
    close = raw["Close"].stack().rename("close")
    volume = raw["Volume"].stack().rename("volume")
    df = pd.concat([close, volume], axis=1).reset_index()
    df.columns = ["date", "ticker", "close", "volume"]
    df = df.dropna(subset=["close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    missing = set(tickers) - set(df["ticker"].unique())
    if missing:
        log.warning("No price data for: %s", sorted(missing))
    df.to_parquet(path, index=False)
    return df
