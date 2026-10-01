"""Trading-calendar alignment: when does a document first become tradable?"""
from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"
CLOSE_HOUR = 16


def to_eastern(ts: pd.Series, source_tz: str) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce", utc=(source_tz == "UTC"))
    if source_tz == "UTC":
        return ts.dt.tz_convert(ET).dt.tz_localize(None)
    if ts.dt.tz is not None:
        return ts.dt.tz_convert(ET).dt.tz_localize(None)
    return ts  # already naive Eastern


def availability_date(ts_eastern: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """First trading day whose close can reflect a document published at ``ts_eastern``.

    Published at/after the 16:00 ET close -> next trading day; non-trading days roll
    forward. Documents after the last calendar day get NaT.
    """
    ts = pd.to_datetime(ts_eastern)
    day = ts.dt.normalize()
    after_close = ts.dt.hour >= CLOSE_HOUR
    day = day + pd.to_timedelta(after_close.astype(int), unit="D")
    days = trading_days.values.astype("datetime64[ns]")
    idx = np.searchsorted(days, day.values.astype("datetime64[ns]"), side="left")
    out = np.full(len(idx), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = (idx < len(days)) & ~pd.isna(day.values)
    out[ok] = days[idx[ok]]
    return pd.Series(out, index=ts_eastern.index)


def period_ends(trading_days: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last trading day of each calendar period (e.g. W-FRI)."""
    s = pd.Series(trading_days, index=trading_days)
    return pd.DatetimeIndex(s.groupby(s.index.to_period(freq)).max().values)
