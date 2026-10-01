"""Independent-media documents (financial news headlines).

Default source is FNSPID (Dong et al., 2024): ~15.7M ticker-tagged financial news
items, 1999-2023, hosted on Hugging Face. The 5.7 GB CSV is downloaded once and
filtered to the universe in chunks. Any other source can be used through
``news.source: csv`` with columns ``ticker, timestamp, text[, publisher, url]``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from . import text as tx

log = logging.getLogger(__name__)


def _download(url: str, dest: Path) -> None:
    """Resumable streaming download."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:  # already complete
            part.rename(dest)
            return
        r.raise_for_status()
        if have and r.status_code != 206:
            have = 0  # server ignored the range; restart
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        with open(part, mode) as f, tqdm(total=total, initial=have, unit="B", unit_scale=True,
                                         desc=f"Downloading {dest.name}") as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    part.rename(dest)


def load_fnspid(cfg: dict, raw_dir: Path) -> pd.DataFrame:
    ncfg = cfg["news"]
    src = raw_dir / "fnspid_all_external.csv"
    if not src.exists():
        _download(ncfg["fnspid_url"], src)
    aliases = {k.upper(): v.upper() for k, v in (ncfg.get("ticker_aliases") or {}).items()}
    tickers = {t.upper() for t in cfg["universe"]["tickers"]} | set(aliases)
    cols = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Lsa_summary"]
    keep = []
    reader = pd.read_csv(src, usecols=cols, chunksize=500_000, dtype=str, on_bad_lines="skip")
    for chunk in tqdm(reader, desc="Filtering FNSPID"):
        chunk = chunk[chunk["Stock_symbol"].str.upper().isin(tickers)]
        if not chunk.empty:
            keep.append(chunk)
    df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=cols)
    title = df["Article_title"].fillna("")
    text = title
    if ncfg.get("text_field") == "title_summary":
        text = (title + ". " + df["Lsa_summary"].fillna("")).str.strip(". ")
    return pd.DataFrame({
        "ticker": df["Stock_symbol"].str.upper().replace(aliases),
        "timestamp_utc": df["Date"],
        "title": title,
        "text": text,
        "publisher": df["Publisher"],
        "url": df["Url"],
    })


def load_csv(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["news"]["csv_path"])
    required = {"ticker", "timestamp", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"news CSV missing columns: {missing}")
    df = df.rename(columns={"timestamp": "timestamp_utc"})
    df["ticker"] = df["ticker"].str.upper()
    df["title"] = df.get("title", df["text"])
    for c in ("publisher", "url"):
        if c not in df:
            df[c] = ""
    return df[["ticker", "timestamp_utc", "title", "text", "publisher", "url"]]


def company_keywords(cfg: dict) -> dict[str, list[str]]:
    """Name keywords per ticker from SEC registrant names, overridable via news.name_keywords."""
    from .edgar import SecClient, company_titles

    titles = company_titles(SecClient(cfg["sec"]["user_agent"], cfg["sec"]["requests_per_second"]))
    kw = {t: tx.name_keywords(titles.get(t, "")) for t in cfg["universe"]["tickers"]}
    for t, words in (cfg["news"].get("name_keywords") or {}).items():
        kw[t.upper()] = [w.lower() for w in words]
    return kw


def clean_news(df: pd.DataFrame, cfg: dict, keywords: dict[str, list[str]] | None = None) -> pd.DataFrame:
    ncfg = cfg["news"]
    start, end = pd.Timestamp(cfg["universe"]["start"]), pd.Timestamp(cfg["universe"]["end"])
    n0 = len(df)
    df = df[df["ticker"].isin([t.upper() for t in cfg["universe"]["tickers"]])]
    df = df[df["text"].fillna("").str.split().str.len() >= 4].copy()
    df["category"] = "news"
    if ncfg.get("drop_roundups", True):
        df.loc[df["title"].apply(tx.is_roundup), "category"] = "roundup"
    if ncfg.get("drop_price_recaps", True):
        recap = [tx.is_price_recap(t, u) for t, u in zip(df["title"], df["url"].fillna(""))]
        df.loc[pd.Series(recap, index=df.index) & (df["category"] == "news"), "category"] = "price_recap"
    if keywords is not None:
        # Vendor ticker tags are noisy (FNSPID tags Alcoa stories with "A"): keep
        # only headlines that name the company or its ticker.
        hit = [tx.mentions_company(t, tk, keywords.get(tk, [])) for t, tk in zip(df["title"], df["ticker"])]
        df.loc[~pd.Series(hit, index=df.index) & (df["category"] == "news"), "category"] = "off_target"
    log.info("News categories: %s", df["category"].value_counts().to_dict())
    df = df[df["category"] == "news"]
    ts = pd.to_datetime(df["timestamp_utc"].str.replace(" UTC", "", regex=False), errors="coerce", utc=True)
    df = df.assign(ts=ts).dropna(subset=["ts"])
    df = df[(df["ts"] >= pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=60)) &
            (df["ts"] <= pd.Timestamp(end, tz="UTC"))]
    # Syndicated duplicates: same headline for the same firm on the same day.
    df = df.assign(day=df["ts"].dt.date).drop_duplicates(["ticker", "day", "title"]).drop(columns="day")
    log.info("News: %d raw -> %d kept after filtering", n0, len(df))
    return df.reset_index(drop=True)


def fetch_all(cfg: dict, raw_dir: Path) -> pd.DataFrame:
    src = cfg["news"]["source"]
    if src == "fnspid":
        df = load_fnspid(cfg, raw_dir)
    elif src == "csv":
        df = load_csv(cfg)
    else:
        raise ValueError(f"unknown news.source {src!r}")
    kw = company_keywords(cfg) if cfg["news"].get("relevance_filter", True) else None
    return clean_news(df, cfg, kw)
