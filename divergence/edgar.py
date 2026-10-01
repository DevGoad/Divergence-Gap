"""Corporate-voice documents from SEC EDGAR.

* 8-K EX-99.x exhibits (press releases) under items 2.02 / 7.01 / 8.01
* MD&A sections of 10-Q and 10-K filings

Raw text is cached per ticker in ``data/raw/sec/<TICKER>.parquet`` so the
download is resumable at ticker granularity.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from . import text as tx

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}"


class SecClient:
    def __init__(self, user_agent: str, requests_per_second: float = 8):
        if "@" not in user_agent or "example.com" in user_agent:
            raise ValueError("SEC requires a User-Agent containing a contact email: set sec.user_agent in config/default.yaml.")
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = 1.0 / requests_per_second
        self._last = 0.0

    def get(self, url: str, retries: int = 5) -> requests.Response | None:
        for attempt in range(retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.s.get(url, timeout=30)
            except requests.RequestException as e:
                log.warning("GET %s failed (%s), retrying", url, e)
                time.sleep(2**attempt)
                continue
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429) or r.status_code >= 500:
                time.sleep(2 ** (attempt + 1))
                continue
            log.warning("GET %s -> HTTP %s", url, r.status_code)
            return None
        log.error("Giving up on %s", url)
        return None


def ticker_to_cik(client: SecClient) -> dict[str, int]:
    r = client.get(TICKERS_URL)
    if r is None:
        raise RuntimeError("Could not download SEC ticker map")
    data = r.json()
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in data.values()}


def company_titles(client: SecClient) -> dict[str, str]:
    r = client.get(TICKERS_URL)
    if r is None:
        return {}
    return {v["ticker"].upper().replace(".", "-"): v["title"] for v in r.json().values()}


def list_filings(client: SecClient, cik: int) -> pd.DataFrame:
    """All filings for a CIK (recent block + paginated history files)."""
    r = client.get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
    if r is None:
        return pd.DataFrame()
    sub = r.json()
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    for f in sub["filings"].get("files", []):
        rr = client.get(SUBMISSIONS_URL.format(name=f["name"]))
        if rr is not None:
            frames.append(pd.DataFrame(rr.json()))
    df = pd.concat(frames, ignore_index=True)
    keep = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items", "primaryDocument"]
    return df[[c for c in keep if c in df.columns]]


def _exhibit_urls(client: SecClient, cik: int, acc: str) -> list[str]:
    base = ARCHIVE.format(cik=cik, acc_nodash=acc.replace("-", ""))
    r = client.get(f"{base}/{acc}-index.htm")
    if r is None:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    urls = []
    for row in soup.select("table.tableFile tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        doc_type = cells[3].get_text(strip=True).upper()
        link = cells[2].find("a")
        if link and doc_type.startswith("EX-99"):
            href = link["href"]
            if href.startswith("/ix?doc="):
                href = href[len("/ix?doc=") :]
            urls.append("https://www.sec.gov" + href if href.startswith("/") else f"{base}/{href}")
    return urls


def _fetch_text(client: SecClient, url: str) -> str:
    r = client.get(url)
    if r is None:
        return ""
    # EDGAR often omits the charset; older filings are cp1252, newer ones UTF-8.
    try:
        body = r.content.decode("utf-8")
    except UnicodeDecodeError:
        body = r.content.decode("cp1252", errors="replace")
    if url.lower().endswith((".htm", ".html", ".xml")) or "<html" in body[:2000].lower():
        return tx.html_to_text(body)
    return tx.normalize(body)


def _accepted_utc(df: pd.DataFrame) -> pd.Series:
    # acceptanceDateTime is UTC (Apple's 16:30 ET earnings 8-Ks show as 20:30Z).
    # Missing values fall back to filingDate 17:30 ET (after the close), which is conservative.
    acc = pd.to_datetime(df["acceptanceDateTime"], errors="coerce", utc=True)
    fallback = (pd.to_datetime(df["filingDate"]) + pd.Timedelta(hours=17, minutes=30))         .dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return acc.fillna(fallback)


def fetch_ticker(client: SecClient, ticker: str, cik: int, cfg: dict) -> pd.DataFrame:
    start, end = pd.Timestamp(cfg["universe"]["start"]), pd.Timestamp(cfg["universe"]["end"])
    sec = cfg["sec"]
    filings = list_filings(client, cik)
    if filings.empty:
        return pd.DataFrame()
    filings["accepted_utc"] = _accepted_utc(filings)
    # One quarter of lead-in so corporate sentiment is populated at the sample start.
    acc = filings["accepted_utc"].dt.tz_localize(None)
    filings = filings[(acc >= start - pd.Timedelta(days=150)) & (acc <= end + pd.Timedelta(days=1))]

    rows = []
    items_wanted = set(sec["press_release_items"])
    eightk = filings[filings["form"].isin(["8-K", "8-K/A"])]
    eightk = eightk[eightk["items"].fillna("").apply(lambda s: bool(items_wanted & set(s.split(","))))]
    for _, f in eightk.iterrows():
        for url in _exhibit_urls(client, cik, f["accessionNumber"]):
            body = _fetch_text(client, url)
            if len(body) < 200:
                continue
            rows.append(dict(accession=f["accessionNumber"], form=f["form"], items=f["items"],
                             doc_type="press_release", accepted_utc=f["accepted_utc"], url=url,
                             text=body[: sec["max_doc_chars"]]))

    if sec.get("include_mdna", True):
        periodic = filings[filings["form"].isin(["10-Q", "10-K"])]
        for _, f in periodic.iterrows():
            if not f.get("primaryDocument"):
                continue
            url = ARCHIVE.format(cik=cik, acc_nodash=f["accessionNumber"].replace("-", "")) + "/" + f["primaryDocument"]
            mdna = tx.extract_mdna(_fetch_text(client, url), sec["max_doc_chars"])
            if len(mdna) < 1000:
                log.debug("%s %s: MD&A not found", ticker, f["accessionNumber"])
                continue
            rows.append(dict(accession=f["accessionNumber"], form=f["form"], items="",
                             doc_type="mdna_10k" if f["form"] == "10-K" else "mdna_10q",
                             accepted_utc=f["accepted_utc"], url=url, text=mdna))

    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(0, "ticker", ticker)
        out.insert(1, "cik", cik)
    return out


def fetch_all(cfg: dict, raw_dir: Path) -> pd.DataFrame:
    sec_dir = raw_dir / "sec"
    sec_dir.mkdir(parents=True, exist_ok=True)
    client = SecClient(cfg["sec"]["user_agent"], cfg["sec"]["requests_per_second"])
    cik_map = ticker_to_cik(client)
    frames = []
    for t in tqdm(cfg["universe"]["tickers"], desc="SEC filings"):
        path = sec_dir / f"{t}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        cik = cik_map.get(t.upper())
        if cik is None:
            log.warning("No CIK for %s; skipping", t)
            continue
        df = fetch_ticker(client, t, cik, cfg)
        df.to_parquet(path, index=False)
        frames.append(df)
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
