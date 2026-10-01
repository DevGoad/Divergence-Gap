"""Text cleaning, boilerplate removal, and sentence splitting."""
from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup

_WS = re.compile(r"[ \t ]+")
_MULTI_NL = re.compile(r"\n{3,}")

# Safe-harbor / forward-looking-statement language is legally mandated and reads
# as strongly negative to FinBERT ("risks", "uncertainties", "adversely"). It says
# nothing about the firm's current tone, so it is removed before scoring.
_SAFE_HARBOR = re.compile(
    r"(forward[\s-]+looking\s+statements?|safe\s+harbor|cautionary\s+(note|statement)|"
    r"private\s+securities\s+litigation\s+reform\s+act)",
    re.IGNORECASE,
)
_NON_GAAP = re.compile(r"(non[\s-]*gaap|reconciliation\s+of)", re.IGNORECASE)

# Sentence boundary: punctuation followed by whitespace and an upper-case / digit start,
# avoiding common abbreviations.
_ABBREVS = ["Inc", "Corp", "Co", "Ltd", "No", "U.S", "Mr", "Ms", "Dr", "vs", "St", "Jan", "Feb", "Aug", "Sept", "Oct", "Nov", "Dec"]
_ABBREV = "".join(rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREVS)
_SENT_SPLIT = re.compile(r"(?<=[.!?])" + _ABBREV + r"\s+(?=[\"'(\[]?[A-Z0-9])")


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    # Hidden XBRL header blocks in inline-XBRL filings.
    for tag in soup.find_all(["ix:header"]):
        tag.decompose()
    for br in soup.find_all(["br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "table"]):
        br.append("\n")
    text = soup.get_text(" ")
    return normalize(text)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r", "\n")
    text = "\n".join(_WS.sub(" ", ln).strip() for ln in text.split("\n"))
    return _MULTI_NL.sub("\n\n", text).strip()


def _is_tabular(line: str) -> bool:
    if not line:
        return False
    chars = [c for c in line if not c.isspace()]
    if not chars:
        return False
    numeric = sum(c.isdigit() or c in "$%(),.-—–" for c in chars)
    return numeric / len(chars) > 0.4


def strip_boilerplate(text: str) -> str:
    """Remove safe-harbor sections, financial tables and non-GAAP reconciliations."""
    lines = [ln for ln in text.split("\n") if not _is_tabular(ln)]
    text = "\n".join(lines)
    # Safe-harbor sections usually sit at the end of a press release: truncate there
    # if the first match occurs after the first 30% of the text, otherwise drop
    # only the paragraphs that contain it.
    m = _SAFE_HARBOR.search(text)
    if m and m.start() > 0.3 * len(text):
        text = text[: m.start()]
    paras = [p for p in re.split(r"\n\s*\n", text) if not _SAFE_HARBOR.search(p)]
    text = "\n\n".join(paras)
    m = _NON_GAAP.search(text)
    if m and m.start() > 0.5 * len(text):
        text = text[: m.start()]
    return text.strip()


def split_sentences(text: str, min_words: int = 5, max_words: int = 120) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        for s in _SENT_SPLIT.split(para):
            s = s.strip()
            n = len(s.split())
            if n < min_words:
                continue
            if n > max_words:
                words = s.split()
                for i in range(0, n, max_words):
                    chunk = " ".join(words[i : i + max_words])
                    if len(chunk.split()) >= min_words:
                        out.append(chunk)
            else:
                out.append(s)
    return out


_MDNA_START = re.compile(
    r"item\s*(2|7)\s*[.:\-–—]?\s*management[’'`s\s]*\s*discussion\s+and\s+analysis", re.IGNORECASE
)
_MDNA_END = re.compile(
    r"item\s*(7a|8|3|4)\s*[.:\-–—]?\s*(quantitative\s+and\s+qualitative|financial\s+statements|controls\s+and\s+procedures)",
    re.IGNORECASE,
)


def extract_mdna(text: str, max_chars: int = 300_000) -> str:
    """Extract the MD&A section from a 10-K/10-Q.

    The section title also appears in the table of contents, so every start match
    is paired with the next end match and the longest span wins.
    """
    best = ""
    for s in _MDNA_START.finditer(text):
        e = _MDNA_END.search(text, s.end())
        end = e.start() if e else min(len(text), s.end() + max_chars)
        span = text[s.end() : end]
        if len(span) > len(best):
            best = span
    return best[:max_chars].strip()


# --- News headline filters -------------------------------------------------

_ROUNDUP = re.compile(
    r"(stocks?\s+(that|to watch|moving|hit|making)|biggest\s+movers|mid[\s-]?day|pre[\s-]?market|"
    r"after[\s-]?hours|52[\s-]?week|price\s+target\s+changes|^\d+\s+stocks|top\s+\d+|"
    r"earnings\s+(scheduled|preview)s?\s+for|options\s+activity|unusual\s+options|"
    r"benzinga'?s?\s+(top|pro)|movers?\s+&?\s*shakers|market\s+(wrap|recap))",
    re.IGNORECASE,
)
_PRICE_RECAP = re.compile(
    r"(shares\s+(are\s+)?(trading|moving|down|up|higher|lower)|trading\s+(higher|lower)|"
    r"stock\s+is\s+(down|up|moving)|\bshares\s+(fell|rose|jumped|slid|dropped|surged|dip|spike|tick|hit)|"
    r"on\s+volume\s+spike|over\s+last\s+few\s+min|halted|circuit\s+breaker)",
    re.IGNORECASE,
)


def is_roundup(title: str) -> bool:
    return bool(_ROUNDUP.search(title or ""))


def is_price_recap(title: str, url: str = "") -> bool:
    return "/wiim/" in (url or "") or bool(_PRICE_RECAP.search(title or ""))


_CORP_SUFFIX = re.compile(r"\b(inc|incorporated|corp|corporation|co|company|companies|ltd|plc|holdings?|group|"
                          r"the|& ?co|n\.?v|s\.?a|ag|class [a-c])\b\.?", re.IGNORECASE)


def name_keywords(title: str) -> list[str]:
    """Distinctive words of a registered company name, e.g. 'JPMORGAN CHASE & CO' -> ['jpmorgan']."""
    words = [w for w in re.split(r"[^a-z0-9&'.-]+", _CORP_SUFFIX.sub(" ", title.lower())) if w and w != "&"]
    if not words:
        return []
    kw = words[0].strip(".'")
    if len(kw) < 4 and len(words) > 1:  # e.g. "3M", "AT&T" handled by ticker; "Bank of America"
        kw = " ".join(words[:3])
    return [kw]


def mentions_company(title: str, ticker: str, keywords: list[str]) -> bool:
    t = title or ""
    # Case-sensitive ticker match; one-letter tickers (C, F, T) are too ambiguous.
    if len(ticker) > 1 and re.search(rf"(?<![A-Za-z]){re.escape(ticker)}(?![A-Za-z])", t):
        return True
    low = t.lower()
    return any(re.search(rf"\b{re.escape(k)}", low) for k in keywords)
