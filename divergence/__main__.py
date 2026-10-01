"""Command-line pipeline.

    python -m divergence fetch-prices | fetch-sec | fetch-news | score | panel | analyze | all
    python -m divergence demo          # synthetic end-to-end run, no downloads
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from . import edgar, models, news, prices, report
from .config import get_paths, load_config
from .panel import build_panel, market_panel

log = logging.getLogger("divergence")


def _scorer(cfg, paths):
    from .finbert import FinBERTScorer

    f = cfg["finbert"]
    return FinBERTScorer(f["model"], f["device"], f["batch_size"], f["max_length"],
                         cache_path=paths.interim / "finbert_sentence_cache.parquet")


def cmd_fetch_prices(cfg, paths):
    df = prices.fetch_prices(cfg, paths.raw)
    log.info("Prices: %d rows, %d tickers", len(df), df["ticker"].nunique())


def cmd_fetch_sec(cfg, paths):
    df = edgar.fetch_all(cfg, paths.raw)
    df.to_parquet(paths.interim / "corp_docs.parquet", index=False)
    log.info("SEC docs: %s", df.groupby("doc_type").size().to_dict() if len(df) else 0)


def cmd_fetch_news(cfg, paths):
    df = news.fetch_all(cfg, paths.raw)
    df.to_parquet(paths.interim / "news_docs.parquet", index=False)
    log.info("News: %d headlines, %d tickers", len(df), df["ticker"].nunique())


def cmd_score(cfg, paths):
    from .finbert import score_documents

    f = cfg["finbert"]
    scorer = _scorer(cfg, paths)
    corp = pd.read_parquet(paths.interim / "corp_docs.parquet")
    sc = score_documents(corp, scorer, split=True, clean=True, min_words=f["min_words"],
                         max_words=f["max_words"], desc="FinBERT corporate")
    sc.to_parquet(paths.processed / "corp_scored.parquet", index=False)
    nw = pd.read_parquet(paths.interim / "news_docs.parquet")
    sn = score_documents(nw, scorer, split=False, clean=False, desc="FinBERT news")
    sn.to_parquet(paths.processed / "news_scored.parquet", index=False)
    log.info("Scored %d corporate docs and %d headlines", len(sc), len(sn))


def cmd_panel(cfg, paths):
    corp = pd.read_parquet(paths.processed / "corp_scored.parquet")
    nw = pd.read_parquet(paths.processed / "news_scored.parquet")
    px = pd.read_parquet(paths.raw / "prices.parquet")
    panel = build_panel(corp, nw, px, cfg)
    panel.to_parquet(paths.processed / "panel.parquet", index=False)


def cmd_analyze(cfg, paths, panel=None, px=None, out_dir=None, title="Divergence Gap - results"):
    panel = panel if panel is not None else pd.read_parquet(paths.processed / "panel.parquet")
    px = px if px is not None else pd.read_parquet(paths.raw / "prices.parquet")
    daily, _ = market_panel(px, cfg)
    res = models.run_all(panel, daily, cfg)
    path = report.write_report(res, panel, cfg, out_dir or paths.results, title)
    log.info("Report written to %s", path)


def cmd_demo(cfg, paths, effect: float):
    from .synthetic import make_world, synthetic_config

    cfg = synthetic_config(cfg)
    cfg["analysis"]["n_permutations"] = min(cfg["analysis"]["n_permutations"], 50)
    px, corp, nw = make_world(effect=effect)
    panel = build_panel(corp, nw, px, cfg)
    tag = f"demo_effect_{effect:g}"
    cmd_analyze(cfg, paths, panel, px, paths.results / tag, f"Synthetic demo (planted effect = {effect:g})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="divergence", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch-prices", "fetch-sec", "fetch-news", "score", "panel",
                                        "analyze", "all", "demo"])
    ap.add_argument("--config", default=None, help="YAML file overriding config/default.yaml")
    ap.add_argument("--effect", type=float, default=1.0, help="demo: planted effect size (0 = null world)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    paths = get_paths(cfg)
    steps = {"fetch-prices": cmd_fetch_prices, "fetch-sec": cmd_fetch_sec, "fetch-news": cmd_fetch_news,
             "score": cmd_score, "panel": cmd_panel, "analyze": cmd_analyze}
    if a.command == "demo":
        cmd_demo(cfg, paths, a.effect)
    elif a.command == "all":
        for name in ("fetch-prices", "fetch-sec", "fetch-news", "score", "panel", "analyze"):
            log.info("=== %s ===", name)
            steps[name](cfg, paths)
    else:
        steps[a.command](cfg, paths)


if __name__ == "__main__":
    main()
