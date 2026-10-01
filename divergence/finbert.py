"""FinBERT sentiment scoring with a persistent sentence-level cache.

Each sentence (or headline) gets probabilities (p_pos, p_neg, p_neu). A document's
score is the mean over its sentences of ``p_pos - p_neg`` (range -1..1). Long
filings are sentence-split before scoring rather than truncated at 512 tokens, so
the score covers the whole document.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import text as tx

log = logging.getLogger(__name__)


def _key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:20]


class FinBERTScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "auto",
                 batch_size: int = 128, max_length: int = 128, cache_path: Path | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        if device == "cuda":
            self.model.half()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.order = [labels["positive"], labels["negative"], labels["neutral"]]
        self.batch_size, self.max_length = batch_size, max_length
        self.cache_path = cache_path
        self.cache: dict[str, np.ndarray] = {}
        if cache_path and cache_path.exists():
            c = pd.read_parquet(cache_path)
            self.cache = dict(zip(c["key"], c[["p_pos", "p_neg", "p_neu"]].to_numpy(np.float32)))
            log.info("Loaded %d cached sentence scores", len(self.cache))

    @torch.inference_mode()
    def _score_batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_length,
                       return_tensors="pt").to(self.device)
        probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
        return probs[:, self.order]

    def score(self, texts: list[str], desc: str = "FinBERT") -> np.ndarray:
        """Probabilities (n, 3) ordered pos/neg/neu; cached by sentence hash."""
        keys = [_key(t) for t in texts]
        todo = {k: t for k, t in zip(keys, texts) if k not in self.cache}
        if todo:
            items = sorted(todo.items(), key=lambda kv: len(kv[1]))  # length-bucketing
            for i in tqdm(range(0, len(items), self.batch_size), desc=desc):
                batch = items[i : i + self.batch_size]
                probs = self._score_batch([t for _, t in batch])
                for (k, _), p in zip(batch, probs):
                    self.cache[k] = p
            self.save_cache()
        return np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, 3), np.float32)

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        keys = list(self.cache)
        arr = np.stack([self.cache[k] for k in keys])
        pd.DataFrame({"key": keys, "p_pos": arr[:, 0], "p_neg": arr[:, 1], "p_neu": arr[:, 2]}) \
            .to_parquet(self.cache_path, index=False)


def score_documents(df: pd.DataFrame, scorer: FinBERTScorer, *, split: bool, clean: bool,
                    min_words: int = 5, max_words: int = 120, desc: str = "FinBERT") -> pd.DataFrame:
    """Add sentence-averaged sentiment columns to a frame with a ``text`` column.

    Returns a copy with: score (mean p_pos-p_neg), p_pos, p_neg, p_neu, n_sent,
    net_tone ((#pos - #neg) / #sentences by argmax label).
    """
    doc_ids, sents = [], []
    for i, t in enumerate(df["text"].fillna("")):
        if clean:
            t = tx.strip_boilerplate(t)
        ss = tx.split_sentences(t, min_words, max_words) if split else ([t.strip()] if t.strip() else [])
        doc_ids.extend([i] * len(ss))
        sents.extend(ss)
    probs = scorer.score(sents, desc=desc)
    s = pd.DataFrame(probs, columns=["p_pos", "p_neg", "p_neu"])
    s["doc"] = doc_ids
    s["polarity"] = s["p_pos"] - s["p_neg"]
    lab = probs.argmax(axis=1) if len(probs) else np.array([], int)
    s["is_pos"], s["is_neg"] = lab == 0, lab == 1
    agg = s.groupby("doc").agg(score=("polarity", "mean"), p_pos=("p_pos", "mean"),
                               p_neg=("p_neg", "mean"), p_neu=("p_neu", "mean"),
                               n_sent=("polarity", "size"), n_pos=("is_pos", "sum"), n_neg=("is_neg", "sum"))
    agg["net_tone"] = (agg["n_pos"] - agg["n_neg"]) / agg["n_sent"]
    out = df.reset_index(drop=True).drop(columns=["text"]).join(agg.drop(columns=["n_pos", "n_neg"]))
    return out.dropna(subset=["score"])
