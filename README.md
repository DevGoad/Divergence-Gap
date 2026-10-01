# The Divergence Gap: Corporate vs. Media Sentiment

**Research question.** Does the divergence between corporate-issued sentiment and
independent media sentiment, as measured by FinBERT, serve as a statistically
significant leading indicator of stock price volatility and anomalous drawdowns?

## Design

```
SEC EDGAR ──► 8-K EX-99 press releases ─┐                         ┌─► forward realized / idiosyncratic vol
          └─► 10-K / 10-Q MD&A ─────────┤  FinBERT (sentence-level) │   forward (abnormal) max drawdown
                                        ├─► C_it  ┐                 │   anomalous-drawdown flag
FNSPID news headlines ──────────────────┴─► N_it  ┴─► D_it = z(C) − z(N) ─► panel tests (weekly)
Yahoo Finance prices ───────────────────────────────────────────────┘
```

### Sentiment streams
* **Document score** = mean over sentences of `P(positive) − P(negative)` from
  `ProsusAI/finbert`. Long filings are sentence-split (not truncated at 512 tokens).
  Safe-harbor / forward-looking-statement boilerplate, financial tables and trailing
  non-GAAP reconciliations are stripped first; otherwise every press release would
  get a mechanical negative tail.
* **Corporate tone C** — exponentially weighted (half-life 45 days, max age 120 days)
  mean of *type-adjusted* scores: each doc type (press release, 10-Q MD&A, 10-K MD&A)
  is demeaned by its expanding cross-firm average, computed only from earlier documents.
* **Media tone N** — EW mean (half-life 7 days, 28-day window, at least 3 articles) of
  headline scores. Roundups ("30 stocks moving pre-market") and price recaps
  ("shares are trading higher after…") are removed: they describe the price move
  rather than independent information, which would create reverse causality.

### Divergence
`D = z(C) − z(N)`, cross-sectionally standardized each week (alternatives:
within-firm expanding z, or the residual of z(C) on z(N); see `panel.standardize`).
**D > 0 means the company sounds rosier than the press.** The pipeline also builds
`|D|`, `D_pos = max(D, 0)` and `D_neg = max(−D, 0)`.

### Timing (no look-ahead)
A document becomes usable on the first trading close at or after publication
(after 16:00 ET → next session). Features at decision date T (last trading day of
the week) use only information through T's close. Targets use returns from T+1 to T+h.
`tests/test_pipeline.py` checks this by perturbing future data.

### Targets (h = 5, 21, 63 trading days)
| Column | Meaning |
|---|---|
| `fwd_log_rv_h` | log annualized realized volatility, T+1..T+h |
| `fwd_log_idio_h` | same for market-model residuals (beta estimated through T) |
| `fwd_dd_h` | max drawdown depth of the price path (positive = worse) |
| `fwd_abdd_h` | max drawdown depth of the cumulative abnormal-return path |
| `anom_dd_h` | 1 if abnormal drawdown exceeds `k·σ_idio·√h` (k = 2), with σ estimated through T |

### Tests (`divergence/models.py`)
1. **Panel OLS** with firm and week fixed effects, using Driscoll–Kraay SEs (robust
   to cross-sectional and serial correlation, including overlapping targets) and
   two-way clustered SEs. Controls: HAR volatility (`log_rv_5/21/63`), 1-month return,
   12-1 momentum, beta, log dollar volume, news count, and an earnings-week dummy.
   * S1: D. S2: D + |D| + media tone (does divergence add beyond media tone?).
     S3: D_pos + D_neg + media tone (asymmetry).
2. **Logit** of anomalous drawdowns, with year dummies, VIX and firm-clustered SEs
   (odds ratios and marginal effects).
3. **Dumitrescu–Hurlin panel Granger causality**, in both directions
   (D → vol and vol → D).
4. **Quintile sorts** on D, with Newey–West t-statistics on the Q5−Q1 spread.
5. **Out-of-sample** expanding window by year, with an h-day embargo: the base model
   (controls + media tone) vs. the augmented model (+ D, |D|). Reports OOS R², the
   Clark–West nested-model test and drawdown-classification AUC.
6. **Placebo**: D is permuted across firms within each week (empirical p-value).
7. **Event study**: cumulative abnormal returns and |abnormal returns| after a firm
   enters the top or bottom decile of D.

## Running

```bash
pip install -r requirements.txt

python -m divergence demo               # synthetic world with a planted effect (~1 min, no downloads)
python -m divergence demo --effect 0    # null world: should find nothing
python -m pytest tests -q

python -m divergence fetch-prices       # Yahoo Finance
python -m divergence fetch-sec          # EDGAR; resumable per ticker; ~1 min per ticker-year at 8 req/s
python -m divergence fetch-news         # downloads FNSPID All_external.csv (5.7 GB) once, then filters
python -m divergence score              # FinBERT on GPU, with a sentence-hash cache
python -m divergence panel
python -m divergence analyze            # writes results/REPORT.md, CSVs and figures
# or: python -m divergence all
```

Use your own settings with `--config my.yaml`; it is merged over `config/default.yaml`.
To use another news vendor (Finnhub, Alpha Vantage, Refinitiv, …), export a CSV with
`ticker,timestamp,text` (UTC timestamps) and set `news.source: csv`.

## Known limitations and threats to validity
* **Survivorship bias**: the default universe is today's large caps. For publication,
  use historical index constituents.
* **FNSPID coverage** varies by year and is concentrated in Benzinga/Nasdaq feeds.
  Headline-only sentiment is noisier than full-text sentiment. The corpus ends in 2023.
* **MD&A extraction** is heuristic. Some filers (e.g. JPM) put MD&A in an EX-13
  annual-report exhibit, so for them C relies on press releases (`C_pr` is also in
  the panel).
* **FinBERT domain shift**: it was trained on Financial PhraseBank (news-style
  sentences), and MD&A language differs. Type-adjustment removes level differences,
  but not differences in scale.
* Media tone partly *reacts* to price moves even after the filters. The models control
  for lagged returns and volatility, and the reverse Granger test quantifies feedback.
