# Quick Run Guide

Background on the research design is in `README.md`. This page only covers how to run it.

## 1. Setup (one time)

You need Python 3.10+, and preferably an NVIDIA GPU for FinBERT (a CPU works, but slowly).

```bash
cd divergence-gap
pip install -r requirements.txt
```
For GPU support, install the CUDA build of PyTorch first. See https://pytorch.org/get-started/locally/

**Required:** in `config/default.yaml`, set your name and email:
```yaml
sec:
  user_agent: "Jane Doe jane@university.edu"
```
SEC EDGAR rejects requests that don't include a contact email.

## 2. Check that everything works (about 2 minutes, no downloads)

```bash
python -m pytest tests -q                # should report 7 passed
python -m divergence demo                # synthetic data with a built-in effect -> results/demo_effect_1/REPORT.md
python -m divergence demo --effect 0     # synthetic data with no effect -> nothing should be significant
```

## 3. Run on real data

Start with a small pilot. Create `pilot.yaml`:
```yaml
universe:
  tickers: [AAPL, MSFT, JPM, XOM, PFE, WMT, BA, INTC, KO, DIS]
  start: "2016-01-01"
  end: "2020-12-31"
```
Then run each step in order:
```bash
python -m divergence fetch-prices --config pilot.yaml
python -m divergence fetch-sec    --config pilot.yaml   # slowest step, ~1 min per ticker-year; restarting resumes where it stopped
python -m divergence fetch-news   --config pilot.yaml   # first run downloads FNSPID (5.7 GB) to data/raw/
python -m divergence score        --config pilot.yaml   # FinBERT
python -m divergence panel        --config pilot.yaml
python -m divergence analyze      --config pilot.yaml
```
Or run everything at once with `python -m divergence all --config pilot.yaml`.

To run the full study, drop `--config` to use the default 60 tickers over 2010–2023. Expect more than 12 hours for the SEC download.

## 4. Outputs (`results/`)

- `REPORT.md`: all tables and figures.
- `regressions.csv`: the main test. Look at the `D`, `absD` and `D_pos` coefficients and their `t_dk` values (|t| > 1.96 means significant at 5%).
- `granger.csv`, `sorts.csv`, `oos.csv`, `logit.csv`, `permutation.csv`, `event_study.csv`.
- `fig_*.png`.

**D > 0** means the company sounds rosier than the media. Drawdown targets are positive numbers, so a positive coefficient means deeper drawdowns.

## 5. Common issues

- **`ValueError: SEC requires a User-Agent`**: set `sec.user_agent` (see step 1).
- **HTTP 403/429 from SEC**: lower `sec.requests_per_second` in the config.
- **CUDA out of memory**: lower `finbert.batch_size`.
- **Using a different news source**: provide a CSV with the columns `ticker,timestamp,text` (timestamps in UTC) and set `news.source: csv` and `news.csv_path`.
- **Changing settings**: edit the config and rerun from `panel` onward. Data that was already downloaded or scored is cached in `data/`.
