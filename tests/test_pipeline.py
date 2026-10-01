import numpy as np
import pandas as pd
import pytest

from divergence.calendar import availability_date
from divergence.config import load_config
from divergence.panel import build_panel, ticker_market_features
from divergence.synthetic import make_world, synthetic_config
from divergence import models, text as tx


@pytest.fixture(scope="module")
def world():
    cfg = synthetic_config(load_config(), n_firms=20)
    px, corp, news = make_world(n_firms=20, end="2016-12-31", seed=1)
    cfg["universe"]["end"] = "2016-06-30"
    return cfg, px, corp, news


def test_availability_after_close_rolls_forward():
    cal = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
    ts = pd.Series(pd.to_datetime(["2020-01-02 15:59", "2020-01-02 16:00", "2020-01-03 18:00", "2020-01-04 10:00"]))
    got = availability_date(ts, cal)
    assert list(got.dt.strftime("%m-%d")) == ["01-02", "01-03", "01-06", "01-06"]


def test_forward_targets_use_only_future_returns():
    cfg = load_config()
    days = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(0)
    px = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), "volume": 1e6}, index=days)
    mkt = pd.Series(rng.normal(0, 0.01, 400), index=days)
    vix = pd.Series(15.0, index=days)
    a = ticker_market_features(px, mkt, vix, cfg)
    # Shock a single day; features at or before t0-1 and targets ending before it must not change.
    t0 = 300
    px2 = px.copy()
    px2.iloc[t0:, 0] *= 0.7
    b = ticker_market_features(px2, mkt, vix, cfg)
    feats = ["log_rv_5", "log_rv_21", "log_rv_63", "ret_21", "beta", "sigma_idio"]
    pd.testing.assert_frame_equal(a[feats].iloc[:t0], b[feats].iloc[:t0])
    h = 21
    tg = [f"fwd_log_rv_{h}", f"fwd_dd_{h}", f"fwd_abdd_{h}"]
    pd.testing.assert_frame_equal(a[tg].iloc[: t0 - h - 1], b[tg].iloc[: t0 - h - 1])
    # ...and the target of the day right before the shock does see it.
    assert b[f"fwd_dd_{h}"].iloc[t0 - 1] > a[f"fwd_dd_{h}"].iloc[t0 - 1] + 0.2


def test_sentiment_features_ignore_future_documents(world):
    cfg, px, corp, news = world
    p1 = build_panel(corp, news, px, cfg)
    cut = pd.Timestamp("2015-06-01")
    corp2 = corp.copy()
    corp2.loc[corp2["accepted_utc"] >= cut.tz_localize("America/New_York"), "score"] = -1.0
    news2 = news.copy()
    news2.loc[news2["ts"] >= cut.tz_localize("America/New_York"), "score"] = 1.0
    p2 = build_panel(corp2, news2, px, cfg)
    early = p1["date"] < cut - pd.Timedelta(days=3)
    for c in ("N", "C", "D_raw", "D_firm"):
        np.testing.assert_allclose(p1.loc[early, c].to_numpy(), p2.loc[early, c].to_numpy(), equal_nan=True)


def test_planted_effect_is_recovered_and_null_is_not(world):
    cfg, px, corp, news = world
    panel = build_panel(corp, news, px, cfg)
    r = models.panel_ols(panel, "fwd_log_idio_21", ["D"], 21)
    assert r.loc[0, "coef"] > 0 and r.loc[0, "t_dk"] > 4
    px0, corp0, news0 = make_world(n_firms=20, end="2016-12-31", seed=1, effect=0.0)
    r0 = models.panel_ols(build_panel(corp0, news0, px0, cfg), "fwd_log_idio_21", ["D"], 21)
    assert abs(r0.loc[0, "t_dk"]) < 3


def test_boilerplate_and_mdna():
    pr = ("Acme reports record quarterly revenue, up 20 percent year over year.\n\n"
          "Revenue $ 1,234 $ 1,000 20%\n\n" + "Margins expanded strongly across all segments this quarter. " * 5 +
          "\n\nForward-Looking Statements\nThis release contains risks and uncertainties that could adversely affect results.")
    clean = tx.strip_boilerplate(pr)
    assert "uncertainties" not in clean and "$ 1,234" not in clean and "record quarterly" in clean
    doc = ("Item 7. Management's Discussion and Analysis ..... 25\nItem 8. Financial Statements ..... 40\n"
           "Item 7. Management's Discussion and Analysis\n" + "Sales grew because demand was strong. " * 50 +
           "\nItem 7A. Quantitative and Qualitative Disclosures About Market Risk\n")
    assert tx.extract_mdna(doc).startswith("Sales grew")


def test_news_filters():
    assert tx.is_roundup("30 Stocks Moving in Friday's Pre-Market Session")
    assert tx.is_roundup("Stocks That Hit 52-Week Highs On Friday")
    assert tx.is_price_recap("Agilent shares are trading higher after earnings", "")
    assert not tx.is_roundup("Agilent Raises Full-Year Guidance on Strong Demand")


def test_relevance_filter():
    assert tx.name_keywords("JPMORGAN CHASE & CO") == ["jpmorgan"]
    assert tx.mentions_company("Agilent Raises Guidance", "A", ["agilent"])
    assert not tx.mentions_company("Alcoa Corp For Q3 Has Recorded $37M Restructuring Charge", "A", ["agilent"])
    assert tx.mentions_company("Analyst upgrades AAPL on services growth", "AAPL", ["apple"])
    assert not tx.mentions_company("Morgan Stanley raises price target on Nike", "TGT", ["target corp", "target's"])
    assert tx.is_price_recap("Alcoa Shares Dip Lower Over Last Few Mins, Down ~1.7% On Volume Spike")
