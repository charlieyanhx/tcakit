"""TCALogger adapter: sign mapping reproduces the logger's own IS cents; timestamps and
partial fills map correctly; the private fixture (gitignored) is exercised when present."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcakit.adapters import tcalogger
from tcakit.shortfall import implementation_shortfall

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def frames():
    return tcalogger.load([FIX / "tcalogger_sample.jsonl"])


def test_side_and_price_mapping(frames):
    orders, fills = frames
    o = orders.set_index("order_id")
    assert o.loc["open-1", "side"] == -1 and o.loc["close-1", "side"] == 1
    assert o.loc["open-1", "arrival_px"] == 1.0 and o.loc["close-1", "arrival_px"] == 1.03
    assert (fills["px"] > 0).all() and (fills["mid"] > 0).all()


def test_reproduces_logger_is_cents(frames):
    orders, fills = frames
    sf = implementation_shortfall(orders, fills).set_index("order_id")
    # logger: open −3.0 c, close +7.0 c per contract; execution is per share × contracts
    assert sf.loc["open-1", "execution"] * 100 == pytest.approx(-3.0)
    assert sf.loc["close-1", "execution"] * 100 == pytest.approx(7.0)


def test_partial_fill_and_opportunity(frames):
    orders, fills = frames
    sf = implementation_shortfall(orders, fills).set_index("order_id")
    assert sf.loc["open-1", "filled_qty"] == 1 and orders.set_index("order_id").loc["open-1", "qty"] == 2
    # unfilled 1 contract, mid moved 1.00 → 0.98 on a sell: side −1 · 1 · (0.98 − 1.00) = +0.02
    assert sf.loc["open-1", "opportunity"] == pytest.approx(0.02)
    assert sf.loc["open-1", "delay"] == 0.0


def test_timestamps(frames):
    orders, fills = frames
    o = orders.set_index("order_id")
    assert (o["decision_ts"] == o["arrival_ts"]).all()
    close_arrival = pd.Timestamp("2026-04-24T17:18:16.278+00:00") - pd.Timedelta(milliseconds=17087.5)
    assert abs((o.loc["close-1", "arrival_ts"] - close_arrival).total_seconds()) < 1e-3
    f = fills.set_index("fill_id")
    assert f.loc["close-1/0", "ts"] > o.loc["close-1", "arrival_ts"]


def test_fees_from_argument():
    _, fills = tcalogger.load([FIX / "tcalogger_sample.jsonl"], fee_per_contract_leg=0.65)
    assert fills.set_index("fill_id").loc["close-1/0", "fee"] == pytest.approx(1.30)


def test_zero_mid_rejected():
    rec = tcalogger.read_jsonl([FIX / "tcalogger_sample.jsonl"])[0] | {"arrival_mid": 0.0}
    with pytest.raises(ValueError):
        tcalogger.to_frames([rec])


PRIVATE = sorted(FIX.glob("private_*.jsonl"))


@pytest.mark.skipif(not PRIVATE, reason="private fixture not present")
def test_private_fixture_round_trips_logger_is():
    orders, fills = tcalogger.load(PRIVATE)
    sf = implementation_shortfall(orders, fills).set_index("order_id")
    recs = pd.DataFrame(tcalogger.read_jsonl(PRIVATE))
    logged = recs.dropna(subset=["implementation_shortfall_cents"]).groupby("intent_id")[
        "implementation_shortfall_cents"].sum()
    got = (sf["execution"] * 100).reindex(logged.index)
    assert np.allclose(got, logged, atol=1e-6)
