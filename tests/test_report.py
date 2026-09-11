from pathlib import Path

from tcakit.adapters import tcalogger
from tcakit.report import gate_section, render, scorecard_section, shortfall_section
from tcakit.scorecards import scorecard
from tcakit.shortfall import implementation_shortfall

FIX = Path(__file__).parent / "fixtures"


def test_report_renders_and_states_units_and_gate():
    orders, fills = tcalogger.load([FIX / "tcalogger_sample.jsonl"])
    sf = implementation_shortfall(orders, fills)
    sc = scorecard(sf.merge(orders[["order_id", "leg"]]), "leg", "execution")
    md = render("t", [shortfall_section(sf, orders, "quote units × contracts"),
                      scorecard_section("by leg", sc, "quote units"),
                      gate_section(len(orders), 50, "metaorder impact")], {"source": "fixture"})
    assert "0 by construction" in md
    assert "NOT MET" in md and "2 / 50" in md
    assert "| open-1 |" in md and "positive = cost" in md
