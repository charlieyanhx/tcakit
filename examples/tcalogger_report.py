"""Post-trade report for multi-leg option fills logged by a `TCALogger`-style JSONL.

    python examples/tcalogger_report.py tests/fixtures/tcalogger_sample.jsonl [more.jsonl ...]
    python examples/tcalogger_report.py --out docs/validation-private.md tests/fixtures/private_*.jsonl

Reports implementation shortfall in $/contract, fraction of half-spread, and scorecards by
leg (open/close), escalator tier, and urgency. Prints the n-gate for any impact-model fit.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tcakit import __version__, contract_costs, implementation_shortfall, scorecard
from tcakit.adapters import tcalogger
from tcakit.report import gate_section, render, scorecard_section, shortfall_section, table

MIN_PARENTS_FOR_IMPACT_FIT = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fee-per-contract-leg", type=float, default=0.65)
    a = ap.parse_args()

    orders, fills = tcalogger.load(a.paths, fee_per_contract_leg=a.fee_per_contract_leg)
    sf = implementation_shortfall(orders, fills)
    units = contract_costs(sf, orders)
    per_order = sf.merge(units, on="order_id").merge(
        orders[["order_id", "leg", "urgency", "half_spread", "latency_ms", "n_children"]], on="order_id")
    tier = fills.groupby("order_id")["tier_hit"].max().rename("tier_hit")
    per_order = per_order.merge(tier, on="order_id", how="left")

    sections = [
        shortfall_section(sf, orders, "quote units × contracts (÷ multiplier 100 → $ per contract)"),
        "## Option cost units per order\n\n" + table(
            per_order, ["order_id", "leg", "tier_hit", "usd_per_contract", "frac_half_spread",
                        "bps_underlying", "latency_ms"], nd=3)
        + "\n`bps_underlying` needs delta and spot per order; this source has neither, so it is `—`.",
        scorecard_section("Scorecard by leg (open = sell combo, close = buy back)",
                          scorecard(per_order, "leg", "frac_half_spread"), "fraction of net half-spread"),
        scorecard_section("Scorecard by escalator tier hit",
                          scorecard(per_order, "tier_hit", "frac_half_spread"), "fraction of net half-spread"),
        scorecard_section("Scorecard by urgency",
                          scorecard(per_order, "urgency", "usd_per_contract"), "$ per contract"),
        gate_section(len(orders), MIN_PARENTS_FOR_IMPACT_FIT, "metaorder impact fit"),
    ]
    prov = {
        "tcakit": __version__,
        "source": ", ".join(f"{Path(p).name} (sha256 {hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]})" for p in a.paths),
        "fees": f"assumed {a.fee_per_contract_leg:.2f} $/contract/leg (not in source)",
        "coverage": f"{len(fills)} child fills → {len(orders)} parents; {int((sf['filled_qty'] > 0).sum())} with fills",
        "period": f"{orders['arrival_ts'].min():%Y-%m-%d} → {orders['arrival_ts'].max():%Y-%m-%d}",
    }
    md = render("Post-trade report — multi-leg option combos", sections, prov)
    if a.out:
        Path(a.out).write_text(md)
        print(f"wrote {a.out}")
    print(md)


if __name__ == "__main__":
    main()
