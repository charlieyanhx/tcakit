"""Markdown reports. Every table states its unit, sign convention, and n; numbers that are
not identified by the data are printed as the reason they are missing, not as zeros.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .shortfall import COMPONENTS


def _fmt(v, nd=4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    return str(v)


def table(df: pd.DataFrame, cols: list[str] | None = None, nd: int = 4) -> str:
    cols = cols or list(df.columns)
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = "".join("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |\n" for _, r in df[cols].iterrows())
    return head + body


def shortfall_section(sf: pd.DataFrame, orders: pd.DataFrame, unit_note: str) -> str:
    n = len(sf)
    filled = int((sf["filled_qty"] > 0).sum())
    lines = [f"## Implementation shortfall (Perold) — n = {n} parent orders, {filled} with fills", ""]
    lines.append(f"Cash units: {unit_note}. Sign: `side · (paid − reference)`, positive = cost.")
    same = (orders["decision_ts"] == orders["arrival_ts"]).all()
    if same:
        lines.append("Delay component is **0 by construction**: this source has no decision "
                     "timestamp distinct from arrival.")
    lines.append("")
    agg = sf[COMPONENTS + ["execution", "total"]].agg(["sum", "mean"]).T.reset_index()
    agg.columns = ["component", "sum", "mean_per_order"]
    lines.append(table(agg, nd=4))
    lines.append("")
    lines.append("Per order:")
    lines.append("")
    lines.append(table(sf, ["order_id", "side", "qty", "filled_qty", "avg_px", "arrival_px",
                            "spread", "timing", "opportunity", "fees", "total"], nd=4))
    return "\n".join(lines)


def scorecard_section(title: str, sc: pd.DataFrame, unit: str) -> str:
    lines = [f"## {title}", "", f"Unit: {unit}. CI = 95 % bootstrap on the mean; `—` where n < 2.", ""]
    cols = [c for c in sc.columns if c not in ("note",)]
    lines.append(table(sc, cols, nd=4))
    if "note" in sc.columns and sc["note"].notna().any():
        lines.append("")
        lines.append(f"Note: {sc['note'].dropna().iloc[0]}")
    return "\n".join(lines)


def gate_section(n_orders: int, min_n: int, what: str) -> str:
    status = "MET" if n_orders >= min_n else "NOT MET"
    return (f"## n-gate for {what}: **{status}** ({n_orders} / {min_n} parent orders)\n\n"
            + ("" if status == "MET" else
               f"No {what} number is quoted below this gate; the sections above are descriptive only.\n"))


def render(title: str, sections: list[str], provenance: dict[str, str]) -> str:
    prov = "\n".join(f"- **{k}**: {v}" for k, v in provenance.items())
    return f"# {title}\n\n{prov}\n\n" + "\n\n".join(sections) + "\n"
