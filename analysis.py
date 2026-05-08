"""DEG loading, consensus calling, trajectory classification.

Conventions
-----------
- Significance: padj < 0.05 (configurable).
- Strict consensus at a timepoint: a gene must be significant in BOTH the
  genetic-control (CRE+CNO vs WT+CNO) and vehicle-control (CRE+CNO vs CRE+SAL)
  comparisons AND have the same sign of log2FoldChange in both.
- Trajectory set = intersection: genes that are strict-consensus at BOTH 2h and 4h.
- Canonical effect size for trajectories: LFC from the vehicle-control comparison
  (CRE+CNO vs CRE+SAL), since it directly captures the chemogenetic activation.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import IO

import numpy as np
import pandas as pd

DEG_COLS = ["gene_id", "gene_name", "log2FoldChange", "pvalue", "padj"]


def load_deg(source: str | IO[bytes]) -> pd.DataFrame:
    """Load a DESeq2 DEG table (tab-separated, .xls extension is misleading)."""
    df = pd.read_csv(source, sep="\t", low_memory=False)
    missing = [c for c in DEG_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"DEG file missing required columns: {missing}")
    df = df[DEG_COLS].copy()
    df["log2FoldChange"] = pd.to_numeric(df["log2FoldChange"], errors="coerce")
    df["padj"] = pd.to_numeric(df["padj"], errors="coerce")
    df = df.dropna(subset=["gene_name", "log2FoldChange", "padj"])
    df = df[df["gene_name"].astype(str).str.len() > 0]
    # Collapse duplicate symbols by keeping the row with smallest padj
    df = df.sort_values("padj").drop_duplicates("gene_name", keep="first")
    return df.reset_index(drop=True)


@dataclass
class TimepointInputs:
    """Two DEG tables for one timepoint."""
    genetic: pd.DataFrame   # CRE+CNO vs WT+CNO
    vehicle: pd.DataFrame   # CRE+CNO vs CRE+SAL


def consensus_at_timepoint(
    tp: TimepointInputs,
    padj_thresh: float = 0.05,
    lfc_thresh: float = 0.0,
) -> pd.DataFrame:
    """Strict consensus: significant in both contrasts, concordant LFC sign.

    Returns a per-gene table with both LFCs and padjs side by side.
    """
    g = tp.genetic.rename(columns={
        "log2FoldChange": "lfc_genetic",
        "padj": "padj_genetic",
        "pvalue": "p_genetic",
        "gene_id": "gene_id_g",
    })
    v = tp.vehicle.rename(columns={
        "log2FoldChange": "lfc_vehicle",
        "padj": "padj_vehicle",
        "pvalue": "p_vehicle",
        "gene_id": "gene_id_v",
    })
    merged = g.merge(v, on="gene_name", how="inner")

    sig = (
        (merged["padj_genetic"] < padj_thresh)
        & (merged["padj_vehicle"] < padj_thresh)
        & (merged["lfc_genetic"].abs() >= lfc_thresh)
        & (merged["lfc_vehicle"].abs() >= lfc_thresh)
    )
    concordant = np.sign(merged["lfc_genetic"]) == np.sign(merged["lfc_vehicle"])
    out = merged[sig & concordant].copy()
    # Canonical effect size = vehicle-control LFC (direct chemogenetic effect)
    out["lfc"] = out["lfc_vehicle"]
    out["direction"] = np.where(out["lfc"] > 0, "up", "down")
    return out.reset_index(drop=True)


def sig_set(df: pd.DataFrame, padj_thresh: float = 0.05) -> set[str]:
    return set(df.loc[df["padj"] < padj_thresh, "gene_name"].astype(str))


def trajectories(
    consensus_2h: pd.DataFrame,
    consensus_4h: pd.DataFrame,
) -> pd.DataFrame:
    """Genes that are strict-consensus at BOTH timepoints, with classes."""
    a = consensus_2h[["gene_name", "lfc", "lfc_genetic", "lfc_vehicle",
                      "padj_genetic", "padj_vehicle"]].rename(columns=lambda c:
        f"{c}_2h" if c != "gene_name" else c)
    b = consensus_4h[["gene_name", "lfc", "lfc_genetic", "lfc_vehicle",
                      "padj_genetic", "padj_vehicle"]].rename(columns=lambda c:
        f"{c}_4h" if c != "gene_name" else c)
    j = a.merge(b, on="gene_name", how="inner")

    def classify(r: pd.Series) -> str:
        s2, s4 = np.sign(r["lfc_2h"]), np.sign(r["lfc_4h"])
        if s2 > 0 and s4 > 0:
            return "stable_up" if abs(r["lfc_4h"]) >= abs(r["lfc_2h"]) else "damping_up"
        if s2 < 0 and s4 < 0:
            return "stable_down" if abs(r["lfc_4h"]) >= abs(r["lfc_2h"]) else "damping_down"
        return "reversed"

    j["class"] = j.apply(classify, axis=1)
    j["delta_lfc"] = j["lfc_4h"] - j["lfc_2h"]
    return j.sort_values("class").reset_index(drop=True)


CLASS_ORDER = [
    "stable_up", "damping_up",
    "stable_down", "damping_down",
    "reversed",
]

CLASS_COLORS = {
    "stable_up":    "#D55E00",  # vermillion
    "damping_up":   "#E69F00",  # orange
    "stable_down":  "#0072B2",  # blue
    "damping_down": "#56B4E9",  # sky blue
    "reversed":     "#999999",  # grey
}


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()
