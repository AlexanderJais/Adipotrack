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

import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import IO

import numpy as np
import pandas as pd

DEG_REQUIRED = ["gene_id", "gene_name", "log2FoldChange", "pvalue", "padj"]
DEG_ANNOTATIONS = [
    "gene_biotype", "gene_description", "tf_family",
    "gene_chr", "gene_start", "gene_end", "gene_strand", "gene_length",
]


def load_deg(source: str | IO[bytes]) -> pd.DataFrame:
    """Load a DESeq2 DEG table (tab-separated, .xls extension is misleading)."""
    df = pd.read_csv(source, sep="\t", low_memory=False, encoding="utf-8-sig")
    missing = [c for c in DEG_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"DEG file missing required columns: {missing}")
    keep = DEG_REQUIRED + [c for c in DEG_ANNOTATIONS if c in df.columns]
    df = df[keep].copy()
    df["log2FoldChange"] = pd.to_numeric(df["log2FoldChange"], errors="coerce")
    df["padj"] = pd.to_numeric(df["padj"], errors="coerce")
    df = df.dropna(subset=["gene_name", "log2FoldChange", "padj"])
    df = df[df["gene_name"].astype(str).str.len() > 0]
    n_before = len(df)
    df = df.sort_values("padj").drop_duplicates("gene_name", keep="first")
    n_collapsed = n_before - len(df)
    if n_collapsed:
        warnings.warn(
            f"load_deg: collapsed {n_collapsed} duplicate gene_name rows "
            "(kept smallest padj per symbol)",
            stacklevel=2,
        )
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

    Returns a per-gene table with both LFCs and padjs side by side, plus any
    annotation columns carried from the genetic-control table.
    """
    g = tp.genetic.rename(columns={
        "log2FoldChange": "lfc_genetic",
        "padj": "padj_genetic",
        "pvalue": "p_genetic",
        "gene_id": "gene_id_g",
    })
    # Strip annotations from the vehicle side so we don't get _x/_y suffix
    # collisions on identical biological metadata.
    v_drop = [c for c in DEG_ANNOTATIONS if c in tp.vehicle.columns]
    v = tp.vehicle.drop(columns=v_drop).rename(columns={
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


_TRAJECTORY_NUMERIC_COLS = [
    "lfc", "lfc_genetic", "lfc_vehicle", "padj_genetic", "padj_vehicle",
]


def trajectories(
    consensus_2h: pd.DataFrame,
    consensus_4h: pd.DataFrame,
) -> pd.DataFrame:
    """Genes that are strict-consensus at BOTH timepoints, with classes."""
    a = consensus_2h[["gene_name", *_TRAJECTORY_NUMERIC_COLS]].rename(
        columns={c: f"{c}_2h" for c in _TRAJECTORY_NUMERIC_COLS}
    )
    b = consensus_4h[["gene_name", *_TRAJECTORY_NUMERIC_COLS]].rename(
        columns={c: f"{c}_4h" for c in _TRAJECTORY_NUMERIC_COLS}
    )
    j = a.merge(b, on="gene_name", how="inner")

    s2 = np.sign(j["lfc_2h"].to_numpy())
    s4 = np.sign(j["lfc_4h"].to_numpy())
    grow = j["lfc_4h"].abs().to_numpy() >= j["lfc_2h"].abs().to_numpy()
    same_up = (s2 > 0) & (s4 > 0)
    same_dn = (s2 < 0) & (s4 < 0)
    j["class"] = np.select(
        [same_up & grow, same_up & ~grow, same_dn & grow, same_dn & ~grow],
        ["stable_up", "damping_up", "stable_down", "damping_down"],
        default="reversed",
    )
    j["delta_lfc"] = j["lfc_4h"] - j["lfc_2h"]

    # Carry biological annotations from the 2 h consensus (identical at 4 h).
    annot_cols = [c for c in DEG_ANNOTATIONS if c in consensus_2h.columns]
    if annot_cols:
        j = j.merge(
            consensus_2h[["gene_name", *annot_cols]],
            on="gene_name", how="left",
        )

    j["class"] = pd.Categorical(j["class"], categories=CLASS_ORDER, ordered=True)
    return j.sort_values(["class", "delta_lfc"]).reset_index(drop=True)


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
