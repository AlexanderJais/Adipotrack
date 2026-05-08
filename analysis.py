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

import re
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import IO

import numpy as np
import pandas as pd

# Per-sample normalized-count column names look like "Cre_Q927" / "Wt_Q919".
# Anchored on both ends so we don't match "Cre_Q927_count" or "Cre_Q927_fpkm".
SAMPLE_COL_RE = re.compile(r"^(?:Cre|Wt)_Q\d+$", re.IGNORECASE)

# Columns to *exclude* when auto-detecting per-sample count columns. Covers
# the standard DEG/DESeq2 outputs and the annotation block carried by load_deg.
NON_SAMPLE_COLS = {
    "gene_id", "gene_name", "baseMean", "log2FoldChange", "lfcSE",
    "stat", "pvalue", "padj", "pvalue_adjusted",
    "gene_biotype", "gene_description", "tf_family",
    "gene_chr", "gene_start", "gene_end", "gene_strand", "gene_length",
}

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
        ["sustained_up", "transient_up", "sustained_down", "transient_down"],
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
    "sustained_up", "transient_up",
    "sustained_down", "transient_down",
    "reversed",
]

CLASS_COLORS = {
    "sustained_up":   "#D55E00",  # vermillion
    "transient_up":   "#E69F00",  # orange
    "sustained_down": "#0072B2",  # blue
    "transient_down": "#56B4E9",  # sky blue
    "reversed":       "#999999",  # grey
}


def is_tf(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: gene has an annotated TF family (tf_family != '-')."""
    if "tf_family" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["tf_family"].astype(str).str.strip().ne("-") & df["tf_family"].notna()


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR adjustment."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    bh = ranked * n / np.arange(1, n + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(bh, 1.0)
    return out


def tf_enrichment(
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    min_family_size: int = 3,
) -> pd.DataFrame:
    """Fisher's exact test: is each tf_family over-represented in `foreground`
    compared to `background`?

    Both inputs need a `tf_family` column. Returns a DataFrame sorted by
    BH-adjusted q-value, with columns: tf_family, n_fg, n_fg_total,
    n_bg, n_bg_total, odds_ratio, p_value, q_value.
    """
    if "tf_family" not in foreground.columns or "tf_family" not in background.columns:
        return pd.DataFrame(columns=[
            "tf_family", "n_fg", "n_fg_total", "n_bg", "n_bg_total",
            "odds_ratio", "p_value", "q_value",
        ])

    try:
        from scipy.stats import fisher_exact
    except ImportError:
        warnings.warn("scipy not installed; tf_enrichment returning empty result")
        return pd.DataFrame()

    # Background of TFs that are NOT in the foreground. Foreground is a
    # subset of the union of tested genes, so leaving it in the background
    # would double-count the in-set TFs and bias the test toward the null.
    fg = foreground[is_tf(foreground)].copy()
    fg_names = set(fg["gene_name"].astype(str))
    bg_excl = background[
        is_tf(background) & ~background["gene_name"].astype(str).isin(fg_names)
    ].copy()
    fg_total = len(fg)
    bg_total = len(bg_excl)
    families = (
        fg["tf_family"].value_counts()
        .loc[lambda s: s >= min_family_size]
        .index.tolist()
    )

    rows = []
    for fam in families:
        a = int((fg["tf_family"] == fam).sum())
        b = fg_total - a
        c = int((bg_excl["tf_family"] == fam).sum())
        d = bg_total - c
        # One-sided "greater": we only care about over-representation.
        res = fisher_exact([[a, b], [c, d]], alternative="greater")
        odds, p = float(res.statistic), float(res.pvalue)
        rows.append({
            "tf_family": fam,
            "n_fg": a, "n_fg_total": fg_total,
            "n_bg": c, "n_bg_total": bg_total,
            "odds_ratio": odds,
            "p_value": p,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = _bh_adjust(out["p_value"].to_numpy())
        out = out.sort_values("q_value").reset_index(drop=True)
    return out


# ---- Sample-level (replicate QC) -------------------------------------------

@dataclass
class SampleData:
    """Per-sample normalized counts plus group metadata for one DEG file."""
    expr: pd.DataFrame      # gene_name (index) × sample (columns)
    metadata: pd.DataFrame  # columns: sample, group


def load_samples(
    source: str | IO[bytes],
    group_a_label: str,
    group_b_label: str,
) -> SampleData:
    """Load per-sample normalized-count columns from a DEG file.

    Group assignment uses the column prefix when both ``Cre_*`` and ``Wt_*``
    samples are present (genetic-control files): ``Cre_*`` → group_a,
    ``Wt_*`` → group_b. When all detected samples share the same prefix
    (vehicle-control files where every sample is ``Cre_*``) the loader
    falls back to the original [first-half, second-half] convention. The
    function raises with a clear message if neither rule produces a valid
    even split.
    """
    raw = pd.read_csv(source, sep="\t", low_memory=False, encoding="utf-8-sig")
    if "gene_name" not in raw.columns:
        raise ValueError("Sample loader: missing gene_name column")

    # Two-pass detection. First, the strict legacy regex (Cre_Q###/Wt_Q###).
    # If nothing matches, fall back to "any numeric column that isn't a known
    # DEG/DESeq2/annotation field" — this picks up files whose per-sample
    # columns use a different naming convention.
    sample_cols = [c for c in raw.columns if SAMPLE_COL_RE.fullmatch(c)]
    if not sample_cols:
        candidates = [c for c in raw.columns if c not in NON_SAMPLE_COLS]
        sample_cols = [
            c for c in candidates
            if pd.to_numeric(raw[c], errors="coerce").notna().any()
        ]
    if not sample_cols:
        raise ValueError(
            "Sample loader: no per-sample count columns found. "
            f"Headers seen: {list(raw.columns)}"
        )

    # Group assignment: prefer case-insensitive 'cre'/'wt' substring matching
    # anywhere in the column name (handles Cre_Q927, Pnoc_Cre_CNO_R1, etc.).
    # If only one of the two markers is present, fall back to splitting by
    # column position (single-prefix files).
    def _has(col: str, needle: str) -> bool:
        return needle in col.lower()

    cre_cols = [c for c in sample_cols if _has(c, "cre")]
    wt_cols = [c for c in sample_cols if _has(c, "wt") and not _has(c, "cre")]

    if cre_cols and wt_cols:
        groups = [
            group_a_label if _has(c, "cre") else group_b_label
            for c in sample_cols
        ]
    else:
        if len(sample_cols) % 2 != 0:
            raise ValueError(
                f"Sample loader: single-prefix file with odd sample count "
                f"({len(sample_cols)}); cannot infer groups. "
                f"Detected columns: {sample_cols}"
            )
        half = len(sample_cols) // 2
        groups = [group_a_label] * half + [group_b_label] * half

    expr = (
        raw[["gene_name", *sample_cols]]
        .dropna(subset=["gene_name"])
        .drop_duplicates("gene_name", keep="first")
        .set_index("gene_name")
    )
    expr = expr.apply(pd.to_numeric, errors="coerce")
    metadata = pd.DataFrame({"sample": sample_cols, "group": groups})
    return SampleData(expr=expr, metadata=metadata)


def compute_pca(
    expr: pd.DataFrame,
    n_top_var: int = 2000,
    n_components: int = 2,
    log_transform: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA of samples on (optionally log-transformed) top-variance genes.

    Returns (scores, var_explained) where scores has shape
    (n_samples, n_components) and var_explained sums to ≤ 1.
    """
    X = expr.to_numpy(dtype=float)
    # Drop genes with any non-finite value rather than imputing zero — a
    # NaN replaced by 0 then log-transformed looks like a real zero-count
    # gene, which silently inflates apparent low-variance rows.
    finite_rows = np.isfinite(X).all(axis=1)
    X = X[finite_rows]
    if log_transform:
        X = np.log2(np.clip(X, a_min=0.0, a_max=None) + 1.0)
    # Drop zero-variance genes — they carry no information for PCA and
    # would otherwise survive the top-variance filter when n_genes < n_top_var.
    var = X.var(axis=1)
    nonzero_var = var > 0
    X = X[nonzero_var]
    var = var[nonzero_var]
    if X.shape[0] > n_top_var:
        top = np.argsort(var)[-n_top_var:]
        X = X[top]
    if X.shape[0] == 0:
        raise ValueError(
            "compute_pca: no finite, non-constant genes left after filtering"
        )
    # samples × genes for PCA
    Xs = X.T
    Xc = Xs - Xs.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt[:n_components].T
    total = float((S ** 2).sum())
    var_explained = (S[:n_components] ** 2) / total if total > 0 else np.zeros(n_components)
    return scores, var_explained


def sample_correlation(expr: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Pairwise sample correlation on log2(x+1) counts."""
    X = np.log2(expr.to_numpy(dtype=float) + 1.0)
    df = pd.DataFrame(X, columns=expr.columns)
    return df.corr(method=method)


# ----------------------------------------------------------------------------


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()
