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
- Earlier timepoints (e.g. 20 min) can be attached to the trajectory table as
  reference-only columns via ``add_earlier_reference`` — they are shown as a
  leading point/column but never change trajectory membership.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import IO

import numpy as np
import pandas as pd

# Per-sample normalized-count column-name shapes we know how to parse.
# Anchored so we don't accidentally match quantification suffixes like
# "S897_TC2_count" or "Cre_Q927_fpkm".
#   - Cre_Q###/Wt_Q###  — original lab convention (group encoded as prefix).
#   - S###_<code><tp>   — sample id + group letter code + timepoint, where
#                         T=transgene/Cre, W=wildtype, C=CNO, S=saline
#                         (e.g. S897_TC2 = transgene + CNO at 2 h).
SAMPLE_COL_PATTERNS = (
    re.compile(r"^(?:Cre|Wt)_Q\d+$", re.IGNORECASE),
    re.compile(r"^[A-Za-z]+\d+_[A-Za-z]+\d+$"),
)

DEG_REQUIRED = ["gene_id", "gene_name", "log2FoldChange", "pvalue", "padj"]
DEG_ANNOTATIONS = [
    "gene_biotype", "gene_description", "tf_family",
    "gene_chr", "gene_start", "gene_end", "gene_strand", "gene_length",
]


def _peek_magic(source: str | IO[bytes], n: int = 8) -> bytes:
    """Read the first ``n`` bytes from ``source`` without disturbing position."""
    if hasattr(source, "read") and hasattr(source, "seek"):
        head = source.read(n)
        source.seek(0)
        return head
    if isinstance(source, (str, bytes)):
        try:
            with open(source, "rb") as f:
                return f.read(n)
        except OSError:
            return b""
    return b""


def load_deg(source: str | IO[bytes]) -> pd.DataFrame:
    """Load a DESeq2 DEG table (tab-separated, .xls extension is misleading)."""
    head = _peek_magic(source)
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        raise ValueError(
            "DEG file looks like a real binary Excel workbook (.xls), not "
            "the tab-separated text we expect. Re-export from the upstream "
            "pipeline as TSV."
        )
    if head.startswith(b"PK\x03\x04"):
        raise ValueError(
            "DEG file looks like a binary .xlsx workbook, not the tab-"
            "separated text we expect. Re-export as TSV."
        )
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


def add_earlier_reference(
    traj: pd.DataFrame,
    tp: TimepointInputs,
    label: str = "20m",
) -> pd.DataFrame:
    """Attach an earlier timepoint's fold changes to the trajectory table.

    The trajectory set stays exactly as computed from the 2 h ∩ 4 h
    intersection — this only *adds* reference columns so the earlier timepoint
    can be drawn as a leading point in the trajectory line plot, an extra block
    in the heatmap, and extra columns in the exported tables. Membership is
    never gated on the earlier timepoint, so fold changes are pulled from the
    raw DEG tables (every tested gene, not just the earlier consensus) and genes
    absent there simply get NaN.

    Adds ``lfc_genetic_<label>``, ``lfc_vehicle_<label>``, and the canonical
    ``lfc_<label>`` (= vehicle-control LFC, matching the effect size used for
    ``lfc_2h`` / ``lfc_4h``).
    """
    g = tp.genetic[["gene_name", "log2FoldChange"]].rename(
        columns={"log2FoldChange": f"lfc_genetic_{label}"}
    )
    v = tp.vehicle[["gene_name", "log2FoldChange"]].rename(
        columns={"log2FoldChange": f"lfc_vehicle_{label}"}
    )
    out = (
        traj.merge(g, on="gene_name", how="left")
            .merge(v, on="gene_name", how="left")
    )
    out[f"lfc_{label}"] = out[f"lfc_vehicle_{label}"]
    return out


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


# ---- Core circadian clock module -------------------------------------------
#
# A curated panel of core clock genes, grouped by their role in the
# transcription–translation feedback loop (TTFL) and by phase "family":
#   - positive limb (activators of the E-box): Bmal1/Arntl, Npas2, Clock; plus
#     the ROR activators of Bmal1 and the Bmal1-phased D-box repressor Nfil3.
#   - repressive limb / E-box outputs (peak antiphase to Bmal1): Per, Cry,
#     Rev-erb (Nr1d1/2), Dec (Bhlhe40/41), the PAR-bZip factors Dbp/Tef/Hlf,
#     and Chrono (Ciart).
#   - accessory post-translational regulators (not cleanly phased).
# Antiphase readout: positive limb up while the repressive limb / outputs go
# down (or vice versa).

CLOCK_ROLES = [
    ("activator", "Activators (positive limb)"),
    ("per_cry",   "Repressors (Per / Cry)"),
    ("nr",        "Rev-erb / ROR"),
    ("dec",       "Dec repressors"),
    ("output",    "PAR-bZip / D-box outputs"),
    ("accessory", "Accessory / post-translational"),
]

# (role, family, display label, reference peak phase, [symbols to try]).
#
# Family ("positive"/"repressive"/"accessory") drives the antiphase colouring
# and summary; role drives the functional grouping in the bar chart.
#
# ``peak_zt`` is the gene's approximate transcript acrophase in mouse
# peripheral tissue (Zeitgeber time, 0–24 h; ZT0 = lights on), taken from
# published circadian atlases (e.g. Zhang et al. 2014 PNAS). It is a REFERENCE
# phase for positioning genes on the clock-face / phase plots — NOT a phase
# measured in this experiment. Genes with no robust/low-amplitude rhythm
# (Clock, the post-translational accessory factors) carry ``None`` and are
# omitted from the phase-based figures. Edit these values to match your own
# tissue/reference; every phase figure reads straight from this column.
CLOCK_GENES = [
    ("activator", "positive",   "Bmal1 (Arntl)",   23.0, ["Arntl", "Bmal1"]),
    ("activator", "positive",   "Bmal2 (Arntl2)",  23.0, ["Arntl2", "Bmal2"]),
    ("activator", "positive",   "Clock",           None, ["Clock"]),
    ("activator", "positive",   "Npas2",           22.0, ["Npas2"]),
    ("per_cry",   "repressive", "Per1",            11.0, ["Per1"]),
    ("per_cry",   "repressive", "Per2",            13.0, ["Per2"]),
    ("per_cry",   "repressive", "Per3",            12.0, ["Per3"]),
    ("per_cry",   "repressive", "Cry1",            18.0, ["Cry1"]),
    ("per_cry",   "repressive", "Cry2",            10.0, ["Cry2"]),
    ("nr",        "repressive", "Rev-erbα (Nr1d1)", 7.0, ["Nr1d1"]),
    ("nr",        "repressive", "Rev-erbβ (Nr1d2)", 8.0, ["Nr1d2"]),
    ("nr",        "positive",   "Rorα (Rora)",      0.0, ["Rora"]),
    ("nr",        "positive",   "Rorβ (Rorb)",      2.0, ["Rorb"]),
    ("nr",        "positive",   "Rorγ (Rorc)",     20.0, ["Rorc"]),
    ("dec",       "repressive", "Dec1 (Bhlhe40)",   6.0, ["Bhlhe40", "Dec1", "Stra13"]),
    ("dec",       "repressive", "Dec2 (Bhlhe41)",   5.0, ["Bhlhe41", "Dec2", "Sharp1"]),
    ("output",    "repressive", "Dbp",              9.0, ["Dbp"]),
    ("output",    "repressive", "Tef",             10.0, ["Tef"]),
    ("output",    "repressive", "Hlf",             11.0, ["Hlf"]),
    ("output",    "positive",   "Nfil3 (E4bp4)",    1.0, ["Nfil3", "E4bp4"]),
    ("accessory", "repressive", "Chrono (Ciart)",  11.0, ["Ciart", "Gm129"]),
    ("accessory", "accessory",  "Timeless",        None, ["Timeless", "Tim"]),
    ("accessory", "accessory",  "Csnk1d",          None, ["Csnk1d"]),
    ("accessory", "accessory",  "Csnk1e",          None, ["Csnk1e"]),
    ("accessory", "accessory",  "Fbxl3",           None, ["Fbxl3"]),
]

CLOCK_FAMILY_COLORS = {
    "positive":   "#D55E00",  # vermillion — positive limb
    "repressive": "#0072B2",  # blue — repressive limb / outputs
    "accessory":  "#999999",  # grey
}

_CLOCK_ROLE_ORDER = [r for r, _ in CLOCK_ROLES]


def clock_gene_table(deg: pd.DataFrame) -> pd.DataFrame:
    """Look up the curated clock panel in one DEG table (one timepoint).

    Matching is case-insensitive and tries each symbol/alias in order. Returns
    one row per curated gene (in ``CLOCK_GENES`` order) with columns: role,
    role_label, family, label, gene (matched symbol), lfc, padj, found.
    Missing genes get NaN LFC/padj and found=False.
    """
    lut: dict[str, tuple[str, float, float]] = {}
    for name, lfc, padj in zip(deg["gene_name"].astype(str),
                               deg["log2FoldChange"], deg["padj"]):
        lut.setdefault(name.lower(), (name, float(lfc), float(padj)))

    role_labels = dict(CLOCK_ROLES)
    rows = []
    for role, family, label, peak_zt, symbols in CLOCK_GENES:
        hit = next((lut[s.lower()] for s in symbols if s.lower() in lut), None)
        rows.append({
            "role": role,
            "role_label": role_labels[role],
            "family": family,
            "label": label,
            "peak_zt": peak_zt if peak_zt is not None else np.nan,
            "gene": hit[0] if hit else symbols[0],
            "lfc": hit[1] if hit else np.nan,
            "padj": hit[2] if hit else np.nan,
            "found": hit is not None,
        })
    out = pd.DataFrame(rows)
    out["role"] = pd.Categorical(out["role"], _CLOCK_ROLE_ORDER, ordered=True)
    return out


def clock_gene_panel(deg_by_tp: "dict[str, pd.DataFrame]") -> pd.DataFrame:
    """Assemble a multi-timepoint clock table.

    ``deg_by_tp`` maps a timepoint label to its DEG table (all the same
    contrast — e.g. all vehicle-control). Because ``clock_gene_table`` returns
    genes in a fixed order, the per-timepoint columns are assembled
    positionally. Returns role, role_label, family, label plus ``lfc_<tp>`` and
    ``padj_<tp>`` for each timepoint key (in the order given).
    """
    tables = {tp: clock_gene_table(deg) for tp, deg in deg_by_tp.items()}
    first = next(iter(tables.values()))
    out = first[["role", "role_label", "family", "label", "peak_zt"]].copy()
    for tp, t in tables.items():
        out[f"lfc_{tp}"] = t["lfc"].to_numpy()
        out[f"padj_{tp}"] = t["padj"].to_numpy()
    return out


def fit_phase_cosine(
    zt: np.ndarray, y: np.ndarray, period: float = 24.0
) -> dict:
    """Least-squares cosine fit of ``y`` against circular phase ``zt``.

    Models ``y ≈ mesor + amplitude·cos(2π(zt − peak_zt)/period)`` by solving
    the linear system in [1, cos(ωzt), sin(ωzt)]. Used by the phase-vs-log2FC
    figure to summarise whether regulation is a coherent function of a gene's
    normal peak phase (this is a fit of the *perturbation effect across genes
    by their reference phase*, NOT a rhythmicity/time-series fit).

    Returns a dict with mesor, amplitude, peak_zt (phase of maximum, in the
    same units as ``zt``), r2, and n. Returns NaNs if fewer than 4 finite
    points are supplied.
    """
    zt = np.asarray(zt, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(zt) & np.isfinite(y)
    zt, y = zt[ok], y[ok]
    nan = {"mesor": np.nan, "amplitude": np.nan, "peak_zt": np.nan,
           "r2": np.nan, "n": int(zt.size)}
    if zt.size < 4:
        return nan
    w = 2.0 * np.pi / period
    X = np.column_stack([np.ones_like(zt), np.cos(w * zt), np.sin(w * zt)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    c0, a, b = (float(coef[0]), float(coef[1]), float(coef[2]))
    amplitude = float(np.hypot(a, b))
    peak = float((np.arctan2(b, a) / w) % period)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"mesor": c0, "amplitude": amplitude, "peak_zt": peak,
            "r2": r2, "n": int(zt.size)}


# ---- Clock-arrest evidence -------------------------------------------------
#
# These are in-silico analyses that BEAR ON the "clock arrest" hypothesis (the
# oscillator stalled at a Bmal1-high state) but cannot prove it: a 20 min /
# 2 h / 4 h perturbation series is far shorter than the ~24 h period, so a
# sustained displacement is necessary but not sufficient for arrest (it is
# equally consistent with a phase shift or a damped oscillation). Deciding
# among those needs a circadian time course / real-time reporter.

_SETTLE_CATEGORIES = ["amplifying", "held", "reverting", "reversed", "unchanged"]


def clock_settling_table(
    panel: pd.DataFrame,
    early_tp: str,
    late_tp: str,
    min_disp: float = 0.25,
    tol: float = 0.15,
) -> pd.DataFrame:
    """Does each clock gene's displacement HOLD across the last interval?

    Compares the log2FC at ``early_tp`` vs ``late_tp`` (columns
    ``lfc_<tp>`` in a ``clock_gene_panel``). For genes displaced by at least
    ``min_disp`` at either timepoint, classifies the late-interval behaviour:

      - ``amplifying``  same sign, |late| ≥ (1+tol)·|early|  (still moving out)
      - ``held``        same sign, within ±tol of |early|     (plateau)
      - ``reverting``   same sign, |late| ≤ (1−tol)·|early|   (returning to 0)
      - ``reversed``    sign flipped between the two timepoints
      - ``unchanged``   |log2FC| < min_disp at both timepoints

    Arrest predicts predominantly ``held``/``amplifying`` (the state persists),
    especially that repressive-limb targets stay suppressed rather than
    ``reverting`` upward. Returns per-gene rows with lfc_early, lfc_late, delta,
    retention (|late|/|early|), and category.
    """
    ec, lc = f"lfc_{early_tp}", f"lfc_{late_tp}"
    d = panel[["label", "family", "role_label", ec, lc]].copy()
    d = d.rename(columns={ec: "lfc_early", lc: "lfc_late"})
    e = d["lfc_early"].to_numpy(dtype=float)
    l = d["lfc_late"].to_numpy(dtype=float)
    ok = np.isfinite(e) & np.isfinite(l)
    d = d[ok].reset_index(drop=True)
    e, l = e[ok], l[ok]

    ae, al = np.abs(e), np.abs(l)
    disp = np.maximum(ae, al) >= min_disp
    same_sign = np.sign(e) == np.sign(l)
    with np.errstate(divide="ignore", invalid="ignore"):
        retention = np.where(ae > 1e-9, al / ae, np.inf)

    cat = np.full(len(d), "unchanged", dtype=object)
    displaced = disp
    rev = displaced & ~same_sign & (ae >= min_disp) & (al >= min_disp)
    amp = displaced & same_sign & (retention >= 1 + tol)
    revert = displaced & same_sign & (retention <= 1 - tol)
    held = displaced & same_sign & ~amp & ~revert
    cat[rev] = "reversed"
    cat[amp] = "amplifying"
    cat[revert] = "reverting"
    cat[held] = "held"

    d["delta"] = l - e
    d["retention"] = np.where(np.isfinite(retention), retention, np.nan)
    d["category"] = pd.Categorical(cat, categories=_SETTLE_CATEGORIES,
                                   ordered=True)
    return d


def clock_settling_summary(settling: pd.DataFrame) -> dict:
    """Summarise a ``clock_settling_table``: category counts, the persistence
    fraction (held+amplifying among displaced genes), and — as the most
    arrest-relevant readout — the fraction of repressive-limb targets that
    were down early and STAY suppressed (held/amplifying) rather than
    reverting.
    """
    counts = {c: int((settling["category"] == c).sum())
              for c in _SETTLE_CATEGORIES}
    displaced = sum(counts[c] for c in
                    ("amplifying", "held", "reverting", "reversed"))
    persist = counts["amplifying"] + counts["held"]
    persistence = persist / displaced if displaced else float("nan")

    rep = settling[(settling["family"] == "repressive")
                   & (settling["lfc_early"] < 0)]
    rep_disp = rep[rep["category"] != "unchanged"]
    held_down = int(rep_disp["category"].isin(["held", "amplifying"]).sum())
    target_hold = (held_down / len(rep_disp)) if len(rep_disp) else float("nan")
    return {
        "counts": counts,
        "n_displaced": displaced,
        "persistence": persistence,
        "target_hold": target_hold,
        "n_targets_down": int(len(rep_disp)),
    }


# Curated CONSENSUS direction of core-clock transcripts in torpor / hibernation
# in peripheral tissue (esp. liver), relative to the euthermic / interbout
# state. +1 = higher in torpor, −1 = lower / suppressed. Keyed by the display
# labels in CLOCK_GENES.
#
# This is a SIMPLIFIED consensus drawn from a heterogeneous literature —
# directions vary with species, tissue and torpor depth — and is provided as an
# EDITABLE reference, not settled ground truth. Basis: Revel et al. 2007 PNAS
# (Djungarian hamster daily torpor, central + peripheral clock genes); Williams
# et al. 2012 BMC Genomics (13-lined ground squirrel liver); and reviews of
# peripheral-clock damping in hibernation. Genes with inconsistent directional
# reports are deliberately omitted. Edit freely to match your own reference.
TORPOR_CLOCK_SIGNATURE = {
    "Bmal1 (Arntl)":    +1,
    "Npas2":            +1,
    "Nfil3 (E4bp4)":    +1,
    "Per1":             -1,
    "Per2":             -1,
    "Per3":             -1,
    "Cry1":             -1,
    "Rev-erbα (Nr1d1)": -1,
    "Rev-erbβ (Nr1d2)": -1,
    "Dec1 (Bhlhe40)":   -1,
    "Dec2 (Bhlhe41)":   -1,
    "Dbp":              -1,
    "Tef":              -1,
}

TORPOR_SIGNATURE_SOURCE = (
    "Curated consensus (Revel 2007 PNAS; Williams 2012 BMC Genomics; "
    "hibernation peripheral-clock reviews) — simplified and editable, not "
    "settled ground truth."
)


def torpor_concordance(
    clock_table: pd.DataFrame,
    signature: "dict[str, int] | None" = None,
) -> "tuple[pd.DataFrame, dict]":
    """Test whether our clock-gene log2FC signs match a torpor reference.

    ``clock_table`` is a single-timepoint ``clock_gene_table``. For each gene
    present in both our data and the reference signature, compares the sign of
    our log2FC to the reference direction. Returns (per-gene table, summary),
    where summary has k (concordant), n (compared), frac, and a one-sided
    binomial p-value against chance (0.5) — NaN if scipy is unavailable.
    """
    sig = signature if signature is not None else TORPOR_CLOCK_SIGNATURE
    rows = []
    for _, r in clock_table.iterrows():
        exp = sig.get(r["label"])
        if exp is None or not r["found"] or not np.isfinite(r["lfc"]):
            continue
        obs = int(np.sign(r["lfc"])) if r["lfc"] != 0 else 0
        rows.append({
            "label": r["label"],
            "lfc": float(r["lfc"]),
            "padj": float(r["padj"]) if np.isfinite(r["padj"]) else np.nan,
            "expected": int(exp),
            "observed_sign": obs,
            "concordant": bool(obs == np.sign(exp)),
        })
    per = pd.DataFrame(
        rows,
        columns=["label", "lfc", "padj", "expected", "observed_sign",
                 "concordant"],
    )
    n = len(per)
    k = int(per["concordant"].sum()) if n else 0
    p = np.nan
    if n:
        try:
            from scipy.stats import binomtest
            p = float(binomtest(k, n, 0.5, alternative="greater").pvalue)
        except ImportError:
            pass
    summary = {"k": k, "n": n,
               "frac": (k / n) if n else float("nan"), "p": p}
    return per, summary


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


_TF_ENRICHMENT_COLS = [
    "tf_family", "n_fg", "n_fg_total", "n_bg", "n_bg_total",
    "odds_ratio", "p_value", "q_value",
]


def tf_enrichment(
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    min_family_size: int = 3,
) -> pd.DataFrame:
    """Fisher's exact test: is each tf_family over-represented in `foreground`
    compared to `background`?

    Both inputs need a `tf_family` column. Returns a DataFrame sorted by
    BH-adjusted q-value, with columns: tf_family, n_fg, n_fg_total,
    n_bg, n_bg_total, odds_ratio, p_value, q_value. Always carries the same
    column schema, even when no families pass the size filter.
    """
    if "tf_family" not in foreground.columns or "tf_family" not in background.columns:
        return pd.DataFrame(columns=_TF_ENRICHMENT_COLS)

    try:
        from scipy.stats import fisher_exact
    except ImportError:
        warnings.warn("scipy not installed; tf_enrichment returning empty result")
        return pd.DataFrame(columns=_TF_ENRICHMENT_COLS)

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
    if not rows:
        return pd.DataFrame(columns=_TF_ENRICHMENT_COLS)
    out = pd.DataFrame(rows)
    out["q_value"] = _bh_adjust(out["p_value"].to_numpy())
    return out.sort_values("q_value").reset_index(drop=True)


# ---- Sample-level (replicate QC) -------------------------------------------

@dataclass
class SampleData:
    """Per-sample normalized counts plus group metadata for one DEG file."""
    expr: pd.DataFrame      # gene_name (index) × sample (columns)
    metadata: pd.DataFrame  # columns: sample, group


def _column_token(col: str) -> str | None:
    """Extract the group-encoding token from a sample column name.

    Returns the uppercase letter code that identifies the sample group, or
    ``None`` if the column does not match a known per-sample shape.
    """
    m = re.match(r"^[A-Za-z]+\d+_([A-Za-z]+)\d+$", col)
    if m:
        return m.group(1).upper()
    m = re.match(r"^(Cre|Wt)_Q\d+$", col, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def _label_tokens(label: str) -> set[str]:
    """All column tokens that could correspond to ``label``.

    Generates both the legacy single-keyword form (``CRE``/``WT``) and the
    two-letter S-style form (``T``/``W`` for treatment, ``C``/``S`` for
    stimulus). Order-tolerant for the two-letter case (``TC`` and ``CT``).
    """
    L = label.upper()
    out: set[str] = set()
    if "CRE" in L: out.add("CRE")
    if "WT" in L:  out.add("WT")
    treat = "T" if "CRE" in L else ("W" if "WT" in L else "")
    stim = "C" if "CNO" in L else ("S" if "SAL" in L else "")
    if treat and stim:
        out.add(treat + stim)
        out.add(stim + treat)
    return out


def load_samples(
    source: str | IO[bytes],
    group_a_label: str,
    group_b_label: str,
) -> SampleData:
    """Load per-sample normalized-count columns from a DEG file.

    Sample columns are detected by name shape (see ``SAMPLE_COL_PATTERNS``).
    Group assignment matches each column's encoded token against tokens
    derived from the supplied labels; if that produces an unambiguous split
    it is used, otherwise the loader falls back to splitting the columns in
    half by position (the shape used by single-prefix files).
    """
    raw = pd.read_csv(source, sep="\t", low_memory=False, encoding="utf-8-sig")
    if "gene_name" not in raw.columns:
        raise ValueError("Sample loader: missing gene_name column")

    sample_cols: list[str] = []
    for pat in SAMPLE_COL_PATTERNS:
        sample_cols = [c for c in raw.columns if pat.fullmatch(c)]
        if sample_cols:
            break
    if not sample_cols:
        raise ValueError(
            "Sample loader: no per-sample count columns found. "
            f"Headers seen: {list(raw.columns)}"
        )

    a_tokens = _label_tokens(group_a_label)
    b_tokens = _label_tokens(group_b_label)
    groups: list[str] | None = []
    for c in sample_cols:
        tok = _column_token(c)
        in_a = tok in a_tokens
        in_b = tok in b_tokens
        if in_a and not in_b:
            groups.append(group_a_label)
        elif in_b and not in_a:
            groups.append(group_b_label)
        else:
            groups = None
            break

    if groups is None:
        # Token matching was ambiguous (e.g. legacy single-prefix file where
        # every column says "Cre"). Split by column position.
        if len(sample_cols) % 2 != 0:
            raise ValueError(
                f"Sample loader: cannot resolve groups for {sample_cols} "
                f"into {group_a_label!r} / {group_b_label!r}; "
                f"odd sample count prevents a positional split."
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
    (n_samples, n_components) and var_explained sums to ≤ 1. Note that
    `var_explained` is the fraction *of variance among the genes used*
    (post top-variance filtering), not of the full transcriptome — this is
    the standard QC reading but worth keeping in mind when comparing across
    runs with different gene-count thresholds.
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
    # Excel caps sheet names at 31 chars. Detect collisions before writing
    # so a longer-named sheet can't silently overwrite an earlier one.
    seen: dict[str, str] = {}
    for name in sheets:
        truncated = name[:31]
        if truncated in seen:
            raise ValueError(
                f"Sheet name collision after 31-char truncation: "
                f"{seen[truncated]!r} and {name!r} both become {truncated!r}"
            )
        seen[truncated] = name
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()
