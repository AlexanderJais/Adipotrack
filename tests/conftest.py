"""Shared fixtures: synthetic DEG and sample tables in TSV byte form.

The real DEG files are 14 MB; the fixtures here are deliberately tiny but
preserve the column conventions (.xls extension is misleading, the body is
tab-separated UTF-8) so loaders are exercised on realistic shapes.
"""

from __future__ import annotations

import io
from typing import Iterable

import pandas as pd
import pytest


def _tsv(df: pd.DataFrame) -> bytes:
    return df.to_csv(sep="\t", index=False).encode("utf-8")


def _deg_frame(rows: Iterable[dict]) -> pd.DataFrame:
    """Build a DataFrame with the columns load_deg expects."""
    base_cols = ["gene_id", "gene_name", "log2FoldChange", "pvalue", "padj"]
    df = pd.DataFrame(list(rows))
    for col in base_cols:
        if col not in df.columns:
            raise AssertionError(f"fixture missing required column {col!r}")
    return df


@pytest.fixture
def deg_genetic_2h_bytes() -> bytes:
    """Genetic-control DEG (CRE+CNO vs WT+CNO) at 2 h.

    Includes a tf_family column so TF enrichment paths can be exercised, and
    one gene_name duplicate so the dedup-warning path is hit by load_deg.
    """
    rows = [
        # Sustained-up consensus genes (also significant in vehicle file below)
        {"gene_id": "ENSMUSG01", "gene_name": "Fos",   "log2FoldChange": 2.5, "pvalue": 1e-10, "padj": 1e-8,  "tf_family": "bZIP"},
        {"gene_id": "ENSMUSG02", "gene_name": "Jun",   "log2FoldChange": 1.8, "pvalue": 1e-9,  "padj": 1e-7,  "tf_family": "bZIP"},
        {"gene_id": "ENSMUSG03", "gene_name": "Atf3",  "log2FoldChange": 1.2, "pvalue": 1e-6,  "padj": 1e-4,  "tf_family": "bZIP"},
        # Sustained-down consensus
        {"gene_id": "ENSMUSG04", "gene_name": "Dnmt3a","log2FoldChange": -1.5,"pvalue": 1e-7,  "padj": 1e-5,  "tf_family": "-"},
        {"gene_id": "ENSMUSG05", "gene_name": "Foxp1", "log2FoldChange": -1.0,"pvalue": 1e-5,  "padj": 1e-3,  "tf_family": "Forkhead"},
        # Discordant: sig in both contrasts but opposite signs (consensus drops)
        {"gene_id": "ENSMUSG06", "gene_name": "Mxd1",  "log2FoldChange": 1.4, "pvalue": 1e-6,  "padj": 1e-4,  "tf_family": "bHLH"},
        # Below padj threshold
        {"gene_id": "ENSMUSG07", "gene_name": "Sox2",  "log2FoldChange": 1.1, "pvalue": 0.5,   "padj": 0.7,   "tf_family": "HMG"},
        # Duplicate gene_name to trigger dedup warning in load_deg
        {"gene_id": "ENSMUSG08", "gene_name": "Fos",   "log2FoldChange": 2.4, "pvalue": 5e-10, "padj": 4e-8,  "tf_family": "bZIP"},
    ]
    return _tsv(_deg_frame(rows))


@pytest.fixture
def deg_vehicle_2h_bytes() -> bytes:
    """Vehicle-control DEG (CRE+CNO vs CRE+SAL) at 2 h.

    Mirrors the genetic file's significant genes for the consensus genes; for
    Mxd1 the LFC sign is *opposite* so the concordance gate excludes it.
    """
    rows = [
        {"gene_id": "ENSMUSG01", "gene_name": "Fos",   "log2FoldChange": 2.2, "pvalue": 1e-10, "padj": 1e-8,  "tf_family": "bZIP"},
        {"gene_id": "ENSMUSG02", "gene_name": "Jun",   "log2FoldChange": 1.5, "pvalue": 1e-8,  "padj": 1e-6,  "tf_family": "bZIP"},
        {"gene_id": "ENSMUSG03", "gene_name": "Atf3",  "log2FoldChange": 1.0, "pvalue": 1e-5,  "padj": 1e-3,  "tf_family": "bZIP"},
        {"gene_id": "ENSMUSG04", "gene_name": "Dnmt3a","log2FoldChange": -1.3,"pvalue": 1e-6,  "padj": 1e-4,  "tf_family": "-"},
        {"gene_id": "ENSMUSG05", "gene_name": "Foxp1", "log2FoldChange": -0.8,"pvalue": 1e-4,  "padj": 1e-2,  "tf_family": "Forkhead"},
        {"gene_id": "ENSMUSG06", "gene_name": "Mxd1",  "log2FoldChange": -1.4,"pvalue": 1e-6,  "padj": 1e-4,  "tf_family": "bHLH"},
        {"gene_id": "ENSMUSG07", "gene_name": "Sox2",  "log2FoldChange": 1.0, "pvalue": 0.6,   "padj": 0.8,   "tf_family": "HMG"},
    ]
    return _tsv(_deg_frame(rows))


@pytest.fixture
def deg_genetic_4h_bytes() -> bytes:
    """4 h genetic file. Same five consensus genes as 2 h, but Foxp1 flips
    sign — so it ends up in the 'reversed' trajectory class.
    """
    rows = [
        {"gene_id": "ENSMUSG01", "gene_name": "Fos",   "log2FoldChange": 3.0, "pvalue": 1e-12, "padj": 1e-10, "tf_family": "bZIP"},
        {"gene_id": "ENSMUSG02", "gene_name": "Jun",   "log2FoldChange": 1.0, "pvalue": 1e-7,  "padj": 1e-5,  "tf_family": "bZIP"},
        {"gene_id": "ENSMUSG03", "gene_name": "Atf3",  "log2FoldChange": 0.8, "pvalue": 1e-4,  "padj": 1e-2,  "tf_family": "bZIP"},
        {"gene_id": "ENSMUSG04", "gene_name": "Dnmt3a","log2FoldChange": -2.0,"pvalue": 1e-9,  "padj": 1e-7,  "tf_family": "-"},
        {"gene_id": "ENSMUSG05", "gene_name": "Foxp1", "log2FoldChange":  1.2,"pvalue": 1e-6,  "padj": 1e-4,  "tf_family": "Forkhead"},
    ]
    return _tsv(_deg_frame(rows))


@pytest.fixture
def deg_vehicle_4h_bytes() -> bytes:
    rows = [
        {"gene_id": "ENSMUSG01", "gene_name": "Fos",   "log2FoldChange": 2.6, "pvalue": 1e-11, "padj": 1e-9,  "tf_family": "bZIP"},
        {"gene_id": "ENSMUSG02", "gene_name": "Jun",   "log2FoldChange": 0.9, "pvalue": 1e-6,  "padj": 1e-4,  "tf_family": "bZIP"},
        {"gene_id": "ENSMUSG03", "gene_name": "Atf3",  "log2FoldChange": 0.6, "pvalue": 1e-3,  "padj": 0.04,  "tf_family": "bZIP"},
        {"gene_id": "ENSMUSG04", "gene_name": "Dnmt3a","log2FoldChange": -1.7,"pvalue": 1e-8,  "padj": 1e-6,  "tf_family": "-"},
        {"gene_id": "ENSMUSG05", "gene_name": "Foxp1", "log2FoldChange":  1.0,"pvalue": 1e-5,  "padj": 1e-3,  "tf_family": "Forkhead"},
    ]
    return _tsv(_deg_frame(rows))


@pytest.fixture
def sample_genetic_bytes() -> bytes:
    """DEG file with per-sample count columns in the new <id>_<code><tp> shape.

    Four CRE+CNO samples (S###_TC2) and four WT+CNO samples (S###_WC2),
    interleaved on purpose to exercise token-based grouping (the audit
    regression case for _column_token).
    """
    rows = [
        {"gene_id": "ENSMUSG01", "gene_name": "Fos",
         "S897_TC2": 800, "T296_TC2": 750, "S895_TC2": 820, "S535_TC2": 780,
         "S869_WC2": 100, "S894_WC2": 110, "S896_WC2": 95,  "S893_WC2": 105,
         "log2FoldChange": 2.5, "pvalue": 1e-10, "padj": 1e-8},
        {"gene_id": "ENSMUSG02", "gene_name": "Jun",
         "S897_TC2": 200, "T296_TC2": 210, "S895_TC2": 195, "S535_TC2": 205,
         "S869_WC2": 80,  "S894_WC2": 85,  "S896_WC2": 90,  "S893_WC2": 75,
         "log2FoldChange": 1.5, "pvalue": 1e-8, "padj": 1e-6},
    ]
    df = pd.DataFrame(rows)
    return _tsv(df)


@pytest.fixture
def sample_vehicle_interleaved_bytes() -> bytes:
    """Vehicle file with CRE+CNO and CRE+SAL columns *interleaved*.

    This is the audit regression case: the old _column_token regex was
    keyed on the 'S' prefix and would silently fall through to the
    positional fallback. With interleaved columns the positional split
    misgroups every sample. The current loader resolves them via tokens.
    """
    rows = [
        {"gene_id": "ENSMUSG01", "gene_name": "Fos",
         "T296_TS2": 50, "S897_TC2": 800, "T298_TS2": 55,
         "S870_TC2": 750, "T641_TS2": 60, "S895_TC2": 820,
         "T640_TS2": 58, "S535_TC2": 780,
         "log2FoldChange": 2.5, "pvalue": 1e-10, "padj": 1e-8},
    ]
    df = pd.DataFrame(rows)
    return _tsv(df)
