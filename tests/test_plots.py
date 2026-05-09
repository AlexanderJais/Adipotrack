"""Smoke + regression tests for plots.py.

These use the Agg backend so they run without a display, and are kept
narrow on purpose — the goal is to catch import-time regressions and
the specific bugs the audit found, not to validate exact rendering.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from plots import (
    fig_to_bytes,
    heatmap,
    lfc_scatter,
    pathway_enrichment_dot,
    sample_corr_heatmap,
    tf_enrichment_dot,
    tf_lfc_panel,
    trajectory_lines,
    upset_plot,
    volcano,
)


# A tiny trajectory frame the figure functions can chew on.
@pytest.fixture
def traj() -> pd.DataFrame:
    df = pd.DataFrame({
        "gene_name":     ["Fos",          "Jun",          "Dnmt3a",         "Foxp1"],
        "class":         ["sustained_up", "transient_up", "sustained_down", "reversed"],
        "lfc_2h":        [2.2,            1.5,            -1.3,             -0.8],
        "lfc_4h":        [2.6,            0.9,            -1.7,             1.0],
        "lfc_genetic_2h":[2.5,            1.8,            -1.5,             -1.0],
        "lfc_vehicle_2h":[2.2,            1.5,            -1.3,             -0.8],
        "lfc_genetic_4h":[3.0,            1.0,            -2.0,             1.2],
        "lfc_vehicle_4h":[2.6,            0.9,            -1.7,             1.0],
        "padj_genetic_2h":[1e-8, 1e-7, 1e-5, 1e-3],
        "padj_vehicle_2h":[1e-8, 1e-6, 1e-4, 1e-2],
        "padj_genetic_4h":[1e-10, 1e-5, 1e-7, 1e-4],
        "padj_vehicle_4h":[1e-9, 1e-4, 1e-6, 1e-3],
        "tf_family":     ["bZIP",         "bZIP",         "-",              "Forkhead"],
        "delta_lfc":     [0.4,            -0.6,           -0.4,             1.8],
    })
    df["class"] = pd.Categorical(
        df["class"],
        categories=["sustained_up", "transient_up", "sustained_down",
                    "transient_down", "reversed"],
        ordered=True,
    )
    return df


@pytest.fixture
def deg_for_volcano() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "gene_name": [f"g{i}" for i in range(n)],
        "log2FoldChange": rng.normal(0, 1.0, size=n),
        "padj": rng.uniform(1e-12, 1.0, size=n),
        "pvalue": rng.uniform(1e-12, 1.0, size=n),
    })


def test_volcano_returns_figure(deg_for_volcano):
    fig = volcano(deg_for_volcano, title="test")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_lfc_scatter_runs(traj):
    fig = lfc_scatter(traj)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_trajectory_lines_runs(traj):
    fig = trajectory_lines(traj)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_heatmap_runs(traj):
    fig = heatmap(traj, max_genes=10)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_heatmap_clustered_runs(traj):
    pytest.importorskip("scipy.cluster.hierarchy")
    fig = heatmap(traj, max_genes=10, cluster=True)
    assert isinstance(fig, plt.Figure)
    # Two main axes: dendrogram + heatmap. (Colorbar adds a third, that's
    # fine — we just want >= 2 from the gridspec.)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_heatmap_clustered_reorders_rows(traj):
    """Clustering must change row order vs the class-sorted default for any
    non-trivial input — otherwise the dendrogram is decorative only."""
    pytest.importorskip("scipy.cluster.hierarchy")
    # Pick out heatmap row labels (gene names) from the heatmap's y-tick
    # text in both modes and confirm they differ.
    fig_classic = heatmap(traj, max_genes=10, cluster=False)
    classic_order = [t.get_text() for t in fig_classic.axes[0].get_yticklabels()]
    plt.close(fig_classic)

    fig_clustered = heatmap(traj, max_genes=10, cluster=True)
    # The first axis in cluster mode is the dendrogram (axis-off); pick the
    # heatmap axis by finding the one with y-tick labels.
    heatmap_ax = next(
        ax for ax in fig_clustered.axes
        if any(t.get_text() for t in ax.get_yticklabels())
    )
    clustered_order = [t.get_text() for t in heatmap_ax.get_yticklabels()]
    plt.close(fig_clustered)

    assert classic_order != clustered_order
    # The same set of genes should appear in both modes.
    assert set(classic_order) == set(clustered_order)


def test_heatmap_clustered_single_row_falls_back_gracefully(traj):
    """A single row can't be clustered; the function should still return a
    Figure rather than raising."""
    pytest.importorskip("scipy.cluster.hierarchy")
    one = traj.iloc[:1]
    fig = heatmap(one, max_genes=5, cluster=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_tf_lfc_panel_runs(traj):
    fig = tf_lfc_panel(traj)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_upset_plot_runs():
    fig = upset_plot({
        "A": {"g1", "g2", "g3"},
        "B": {"g2", "g3", "g4"},
        "C": {"g3", "g5"},
        "D": {"g1", "g6"},
    })
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_fig_to_bytes_returns_pdf_and_closes_by_default():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    n_open_before = len(plt.get_fignums())
    out = fig_to_bytes(fig, "pdf")
    assert out.startswith(b"%PDF")
    n_open_after = len(plt.get_fignums())
    assert n_open_after == n_open_before - 1, "fig_to_bytes should close the figure"


def test_fig_to_bytes_close_false_keeps_figure_open():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig_to_bytes(fig, "pdf", close=False)
    assert fig.number in plt.get_fignums()
    plt.close(fig)


# ---- Regression tests for audit bugs ---------------------------------------


def test_tf_enrichment_dot_no_double_render_for_offscale():
    """Audit bug #2 regression: off-scale points were being drawn both as a
    clipped dot at ±log_cap by the main scatter AND as a triangle marker.

    Build a tiny enrichment table with one off-high (inf OR), one off-low
    (very negative log_or), one undefined (OR=0), and one in-range row.
    The main scatter should hold only the in-range row; the off-scale and
    undefined rows are scattered separately.
    """
    enr = pd.DataFrame({
        "tf_family":   ["off_high", "off_low", "undef", "ok"],
        "n_fg":        [3, 3, 3, 3],
        "n_fg_total":  [10, 10, 10, 10],
        "n_bg":        [1, 1000, 0, 50],   # OR -> inf, very small, 0, finite
        "n_bg_total":  [1000, 1000, 1000, 1000],
        "odds_ratio":  [np.inf, 1e-5, 0.0, 2.0],
        "p_value":     [1e-3, 0.5, 0.5, 0.05],
        "q_value":     [1e-3, 0.5, 0.5, 0.05],
    })
    fig = tf_enrichment_dot(enr)
    ax = fig.axes[0]

    # Inspect the main PathCollection (the colour-mapped scatter — first
    # collection added). Its offsets should contain one point: only the
    # in-range "ok" row. The off_high/off_low/undef rows are drawn by
    # subsequent scatter() calls.
    main_offsets = ax.collections[0].get_offsets()
    assert len(main_offsets) == 1, (
        "main scatter should contain exactly one in-range row; if it has more, "
        "off-scale points are being double-rendered (audit bug #2)"
    )
    plt.close(fig)


def test_pathway_enrichment_dot_runs():
    enr = pd.DataFrame({
        "set_name":   ["A_LONG_PATHWAY_NAME_THAT_WILL_BE_TRUNCATED_BECAUSE_IT_EXCEEDS_THE_LIMIT_BY_A_LOT", "PATHWAY_B", "PATHWAY_C"],
        "n_fg":       [5, 3, 2],
        "n_fg_total": [10, 10, 10],
        "n_bg":       [10, 30, 50],
        "n_bg_total": [1000, 1000, 1000],
        "odds_ratio": [10.0, 2.0, 0.5],
        "p_value":    [1e-4, 0.05, 0.5],
        "q_value":    [1e-4, 0.05, 0.5],
    })
    fig = pathway_enrichment_dot(enr, top_n=3)
    ax = fig.axes[0]
    # Long names get truncated with an ellipsis in the y-tick label.
    yticks = [t.get_text() for t in ax.get_yticklabels()]
    assert any(label.endswith("…") for label in yticks)
    plt.close(fig)


def test_pathway_enrichment_dot_empty():
    empty = pd.DataFrame(columns=[
        "set_name", "n_fg", "n_fg_total", "n_bg", "n_bg_total",
        "odds_ratio", "p_value", "q_value",
    ])
    fig = pathway_enrichment_dot(empty)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_sample_corr_heatmap_runs():
    corr = pd.DataFrame(
        np.eye(4),
        index=[f"S{i}" for i in range(4)],
        columns=[f"S{i}" for i in range(4)],
    )
    metadata = pd.DataFrame({
        "sample": [f"S{i}" for i in range(4)],
        "group":  ["A", "A", "B", "B"],
    })
    fig = sample_corr_heatmap(corr, metadata, title="t")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
