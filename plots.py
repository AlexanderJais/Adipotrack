"""Publication-grade matplotlib figures.

Style targets Nature/Cell-style figures: small sans-serif type, hairline axes,
top/right spines removed, colorblind-safe Okabe-Ito palette, vector output.
"""

from __future__ import annotations

from io import BytesIO

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import (
    CLASS_COLORS,
    CLASS_ORDER,
    CLOCK_FAMILY_COLORS,
    CLOCK_ROLES,
    fit_phase_cosine,
)


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        # Render mathtext (used for subscripts like log$_2$) in the regular
        # body font rather than the italic math font, and avoid Unicode
        # subscript glyphs that the embedded Arial/type-42 font lacks (they
        # otherwise render as missing-glyph boxes in the PDF).
        "mathtext.default": "regular",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
    })


def fig_to_bytes(fig: plt.Figure, fmt: str = "pdf", close: bool = True) -> bytes:
    """Render ``fig`` to bytes; closes it by default to avoid leaks across
    Streamlit reruns. Pass ``close=False`` if you need to keep using the
    figure (e.g. for a follow-up ``st.pyplot`` call)."""
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    if close:
        plt.close(fig)
    return buf.getvalue()


def volcano(
    df: pd.DataFrame,
    title: str,
    highlight: set[str] | None = None,
    padj_thresh: float = 0.05,
    lfc_thresh: float = 0.0,
    label_top_n: int = 12,
    x_lim: float | None = None,
) -> plt.Figure:
    apply_style()
    d = df.copy()
    d["neglog10padj"] = -np.log10(d["padj"].clip(lower=1e-300))
    sig = (d["padj"] < padj_thresh) & (d["log2FoldChange"].abs() >= lfc_thresh)

    # X-axis: a caller-supplied ``x_lim`` fixes the |log2FC| range so panels
    # are directly comparable across experiments; genes beyond it are drawn as
    # off-scale triangles at the boundary (so a single huge outlier doesn't
    # stretch every panel). When ``x_lim`` is None, fall back to auto-scaling
    # from the significant genes. The Y-axis (−log10 P) is always per-panel
    # because significance varies wildly between timepoints.
    if sig.any():
        y_ref = d.loc[sig, "neglog10padj"].max()
    else:
        y_ref = d["neglog10padj"].quantile(0.99)
    if x_lim is None:
        if sig.any():
            x_ref = d.loc[sig, "log2FoldChange"].abs().max()
        else:
            x_ref = d["log2FoldChange"].abs().quantile(0.99)
        x_lim = max(float(x_ref) * 1.2, 1.0)
    else:
        x_lim = float(x_lim)
    y_lim = max(float(y_ref) * 1.10, -np.log10(padj_thresh) * 1.5)

    x_clip = d["log2FoldChange"].clip(-x_lim, x_lim)
    y_clip = d["neglog10padj"].clip(upper=y_lim)
    off_x = d["log2FoldChange"].abs() > x_lim
    off_y = d["neglog10padj"] > y_lim

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ns = ~sig
    ax.scatter(x_clip[ns & ~off_x & ~off_y], y_clip[ns & ~off_x & ~off_y],
               s=3, c="#BBBBBB", alpha=0.5, linewidths=0, rasterized=True)
    up = sig & (d["log2FoldChange"] > 0)
    dn = sig & (d["log2FoldChange"] < 0)
    ax.scatter(x_clip[up], y_clip[up], s=5, c="#D55E00", alpha=0.85,
               linewidths=0, rasterized=True)
    ax.scatter(x_clip[dn], y_clip[dn], s=5, c="#0072B2", alpha=0.85,
               linewidths=0, rasterized=True)

    # Off-scale markers at the boundary
    if off_x.any():
        right = off_x & (d["log2FoldChange"] > 0)
        left = off_x & (d["log2FoldChange"] < 0)
        ax.scatter([x_lim] * int(right.sum()), y_clip[right],
                   marker=">", s=10, c="#888888", linewidths=0)
        ax.scatter([-x_lim] * int(left.sum()), y_clip[left],
                   marker="<", s=10, c="#888888", linewidths=0)
    if off_y.any():
        ax.scatter(x_clip[off_y], [y_lim] * int(off_y.sum()),
                   marker="^", s=10, c="#888888", linewidths=0)

    if highlight:
        h_mask = d["gene_name"].isin(highlight)
        ax.scatter(x_clip[h_mask], y_clip[h_mask],
                   s=12, facecolors="none", edgecolors="black", linewidths=0.4)

    ax.axhline(-np.log10(padj_thresh), color="black", lw=0.4, ls="--")
    if lfc_thresh > 0:
        ax.axvline(lfc_thresh, color="black", lw=0.4, ls="--")
        ax.axvline(-lfc_thresh, color="black", lw=0.4, ls="--")

    # Gene labels (top by significance) with adjustText repulsion
    top = d[sig].nlargest(label_top_n, "neglog10padj")
    anchor_x = top["log2FoldChange"].clip(-x_lim, x_lim).to_numpy(dtype=float)
    anchor_y = top["neglog10padj"].clip(upper=y_lim).to_numpy(dtype=float)
    texts = [
        ax.text(x, y, name, fontsize=5)
        for x, y, name in zip(anchor_x, anchor_y, top["gene_name"])
    ]
    if texts:
        try:
            from adjustText import adjust_text
            try:
                adjust_text(
                    texts,
                    x=anchor_x.tolist(), y=anchor_y.tolist(), ax=ax,
                    arrowprops=dict(arrowstyle="-", color="black",
                                    lw=0.3, shrinkA=0, shrinkB=2),
                    only_move={"text": "xy", "static": "xy", "explode": "xy"},
                    expand=(1.2, 1.4),
                    avoid_self=True,
                )
            except TypeError:
                # adjustText 0.x signature
                adjust_text(
                    texts,
                    x=anchor_x.tolist(), y=anchor_y.tolist(), ax=ax,
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.3),
                    expand_points=(1.2, 1.4),
                    only_move={"points": "y", "texts": "xy"},
                )
        except ImportError:
            pass

    # Up/down counts
    n_up = int(up.sum())
    n_dn = int(dn.sum())
    ax.text(0.02, 0.98, f"↓ {n_dn}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6, color="#0072B2")
    ax.text(0.98, 0.98, f"↑ {n_up}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color="#D55E00")

    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xlabel("log$_2$ fold change")
    ax.set_ylabel("−log$_{10}$ adjusted P")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def lfc_scatter(traj: pd.DataFrame) -> plt.Figure:
    """LFC at 2h vs LFC at 4h for genes consensus at both timepoints."""
    apply_style()
    fig, ax = plt.subplots(figsize=(3.6, 3.4))

    counts = traj["class"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    for cls in CLASS_ORDER:
        sub = traj[traj["class"] == cls]
        if sub.empty:
            continue
        ax.scatter(sub["lfc_2h"], sub["lfc_4h"],
                   s=10, c=CLASS_COLORS[cls], alpha=0.85,
                   linewidths=0,
                   label=f"{cls.replace('_', ' ')}  (n={counts[cls]})")

    # Symmetric, robust limits: 99th percentile padded — outliers stay visible
    # but don't compress the bulk of the consensus genes.
    arr = np.concatenate([traj["lfc_2h"].to_numpy(),
                          traj["lfc_4h"].to_numpy()])
    if arr.size:
        q = float(np.nanquantile(np.abs(arr), 0.99))
        lim = max(q * 1.15, 1.0)
    else:
        lim = 1.0
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.4, ls="--")
    ax.axhline(0, color="black", lw=0.3)
    ax.axvline(0, color="black", lw=0.3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Spearman ρ on the consensus set (rank-based, robust to outliers)
    if len(traj) >= 3:
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(traj["lfc_2h"], traj["lfc_4h"])
            rho_text = f"Spearman ρ = {rho:.2f}"
        except ImportError:
            r = float(np.corrcoef(traj["lfc_2h"], traj["lfc_4h"])[0, 1])
            rho_text = f"Pearson r = {r:.2f}"
        ax.text(0.03, 0.97, rho_text, transform=ax.transAxes,
                ha="left", va="top", fontsize=6)

    ax.set_xlabel("log$_2$FC at 2 h (CRE+CNO vs CRE+SAL)")
    ax.set_ylabel("log$_2$FC at 4 h (CRE+CNO vs CRE+SAL)")
    ax.set_title(f"Trajectory genes (n = {len(traj)})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def trajectory_lines(
    traj: pd.DataFrame,
    label_top_n: int = 8,
    panel_width: float = 2.8,
    panel_height: float = 4.2,
) -> plt.Figure:
    """One line per gene across timepoints, faceted by class.

    Each panel reserves a right-hand label gutter. Labelled genes get a dot at
    their final-timepoint endpoint and a leader line to the label, which is
    spread vertically so labels never overlap while still pointing at the line
    they belong to.
    """
    apply_style()
    classes = [c for c in CLASS_ORDER if (traj["class"] == c).any()]
    n = len(classes)
    if n == 0:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.text(0.5, 0.5, "No consensus genes", ha="center", va="center")
        ax.axis("off")
        return fig

    # A leading 20 min reference point is drawn when add_earlier_reference has
    # populated ``lfc_20m``. Genes not tested at 20 min carry NaN there, so
    # their line simply starts at the 2 h point.
    if "lfc_20m" in traj.columns:
        x_cols = ["lfc_20m", "lfc_2h", "lfc_4h"]
        x_labels = ["20 m", "2 h", "4 h"]
    else:
        x_cols = ["lfc_2h", "lfc_4h"]
        x_labels = ["2 h", "4 h"]
    x_pos = list(range(len(x_cols)))
    last_x = x_pos[-1]

    # Shared symmetric y-range so panels can be compared at a glance. Use the
    # full data extent (not a 99th percentile) so no trajectory line is ever
    # clipped by the panel boundary; a little headroom leaves space for labels.
    arr = np.concatenate([traj[c].to_numpy(dtype=float) for c in x_cols])
    finite = arr[np.isfinite(arr)]
    if finite.size:
        y_lim = max(float(np.abs(finite).max()) * 1.08, 1.0)
    else:
        y_lim = 1.0

    fig, axes = plt.subplots(
        1, n, figsize=(panel_width * n, panel_height), sharey=True,
    )
    if n == 1:
        axes = [axes]

    label_x = last_x + 0.24
    for ax, cls in zip(axes, classes):
        sub = traj[traj["class"] == cls]

        # Per-gene trajectories
        for _, r in sub.iterrows():
            ax.plot(x_pos, [r[c] for c in x_cols],
                    color=CLASS_COLORS[cls], lw=0.5, alpha=0.45,
                    zorder=2)

        ax.axhline(0, color="black", lw=0.3, zorder=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlim(-0.15, last_x + 0.75)  # extra room on the right for labels
        ax.set_ylim(-y_lim, y_lim)
        ax.set_title(f"{cls.replace('_', ' ')}\n(n = {len(sub)})")

        # Right-edge labels: pick the largest-|LFC| genes, mark each one's
        # endpoint with a dot, and draw a leader line to a vertically-spread
        # label so the label unambiguously belongs to its trajectory.
        if not sub.empty:
            extreme = sub.iloc[
                sub["lfc_4h"].abs().to_numpy().argsort()[::-1]
            ].head(label_top_n)
            y_end = extreme["lfc_4h"].to_numpy(dtype=float)
            names = list(extreme["gene_name"])

            label_y = _spread_labels(y_end, min_gap=y_lim * 0.09,
                                     lo=-y_lim, hi=y_lim)
            for ye, yl, name in zip(y_end, label_y, names):
                ax.plot([last_x], [ye], "o", ms=2.2,
                        color=CLASS_COLORS[cls], zorder=5, clip_on=False)
                ax.plot([last_x, label_x - 0.02], [ye, yl],
                        color="grey", lw=0.3, zorder=4, clip_on=False)
                ax.text(label_x, yl, name, fontsize=5.5, va="center",
                        ha="left", zorder=6)

    axes[0].set_ylabel("log$_2$ fold change (CRE+CNO vs CRE+SAL)")
    fig.tight_layout()
    return fig


def tf_lfc_panel(traj: pd.DataFrame, max_per_class: int = 12) -> plt.Figure:
    """Horizontal LFC bars of TFs in the trajectory set, faceted by class.

    Sign of the bar = direction at 4 h (canonical effect size).
    """
    apply_style()
    if traj.empty or "tf_family" not in traj.columns:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.text(0.5, 0.5, "No TF annotations available",
                ha="center", va="center")
        ax.axis("off")
        return fig

    tfs = traj[traj["tf_family"].astype(str).str.strip().ne("-")
               & traj["tf_family"].notna()].copy()
    if tfs.empty:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.text(0.5, 0.5, "No TFs in the trajectory set",
                ha="center", va="center")
        ax.axis("off")
        return fig

    classes = [c for c in CLASS_ORDER if (tfs["class"] == c).any()]
    n = len(classes)
    fig, axes = plt.subplots(
        1, n, figsize=(2.4 * n, 0.18 * max_per_class + 1.2), sharex=True,
    )
    if n == 1:
        axes = [axes]

    arr = np.concatenate([tfs["lfc_2h"].to_numpy(), tfs["lfc_4h"].to_numpy()])
    x_lim = max(float(np.nanquantile(np.abs(arr), 0.99)) * 1.15, 1.0)

    for ax, cls in zip(axes, classes):
        sub = tfs[tfs["class"] == cls]
        sub = sub.iloc[sub["lfc_4h"].abs().to_numpy().argsort()[::-1]].head(max_per_class)
        # Plot in ascending LFC so largest bars sit at the top
        sub = sub.iloc[sub["lfc_4h"].to_numpy().argsort()]
        y = np.arange(len(sub))
        ax.barh(y, sub["lfc_4h"], color=CLASS_COLORS[cls], height=0.7,
                edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{g}  · {fam}" for g, fam in zip(sub["gene_name"], sub["tf_family"])],
            fontsize=5,
        )
        ax.axvline(0, color="black", lw=0.4)
        ax.set_xlabel("log$_2$FC at 4 h")
        ax.set_title(f"{cls.replace('_', ' ')}\n(n = {len(sub)} of {(tfs['class'] == cls).sum()})")
    # Hoisted out of the loop — sharex propagates to all panels in one go.
    axes[0].set_xlim(-x_lim, x_lim)
    fig.tight_layout()
    return fig


def tf_enrichment_dot(enrichment: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Dot plot of TF-family enrichment. Dot size = n_fg, colour = -log10 q."""
    apply_style()
    if enrichment.empty:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.text(0.5, 0.5, "No TF families pass the size filter",
                ha="center", va="center")
        ax.axis("off")
        return fig

    d = enrichment.head(top_n).copy()
    d = d.iloc[::-1]  # so smallest q ends up at the top

    # All masks built in numpy so the boolean ops never mix Series and
    # ndarray. ±log_cap clamps values beyond the cap; rows with undefined
    # OR (e.g. 0 in foreground or background) end up as NaN and are
    # rendered as hollow rings at x = 0.
    or_arr = d["odds_ratio"].to_numpy(dtype=float)
    or_pos = np.where(or_arr > 0, or_arr, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_or = np.log2(or_pos)
    log_cap = 6.0  # ±6 in log2 space ≈ 64-fold enrichment; plenty of headroom
    log_or_capped = np.clip(log_or, -log_cap, log_cap)

    off_high = (or_arr == np.inf) | (log_or > log_cap)
    off_low = log_or < -log_cap
    nan_mask = ~np.isfinite(log_or) & ~off_high & ~off_low
    # Off-scale points are drawn as triangles below; exclude them from the
    # main scatter so they don't get a clipped dot at ±log_cap on top.
    main_mask = ~(nan_mask | off_high | off_low)

    n_fg = d["n_fg"].to_numpy()
    mlog10q = -np.log10(d["q_value"].clip(lower=1e-300).to_numpy(dtype=float))
    y_pos = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(3.8, 0.22 * len(d) + 1.0))
    sc = ax.scatter(
        log_or_capped[main_mask], y_pos[main_mask],
        s=n_fg[main_mask] * 12 + 10,
        c=mlog10q[main_mask], cmap="viridis",
        edgecolor="black", linewidths=0.3,
    )
    if nan_mask.any():
        ax.scatter(
            np.zeros(nan_mask.sum()), y_pos[nan_mask],
            s=n_fg[nan_mask] * 12 + 10,
            facecolors="none", edgecolor="grey", linewidths=0.3,
        )
    if off_high.any():
        ax.scatter(
            np.full(off_high.sum(), log_cap), y_pos[off_high],
            marker=">", s=30, c="black",
        )
    if off_low.any():
        ax.scatter(
            np.full(off_low.sum(), -log_cap), y_pos[off_low],
            marker="<", s=30, c="black",
        )
    ax.set_xlim(-log_cap * 1.05, log_cap * 1.05)
    ax.axvline(0, color="black", lw=0.3, ls="--")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["tf_family"], fontsize=6)
    ax.set_xlabel("log$_2$ odds ratio (foreground vs background)")
    ax.set_title(f"TF-family enrichment (top {len(d)})")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("−log$_{10}$ q", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.4)

    # Size legend
    handles = []
    for k in (1, 5, 10):
        handles.append(plt.scatter([], [], s=k * 12 + 10,
                                   facecolor="white", edgecolor="black",
                                   linewidths=0.3, label=str(k)))
    ax.legend(handles=handles, title="n in foreground",
              loc="lower right", bbox_to_anchor=(1.02, -0.02),
              fontsize=5, title_fontsize=6, labelspacing=0.6,
              borderpad=0.4, handletextpad=0.4)
    fig.tight_layout()
    return fig


# ---- Replicate QC ----------------------------------------------------------

def pca_scatter(
    scores: np.ndarray,
    var_explained: np.ndarray,
    metadata: pd.DataFrame,
    title: str,
) -> plt.Figure:
    """PC1 vs PC2 coloured by group, sample IDs annotated."""
    apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    groups = metadata["group"].unique().tolist()
    palette = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9"]
    for i, grp in enumerate(groups):
        mask = (metadata["group"] == grp).to_numpy()
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   s=24, c=palette[i % len(palette)], alpha=0.9,
                   edgecolor="black", linewidths=0.3, label=grp)
    for (x, y), name in zip(scores[:, :2], metadata["sample"]):
        ax.annotate(name, (x, y), fontsize=4.5, xytext=(3, 3),
                    textcoords="offset points")
    ax.axhline(0, color="black", lw=0.3, alpha=0.5)
    ax.axvline(0, color="black", lw=0.3, alpha=0.5)
    ax.set_xlabel(f"PC1 ({var_explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1] * 100:.1f}%)" if len(var_explained) > 1
                  else "PC2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=5)
    fig.tight_layout()
    return fig


def sample_corr_heatmap(corr: pd.DataFrame, metadata: pd.DataFrame,
                        title: str) -> plt.Figure:
    """Pairwise correlation heatmap of samples with group labels."""
    apply_style()
    n = len(corr)
    arr = corr.to_numpy()
    # nanmin so a NaN cell (e.g. a constant sample column) doesn't blank the
    # whole panel via vmin=NaN; fall back to 0 if the entire matrix is NaN.
    finite = arr[np.isfinite(arr)]
    vmin = float(finite.min()) if finite.size else 0.0
    fig, ax = plt.subplots(figsize=(0.32 * n + 1.6, 0.32 * n + 1.4))
    im = ax.imshow(arr, vmin=vmin, vmax=1.0,
                   cmap="magma", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
    ax.set_yticklabels(corr.index, fontsize=5)
    ax.tick_params(left=False, bottom=False)

    # Group separator lines based on metadata order
    group_order = metadata.set_index("sample").loc[corr.columns, "group"].tolist()
    boundaries = [i for i in range(1, n) if group_order[i] != group_order[i - 1]]
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", lw=1.0)
        ax.axvline(b - 0.5, color="white", lw=1.0)

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Pearson r", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.4)
    fig.tight_layout()
    return fig


def _spread_labels(y, min_gap: float, lo: float, hi: float) -> np.ndarray:
    """Vertically de-overlap label positions while staying near their targets.

    Given desired y-positions ``y`` (one per label), return positions in the
    same order such that adjacent labels (in sort order) are at least
    ``min_gap`` apart, kept within ``[lo, hi]`` where possible and otherwise
    spread evenly. Deterministic — no dependency on adjustText — so the leader
    lines drawn to these positions always line up.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    order = np.argsort(y)
    s = y[order].copy()
    # Push up from the bottom to enforce the minimum gap.
    for i in range(1, len(s)):
        if s[i] - s[i - 1] < min_gap:
            s[i] = s[i - 1] + min_gap
    # If the stack overflowed the top, slide it down as a block.
    if s[-1] > hi:
        s -= s[-1] - hi
    # If it now underflows the bottom, either pin-and-repush or, when the
    # labels can't all fit at min_gap, distribute them evenly across the range.
    if s[0] < lo:
        if (len(s) - 1) * min_gap <= (hi - lo):
            s[0] = lo
            for i in range(1, len(s)):
                if s[i] - s[i - 1] < min_gap:
                    s[i] = s[i - 1] + min_gap
        else:
            s = np.linspace(lo, hi, len(s))
    out = np.empty_like(s)
    out[order] = s
    return out


def upset_plot(sets: dict[str, set[str]], max_rows: int = 20) -> plt.Figure:
    """N-way overlap rendered as an UpSet-style bar chart (a 4-way Venn is
    unreadable).

    The all-N intersection is always pinned at the front so it's never dropped
    by the size cap; remaining cells follow in size-descending order.
    """
    apply_style()
    from itertools import combinations

    keys = list(sets.keys())
    rows: list[tuple[tuple[str, ...], int]] = []
    for r in range(1, len(keys) + 1):
        for combo in combinations(keys, r):
            inter = set.intersection(*(sets[k] for k in combo))
            others = set.union(*(sets[k] for k in keys if k not in combo)) if r < len(keys) else set()
            exclusive = inter - others
            if exclusive:
                rows.append((combo, len(exclusive)))

    all_n = tuple(keys)
    pinned = [r for r in rows if r[0] == all_n]
    rest = sorted([r for r in rows if r[0] != all_n], key=lambda x: -x[1])
    rows = (pinned + rest)[:max_rows]

    if not rows:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.text(0.5, 0.5, "No significant genes", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, (ax_bar, ax_dot) = plt.subplots(
        2, 1, figsize=(0.45 * len(rows) + 1.2, 3.4),
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.05},
        sharex=True,
    )
    xs = np.arange(len(rows))
    ax_bar.bar(xs, [r[1] for r in rows], color="#333333", width=0.7)
    for x, (_, n) in zip(xs, rows):
        ax_bar.text(x, n, str(n), ha="center", va="bottom", fontsize=5)
    ax_bar.set_ylabel("genes (exclusive)")

    for yi, k in enumerate(keys):
        for xi, (combo, _) in enumerate(rows):
            on = k in combo
            ax_dot.plot(xi, yi, "o",
                        color="black" if on else "#DDDDDD",
                        markersize=4)
    for xi, (combo, _) in enumerate(rows):
        ys = [keys.index(k) for k in combo]
        if len(ys) > 1:
            ax_dot.plot([xi, xi], [min(ys), max(ys)], "-", color="black", lw=0.6)
    ax_dot.set_yticks(range(len(keys)))
    ax_dot.set_yticklabels(keys)
    ax_dot.set_xticks([])
    ax_dot.set_ylim(-0.5, len(keys) - 0.5)
    ax_dot.invert_yaxis()
    for spine in ("top", "right", "bottom", "left"):
        ax_dot.spines[spine].set_visible(False)

    fig.suptitle("Significant gene set overlaps", fontsize=8, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def heatmap(traj: pd.DataFrame, max_genes: int = 60) -> plt.Figure:
    """Heatmap of LFC across the per-timepoint comparisons for top consensus
    genes. Includes a leading 20 min block when those reference columns are
    present (see ``add_earlier_reference``)."""
    apply_style()
    if traj.empty:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.text(0.5, 0.5, "No consensus genes", ha="center", va="center")
        ax.axis("off")
        return fig

    d = traj.copy()
    d = d.sort_values(["class", "lfc_4h"], ascending=[True, False])
    if len(d) > max_genes:
        per_class = max(2, max_genes // max(1, d["class"].nunique()))
        # Iterate groups explicitly — avoids the pandas 2.2 FutureWarning
        # about implicit `include_groups` in DataFrameGroupBy.apply.
        chunks = []
        for _, g in d.groupby("class", observed=True, sort=False):
            top_idx = g["lfc_4h"].abs().sort_values(ascending=False).index[:per_class]
            chunks.append(g.loc[top_idx])
        d = pd.concat(chunks) if chunks else d.iloc[0:0]

    # Timepoint blocks, earliest first. The 20 min reference block is included
    # only when add_earlier_reference has populated its columns; genes not
    # tested at 20 min leave NaN cells (drawn blank).
    blocks = []
    if {"lfc_genetic_20m", "lfc_vehicle_20m"} <= set(d.columns):
        blocks.append(("lfc_genetic_20m", "lfc_vehicle_20m", "20m vs WT", "20m vs SAL"))
    blocks.append(("lfc_genetic_2h", "lfc_vehicle_2h", "2h vs WT", "2h vs SAL"))
    blocks.append(("lfc_genetic_4h", "lfc_vehicle_4h", "4h vs WT", "4h vs SAL"))
    cols: list[str] = []
    col_labels: list[str] = []
    for gcol, vcol, glab, vlab in blocks:
        cols += [gcol, vcol]
        col_labels += [glab, vlab]
    mat = d[cols].to_numpy()

    # Symmetric, robust colour scale (99th percentile) so a single huge LFC
    # doesn't desaturate everything else.
    if mat.size:
        vmax = float(np.nanquantile(np.abs(mat), 0.99))
        vmax = max(vmax, 0.5)
    else:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 1.2, 0.13 * len(d) + 1.6))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")

    # Class separator lines + right-side class label bar
    classes_in_order = list(d["class"].astype(str))
    boundaries = [i for i in range(1, len(classes_in_order))
                  if classes_in_order[i] != classes_in_order[i - 1]]
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", lw=1.0)
        ax.axhline(b - 0.5, color="black", lw=0.4)

    # Vertical separators between timepoint blocks (every 2 columns)
    for b in range(2, len(cols), 2):
        ax.axvline(b - 0.5, color="white", lw=1.0)
        ax.axvline(b - 0.5, color="black", lw=0.4)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["gene_name"], fontsize=5)
    ax.tick_params(left=False, bottom=False)

    # Colour-coded class swatch on the right edge
    starts = [0] + boundaries + [len(classes_in_order)]
    for s, e in zip(starts[:-1], starts[1:]):
        cls = classes_in_order[s]
        ax.add_patch(plt.Rectangle(
            (len(col_labels) - 0.4, s - 0.5),
            0.18, e - s,
            facecolor=CLASS_COLORS.get(cls, "#999999"),
            edgecolor="none",
            transform=ax.transData,
            clip_on=False,
        ))
        ax.text(len(col_labels) - 0.1, (s + e) / 2 - 0.5,
                cls.replace("_", " "), fontsize=5, va="center", ha="left",
                clip_on=False)

    # Horizontal colourbar at the bottom so it doesn't collide with the
    # right-side class swatch and labels.
    cbar = fig.colorbar(
        im, ax=ax,
        orientation="horizontal",
        fraction=0.05, pad=0.16, aspect=30, shrink=0.6,
    )
    cbar.set_label("log$_2$FC", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.4)

    # Reserve space on the right for the class swatch + labels (~25% of axes
    # width is plenty for short class names at fontsize 5).
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    return fig


# ---- Circadian clock module ------------------------------------------------

def clock_bars(
    clock: pd.DataFrame,
    padj_thresh: float = 0.05,
    title: str = "Core clock genes",
) -> plt.Figure:
    """Diverging horizontal bars of clock-gene LFC, grouped by TTFL role.

    ``clock`` is a single-timepoint table from ``clock_gene_table`` (columns
    role, role_label, family, label, lfc, padj, found). Bars are coloured by
    LFC sign; genes that are not significant at ``padj_thresh`` are faded.
    Grouping by role makes the antiphase pattern (activators up while
    repressors/outputs go down) read at a glance.
    """
    apply_style()
    sub = clock[clock["found"]].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(3.4, 2))
        ax.text(0.5, 0.5, "No clock genes found in this file",
                ha="center", va="center")
        ax.axis("off")
        return fig

    # Role order (activator → accessory), and within a role by descending LFC.
    sub = sub.sort_values(["role", "lfc"], ascending=[True, False])
    n = len(sub)
    y = np.arange(n)[::-1]  # first row at the top
    lfc = sub["lfc"].to_numpy(dtype=float)
    padj = sub["padj"].to_numpy(dtype=float)
    sig = np.isfinite(padj) & (padj < padj_thresh)

    x_lim = max(float(np.nanmax(np.abs(lfc))) * 1.18, 1.0)

    fig, ax = plt.subplots(figsize=(4.2, 0.20 * n + 1.2))
    for yi, val, is_sig in zip(y, lfc, sig):
        color = "#D55E00" if val > 0 else "#0072B2"
        ax.barh(yi, val, height=0.7, color=color,
                alpha=1.0 if is_sig else 0.32,
                edgecolor=color if is_sig else "none",
                linewidth=0.0)

    ax.axvline(0, color="black", lw=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=5.5)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_xlabel("log$_2$ fold change")
    ax.tick_params(left=False)
    ax.set_title(title)

    # Role separators + a colour swatch/label per block in the right margin.
    roles = sub["role"].astype(str).tolist()
    role_labels = dict(CLOCK_ROLES)
    # families are consistent enough within a role for the swatch colour; use
    # the block's majority family so the swatch tracks the antiphase colouring.
    boundaries = [i for i in range(1, n) if roles[i] != roles[i - 1]]
    starts = [0] + boundaries + [n]
    swatch_x = x_lim * 1.04
    for s, e in zip(starts[:-1], starts[1:]):
        if s > 0:
            ax.axhline((y[s - 1] + y[s]) / 2, color="#dddddd", lw=0.6)
        block = sub.iloc[s:e]
        fam = block["family"].mode().iat[0]
        y_top, y_bot = y[s], y[e - 1]
        ax.add_patch(plt.Rectangle(
            (swatch_x, y_bot - 0.35), x_lim * 0.03, (y_top - y_bot) + 0.7,
            facecolor=CLOCK_FAMILY_COLORS.get(fam, "#999999"),
            edgecolor="none", clip_on=False,
        ))
        ax.text(swatch_x + x_lim * 0.06, (y_top + y_bot) / 2,
                role_labels.get(roles[s], roles[s]),
                fontsize=5.5, va="center", ha="left", clip_on=False)

    # Significance legend (colour already encodes up/down direction).
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#777777", edgecolor="none", label="padj < thresh"),
            Patch(facecolor="#777777", edgecolor="none", alpha=0.32, label="n.s."),
        ],
        loc="lower left", fontsize=5, borderpad=0.4, handlelength=1.0,
        handletextpad=0.4, labelspacing=0.3,
    )
    n_missing = int((~clock["found"]).sum())
    if n_missing:
        ax.text(0.02, 0.99, f"{n_missing} gene(s) not detected",
                transform=ax.transAxes, fontsize=5, color="#666666",
                ha="left", va="top")

    # Reserve room on the right for the role swatch + labels.
    fig.tight_layout(rect=(0, 0, 0.72, 1))
    return fig


def clock_trajectory(
    panel: pd.DataFrame,
    tp_labels: list[str],
    title: str = "Clock gene trajectories",
) -> plt.Figure:
    """One line per clock gene across timepoints, coloured by phase family.

    ``panel`` is a ``clock_gene_panel`` table with ``lfc_<tp>`` columns for
    each label in ``tp_labels`` (in order). Positive-limb genes (warm) and
    repressive-limb/output genes (cool) pulling apart over time is the
    antiphase signature.
    """
    apply_style()
    lfc_cols = [f"lfc_{tp}" for tp in tp_labels]
    present = [c for c in lfc_cols if c in panel.columns]
    if len(present) < 2 or panel.empty:
        fig, ax = plt.subplots(figsize=(3.4, 2))
        ax.text(0.5, 0.5, "Need ≥2 timepoints for a trajectory",
                ha="center", va="center")
        ax.axis("off")
        return fig

    x_pos = list(range(len(lfc_cols)))
    last_x = x_pos[-1]
    mat = panel[lfc_cols].to_numpy(dtype=float)
    finite = mat[np.isfinite(mat)]
    y_lim = max(float(np.abs(finite).max()) * 1.10, 1.0) if finite.size else 1.0

    fig, ax = plt.subplots(figsize=(3.8, 4.4))
    for _, r in panel.iterrows():
        ys = [r[c] for c in lfc_cols]
        if np.all([not np.isfinite(v) for v in ys]):
            continue
        ax.plot(x_pos, ys, color=CLOCK_FAMILY_COLORS.get(r["family"], "#999999"),
                lw=0.9, alpha=0.75, zorder=2)

    ax.axhline(0, color="black", lw=0.3, zorder=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tp_labels)
    ax.set_xlim(-0.15, last_x + 0.95)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_ylabel("log$_2$ fold change")
    ax.set_title(title)

    # Right-edge labels at the final timepoint, spread to avoid overlap.
    end_col = lfc_cols[-1]
    labelled = panel[np.isfinite(panel[end_col].to_numpy(dtype=float))]
    if not labelled.empty:
        y_end = labelled[end_col].to_numpy(dtype=float)
        label_y = _spread_labels(y_end, min_gap=y_lim * 0.06, lo=-y_lim, hi=y_lim)
        label_x = last_x + 0.30
        for ye, yl, name, fam in zip(
            y_end, label_y, labelled["label"], labelled["family"]
        ):
            col = CLOCK_FAMILY_COLORS.get(fam, "#999999")
            ax.plot([last_x], [ye], "o", ms=2.0, color=col,
                    zorder=5, clip_on=False)
            ax.plot([last_x, label_x - 0.03], [ye, yl], color="grey",
                    lw=0.3, zorder=4, clip_on=False)
            ax.text(label_x, yl, name, fontsize=5, va="center", ha="left",
                    color=col, zorder=6)

    # Family legend
    handles = [
        plt.Line2D([], [], color=CLOCK_FAMILY_COLORS["positive"], lw=1.2,
                   label="positive limb"),
        plt.Line2D([], [], color=CLOCK_FAMILY_COLORS["repressive"], lw=1.2,
                   label="repressive / output"),
        plt.Line2D([], [], color=CLOCK_FAMILY_COLORS["accessory"], lw=1.2,
                   label="accessory"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=5,
              bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    return fig


def _phase_theta(zt: np.ndarray) -> np.ndarray:
    """Map Zeitgeber time (0–24 h) to a polar angle for a clock face:
    ZT0 at the top, advancing clockwise (ZT6 right, ZT12 bottom, ZT18 left)."""
    return (np.asarray(zt, dtype=float) / 24.0) * 2.0 * np.pi


def clock_dial(
    clock: pd.DataFrame,
    padj_thresh: float = 0.05,
    title: str = "Clock-face dial",
) -> plt.Figure:
    """Radial clock face: each gene at its reference peak phase (ZT), radius =
    |log2FC|, colour = signed log2FC. Night half (ZT12–24) is lightly shaded.

    Positions come from the literature reference phases in ``CLOCK_GENES`` — a
    perturbation-by-known-phase view, not a measured rhythm.
    """
    apply_style()
    d = clock[clock["found"] & np.isfinite(clock["peak_zt"])].copy()
    if d.empty:
        fig, ax = plt.subplots(figsize=(3.4, 2))
        ax.text(0.5, 0.5, "No phase-annotated clock genes found",
                ha="center", va="center")
        ax.axis("off")
        return fig

    zt = d["peak_zt"].to_numpy(dtype=float)
    lfc = d["lfc"].to_numpy(dtype=float)
    padj = d["padj"].to_numpy(dtype=float)
    sig = np.isfinite(padj) & (padj < padj_thresh)
    theta = _phase_theta(zt)
    r = np.abs(lfc)
    r_max = max(float(np.nanmax(r)) * 1.15, 1.0)
    vmax = max(float(np.nanmax(np.abs(lfc))), 0.5)

    fig = plt.figure(figsize=(4.6, 4.2))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # A donut hole (r=0 sits on an inner ring) so genes with |log2FC|≈0 spread
    # around the ring by phase instead of collapsing onto the centre.
    r_inner = r_max * 0.35
    ax.set_rorigin(-r_inner)

    # Night shading (ZT12–24).
    night = np.linspace(_phase_theta(12), _phase_theta(24), 60)
    ax.fill_between(night, 0, r_max, color="#000000", alpha=0.05, zorder=0)

    sc = ax.scatter(theta[sig], r[sig], c=lfc[sig], cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, s=42, edgecolor="black",
                    linewidths=0.4, zorder=4)
    if (~sig).any():
        ax.scatter(theta[~sig], r[~sig], c=lfc[~sig], cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, s=26, edgecolor="none",
                   alpha=0.4, zorder=3)
    # Label only the genes that actually moved (significant, or |log2FC| ≥ 0.5)
    # so the unchanged inner-ring cluster stays uncluttered.
    lab_mask = sig | (np.abs(lfc) >= 0.5)
    for th, rr, name, show in zip(theta, r, d["label"], lab_mask):
        if show:
            ax.text(th, rr + r_max * 0.07, name, fontsize=4.5,
                    ha="center", va="center", zorder=6)

    ax.set_rlim(0, r_max)
    ax.set_rlabel_position(135)
    ax.set_xticks([_phase_theta(z) for z in (0, 6, 12, 18)])
    ax.set_xticklabels(["ZT0", "ZT6", "ZT12", "ZT18"], fontsize=6)
    ax.tick_params(pad=0.5)
    ax.set_title(title, pad=14)
    ax.grid(color="#dddddd", lw=0.4)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.10)
    cbar.set_label("log$_2$FC", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.4)
    ax.text(0.5, -0.06,
            "radius = |log$_2$FC| · night (ZT12–24) shaded · only moved genes labelled",
            transform=ax.transAxes, fontsize=5, color="#666666",
            ha="center", va="top")
    fig.tight_layout()
    return fig


def clock_phase_scatter(
    clock: pd.DataFrame,
    padj_thresh: float = 0.05,
    title: str = "Regulation by reference phase",
) -> plt.Figure:
    """log2FC vs each gene's reference peak phase (ZT), with a least-squares
    cosine fit summarising phase-dependent regulation. Night (ZT12–24) shaded.
    """
    apply_style()
    d = clock[clock["found"] & np.isfinite(clock["peak_zt"])].copy()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    if d.empty:
        ax.text(0.5, 0.5, "No phase-annotated clock genes found",
                ha="center", va="center")
        ax.axis("off")
        return fig

    zt = d["peak_zt"].to_numpy(dtype=float)
    lfc = d["lfc"].to_numpy(dtype=float)
    padj = d["padj"].to_numpy(dtype=float)
    sig = np.isfinite(padj) & (padj < padj_thresh)

    ax.axvspan(12, 24, color="#000000", alpha=0.05, zorder=0)
    ax.axhline(0, color="black", lw=0.4, zorder=1)

    colors = np.where(lfc > 0, "#D55E00", "#0072B2")
    ax.scatter(zt[sig], lfc[sig], c=colors[sig], s=26, edgecolor="black",
               linewidths=0.4, zorder=4)
    if (~sig).any():
        ax.scatter(zt[~sig], lfc[~sig], c=colors[~sig], s=18, alpha=0.4,
                   edgecolor="none", zorder=3)
    for x, y, name in zip(zt, lfc, d["label"]):
        ax.text(x + 0.25, y, name, fontsize=4.5, va="center", ha="left",
                zorder=5)

    fit = fit_phase_cosine(zt, lfc)
    if np.isfinite(fit["r2"]):
        g = np.linspace(0, 24, 200)
        w = 2 * np.pi / 24.0
        yhat = fit["mesor"] + fit["amplitude"] * np.cos(w * (g - fit["peak_zt"]))
        ax.plot(g, yhat, color="#444444", lw=0.9, ls="--", zorder=2)
        ax.text(0.02, 0.98,
                f"cosine peak ≈ ZT{fit['peak_zt']:.1f}\n"
                f"amp = {fit['amplitude']:.2f}, R² = {fit['r2']:.2f}",
                transform=ax.transAxes, fontsize=5, va="top", ha="left")

    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("reference peak phase (ZT, h)")
    ax.set_ylabel("log$_2$ fold change")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def clock_network(
    clock: pd.DataFrame,
    title: str = "Clock TTFL — module regulation",
) -> plt.Figure:
    """Schematic of the core transcription–translation feedback loop with each
    module coloured by the mean log2FC of its detected genes."""
    apply_style()
    from matplotlib.patches import FancyArrowPatch, Circle

    def module_mean(mask):
        v = clock.loc[mask & clock["found"], "lfc"]
        return float(v.mean()) if len(v) else np.nan

    role = clock["role"].astype(str)
    fam = clock["family"]
    modules = {
        "bmal":   ("BMAL1 : CLOCK\nNPAS2", 0.0, 1.35, role == "activator"),
        "percry": ("PER / CRY", 2.1, 0.1, role == "per_cry"),
        "reverb": ("REV-ERB", 0.75, -1.3, (role == "nr") & (fam == "repressive")),
        "ror":    ("ROR", -2.1, 0.1, (role == "nr") & (fam == "positive")),
        "dec":    ("DEC", -0.75, -1.3, role == "dec"),
        "output": ("DBP / TEF / HLF", 2.1, -1.5, role == "output"),
    }
    vals = {k: module_mean(m) for k, (_, _, _, m) in modules.items()}
    finite = [v for v in vals.values() if np.isfinite(v)]
    vmax = max(max(abs(v) for v in finite), 0.5) if finite else 1.0
    cmap = mpl.colormaps["RdBu_r"]
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    # Edges: (src, dst, kind) — "act" = activation arrow, "rep" = repression.
    edges = [
        ("bmal", "percry", "act"), ("bmal", "reverb", "act"),
        ("bmal", "dec", "act"), ("bmal", "output", "act"),
        ("percry", "bmal", "rep"), ("reverb", "bmal", "rep"),
        ("dec", "bmal", "rep"), ("ror", "bmal", "act"),
    ]
    pos = {k: (x, y) for k, (_, x, y, _) in modules.items()}
    R = 0.42
    for s, t, kind in edges:
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        style = "-|>" if kind == "act" else "-["
        color = "#333333" if kind == "act" else "#B22222"
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), shrinkA=R * 62, shrinkB=R * 62,
            arrowstyle=style, mutation_scale=8, lw=0.8, color=color,
            connectionstyle="arc3,rad=0.12", zorder=1,
        ))

    for k, (name, x, y, _) in modules.items():
        v = vals[k]
        face = cmap(norm(v)) if np.isfinite(v) else "#eeeeee"
        ax.add_patch(Circle((x, y), R, facecolor=face, edgecolor="black",
                            lw=0.6, zorder=3))
        txt = name + (f"\n{v:+.2f}" if np.isfinite(v) else "\nn.d.")
        # dark text on light fill, white on saturated fill
        lum = 0 if not np.isfinite(v) else abs(norm(v) - 0.5) * 2
        tc = "white" if lum > 0.6 else "black"
        ax.text(x, y, txt, ha="center", va="center", fontsize=5,
                color=tc, zorder=4)

    ax.set_xlim(-3.0, 3.2)
    ax.set_ylim(-2.4, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    # Legend for edge types
    ax.plot([], [], color="#333333", lw=0.9, label="activation")
    ax.plot([], [], color="#B22222", lw=0.9, label="repression")
    ax.legend(loc="lower left", fontsize=5, frameon=False)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("module mean log$_2$FC", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.4)
    fig.tight_layout()
    return fig


def clock_phase_heatmap(
    panel: pd.DataFrame,
    tp_labels: list[str],
    title: str = "Clock genes ordered by phase",
) -> plt.Figure:
    """Heatmap of clock-gene log2FC across timepoints, rows ordered by the
    reference peak phase. A cyclic phase swatch on the left shows the ZT order.
    """
    apply_style()
    lfc_cols = [f"lfc_{tp}" for tp in tp_labels if f"lfc_{tp}" in panel.columns]
    d = panel[np.isfinite(panel["peak_zt"])].copy()
    d = d.sort_values("peak_zt")
    if d.empty or not lfc_cols:
        fig, ax = plt.subplots(figsize=(3.4, 2))
        ax.text(0.5, 0.5, "No phase-annotated clock genes", ha="center",
                va="center")
        ax.axis("off")
        return fig

    mat = d[lfc_cols].to_numpy(dtype=float)
    n = len(d)
    finite = mat[np.isfinite(mat)]
    vmax = max(float(np.nanquantile(np.abs(finite), 0.99)), 0.5) if finite.size else 1.0

    fig, ax = plt.subplots(figsize=(0.6 * len(lfc_cols) + 2.2, 0.20 * n + 1.3))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(lfc_cols)))
    ax.set_xticklabels(tp_labels[:len(lfc_cols)])
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{lab}  ·  ZT{zt:.0f}" for lab, zt in zip(d["label"], d["peak_zt"])],
        fontsize=5,
    )
    ax.tick_params(left=False, bottom=False)

    # Cyclic phase swatch to the left of the grid (twilight = circular map).
    phase_cmap = mpl.colormaps["twilight"]
    for i, zt in enumerate(d["peak_zt"].to_numpy(dtype=float)):
        ax.add_patch(plt.Rectangle(
            (-0.62, i - 0.5), 0.22, 1.0, facecolor=phase_cmap((zt % 24) / 24.0),
            edgecolor="none", clip_on=False,
        ))
    ax.set_xlim(-0.7, len(lfc_cols) - 0.5)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        fraction=0.06, pad=0.18, aspect=24, shrink=0.7)
    cbar.set_label("log$_2$FC", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.4)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---- Clock-arrest evidence -------------------------------------------------

_SETTLE_FAMILY_ORDER = ["positive", "repressive", "accessory"]


def clock_settling(
    settling: pd.DataFrame,
    early_tp: str,
    late_tp: str,
    title: str = "Does the displaced state hold?",
) -> plt.Figure:
    """Scatter of log2FC at ``early_tp`` (x) vs ``late_tp`` (y) for clock genes.

    The dashed identity line is "held" (the displacement is maintained). Points
    collapsing toward the x-axis are reverting to baseline; points beyond the
    identity line are still amplifying. Coloured by phase family.
    """
    apply_style()
    d = settling[settling["category"] != "unchanged"].copy()
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    if d.empty:
        ax.text(0.5, 0.5, "No displaced clock genes", ha="center", va="center")
        ax.axis("off")
        return fig

    x = d["lfc_early"].to_numpy(dtype=float)
    y = d["lfc_late"].to_numpy(dtype=float)
    lim = max(float(np.nanmax(np.abs(np.concatenate([x, y])))) * 1.18, 1.0)

    # Reference lines: identity (held), axes (baseline).
    ax.plot([-lim, lim], [-lim, lim], ls="--", lw=0.6, color="#888888",
            zorder=1)
    ax.axhline(0, color="black", lw=0.3, zorder=1)
    ax.axvline(0, color="black", lw=0.3, zorder=1)

    for fam in _SETTLE_FAMILY_ORDER:
        sub = d[d["family"] == fam]
        if sub.empty:
            continue
        ax.scatter(sub["lfc_early"], sub["lfc_late"], s=22,
                   c=CLOCK_FAMILY_COLORS.get(fam, "#999999"),
                   edgecolor="black", linewidths=0.3, zorder=3,
                   label={"positive": "positive limb",
                          "repressive": "repressive / output",
                          "accessory": "accessory"}[fam])
    for xi, yi, name in zip(x, y, d["label"]):
        ax.text(xi + lim * 0.02, yi, name, fontsize=4.5, va="center",
                ha="left", zorder=5)

    # Zone hints
    ax.text(0.97, 0.97, "held →", transform=ax.transAxes, fontsize=5,
            color="#888888", ha="right", va="top", rotation=45)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"log$_2$FC at {early_tp}")
    ax.set_ylabel(f"log$_2$FC at {late_tp}")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=5)
    fig.tight_layout()
    return fig


def torpor_concordance_plot(
    per: pd.DataFrame,
    summary: dict,
    source: str = "",
    title: str = "Torpor clock-signature concordance",
) -> plt.Figure:
    """Our clock-gene log2FC vs a torpor reference direction.

    Horizontal bars = our log2FC per reference gene; a caret marks the expected
    torpor direction; bars are highlighted when our sign matches (concordant).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(4.2, 0.24 * max(len(per), 1) + 1.2))
    if per.empty:
        ax.text(0.5, 0.5, "No genes shared with the torpor reference",
                ha="center", va="center")
        ax.axis("off")
        return fig

    d = per.sort_values(["expected", "lfc"], ascending=[False, False])
    n = len(d)
    y = np.arange(n)[::-1]
    lfc = d["lfc"].to_numpy(dtype=float)
    exp = d["expected"].to_numpy(dtype=int)
    conc = d["concordant"].to_numpy(dtype=bool)
    lim = max(float(np.nanmax(np.abs(lfc))) * 1.25, 1.0)

    for yi, val, ok in zip(y, lfc, conc):
        ax.barh(yi, val, height=0.66,
                color="#009E73" if ok else "#BBBBBB",
                edgecolor="none", zorder=2)
    # Expected torpor direction: caret at the expected side.
    for yi, e in zip(y, exp):
        ax.plot(0.90 * lim * np.sign(e), yi,
                marker=(">" if e > 0 else "<"), ms=5,
                color="#444444", zorder=3)

    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=5.5)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("our log$_2$ fold change")
    ax.tick_params(left=False)

    k, ntot, frac, p = (summary.get("k"), summary.get("n"),
                        summary.get("frac"), summary.get("p"))
    ptxt = f", binomial p = {p:.3g}" if p is not None and np.isfinite(p) else ""
    ax.set_title(f"{title}\n{k}/{ntot} concordant ({frac:.0%}){ptxt}",
                 fontsize=7)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#009E73", label="concordant"),
        Patch(facecolor="#BBBBBB", label="discordant"),
        plt.Line2D([], [], marker=">", color="#444444", lw=0,
                   label="expected torpor direction"),
    ], loc="lower right", fontsize=5, borderpad=0.4)
    # ``source`` provenance is surfaced by the caller (app caption / README)
    # rather than drawn here, to avoid colliding with the x-axis label.
    fig.tight_layout()
    return fig
