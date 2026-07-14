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

from analysis import CLASS_COLORS, CLASS_ORDER


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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
) -> plt.Figure:
    apply_style()
    d = df.copy()
    d["neglog10padj"] = -np.log10(d["padj"].clip(lower=1e-300))
    sig = (d["padj"] < padj_thresh) & (d["log2FoldChange"].abs() >= lfc_thresh)

    # Axis limits driven by *significant* genes so non-sig outliers don't
    # collapse the visible range. Non-sig points outside the window are
    # rendered as off-scale triangles at the boundary.
    if sig.any():
        x_ref = d.loc[sig, "log2FoldChange"].abs().max()
        y_ref = d.loc[sig, "neglog10padj"].max()
    else:
        x_ref = d["log2FoldChange"].abs().quantile(0.99)
        y_ref = d["neglog10padj"].quantile(0.99)
    x_lim = max(float(x_ref) * 1.2, 1.0)
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
    ax.set_xlabel("log₂ fold change")
    ax.set_ylabel("−log₁₀ adjusted P")
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

    ax.set_xlabel("log₂FC at 2 h (CRE+CNO vs CRE+SAL)")
    ax.set_ylabel("log₂FC at 4 h (CRE+CNO vs CRE+SAL)")
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
    """One line per gene from 2 h to 4 h, faceted by class.

    Each panel reserves a right-hand label gutter (x = 1.0 → 1.6) so labels
    can be repelled outward without colliding with the trajectories.
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

    # Shared symmetric y-range so panels can be compared at a glance.
    arr = np.concatenate([traj[c].to_numpy(dtype=float) for c in x_cols])
    if arr.size:
        y_lim = max(float(np.nanquantile(np.abs(arr), 0.99)) * 1.18, 1.0)
    else:
        y_lim = 1.0

    fig, axes = plt.subplots(
        1, n, figsize=(panel_width * n, panel_height), sharey=True,
    )
    if n == 1:
        axes = [axes]

    try:
        from adjustText import adjust_text
        have_adjust = True
    except ImportError:
        have_adjust = False

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
        ax.set_xlim(-0.15, last_x + 0.65)  # extra room on the right for labels
        ax.set_ylim(-y_lim, y_lim)
        ax.set_title(f"{cls.replace('_', ' ')}\n(n = {len(sub)})")

        # Right-edge label gutter
        if not sub.empty:
            extreme = sub.iloc[
                sub["lfc_4h"].abs().to_numpy().argsort()[::-1]
            ].head(label_top_n)

            anchor_x = np.full(len(extreme), float(last_x))
            anchor_y = extreme["lfc_4h"].to_numpy(dtype=float)

            texts = [
                ax.text(
                    last_x + 0.18, y, name,
                    fontsize=5.5, va="center", ha="left", zorder=6,
                )
                for y, name in zip(anchor_y, extreme["gene_name"])
            ]

            if have_adjust and texts:
                # adjustText API differs across versions — try kwargs that
                # work on 0.8 / 1.x and silently fall back if something
                # in the call signature changed.
                try:
                    adjust_text(
                        texts,
                        x=anchor_x.tolist(),
                        y=anchor_y.tolist(),
                        ax=ax,
                        arrowprops=dict(
                            arrowstyle="-", color="grey",
                            lw=0.3, shrinkA=0, shrinkB=2,
                        ),
                        only_move={"text": "y", "static": "y", "explode": "y"},
                        expand=(1.1, 1.4),
                        force_text=(0.0, 0.6),
                        autoalign=False,
                        avoid_self=True,
                    )
                except TypeError:
                    adjust_text(
                        texts,
                        x=anchor_x.tolist(),
                        y=anchor_y.tolist(),
                        ax=ax,
                        arrowprops=dict(
                            arrowstyle="-", color="grey", lw=0.3,
                        ),
                        only_move={"points": "", "texts": "y"},
                    )
            else:
                # Plain fallback: nudge labels apart by sorting on y and
                # spacing them at a minimum vertical separation.
                _stack_labels(texts, ax, min_gap=y_lim * 0.06)

    axes[0].set_ylabel("log₂ fold change (CRE+CNO vs CRE+SAL)")
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
        ax.set_xlabel("log₂FC at 4 h")
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
    ax.set_xlabel("log₂ odds ratio (foreground vs background)")
    ax.set_title(f"TF-family enrichment (top {len(d)})")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("−log₁₀ q", fontsize=6)
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


def _stack_labels(texts, ax, min_gap: float) -> None:
    """Greedy vertical-stacking fallback if adjustText isn't available."""
    items = sorted(
        ((t, t.get_position()[1]) for t in texts),
        key=lambda p: p[1],
    )
    last_y = -np.inf
    for t, y in items:
        new_y = max(y, last_y + min_gap)
        x = t.get_position()[0]
        t.set_position((x, new_y))
        last_y = new_y


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
    cbar.set_label("log₂FC", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.4)

    # Reserve space on the right for the class swatch + labels (~25% of axes
    # width is plenty for short class names at fontsize 5).
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    return fig
