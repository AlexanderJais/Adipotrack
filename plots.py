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


def fig_to_bytes(fig: plt.Figure, fmt: str = "pdf") -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
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
    texts = [
        ax.text(
            float(np.clip(r["log2FoldChange"], -x_lim, x_lim)),
            float(min(r["neglog10padj"], y_lim)),
            r["gene_name"],
            fontsize=5,
        )
        for _, r in top.iterrows()
    ]
    if texts:
        try:
            from adjustText import adjust_text
            adjust_text(
                texts, ax=ax,
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
    ax.set_ylabel(r"$-\log_{10}$ adjusted $P$")
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
            rho_text = rf"Spearman $\rho$ = {rho:.2f}"
        except ImportError:
            r = float(np.corrcoef(traj["lfc_2h"], traj["lfc_4h"])[0, 1])
            rho_text = rf"Pearson $r$ = {r:.2f}"
        ax.text(0.03, 0.97, rho_text, transform=ax.transAxes,
                ha="left", va="top", fontsize=6)

    ax.set_xlabel("log$_2$FC at 2 h (CRE+CNO vs CRE+SAL)")
    ax.set_ylabel("log$_2$FC at 4 h (CRE+CNO vs CRE+SAL)")
    ax.set_title(f"Trajectory genes (n = {len(traj)})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def trajectory_lines(traj: pd.DataFrame, label_top_n: int = 6) -> plt.Figure:
    """One line per gene from 2h to 4h, faceted by class."""
    apply_style()
    classes = [c for c in CLASS_ORDER if (traj["class"] == c).any()]
    n = len(classes)
    if n == 0:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.text(0.5, 0.5, "No consensus genes", ha="center", va="center")
        ax.axis("off")
        return fig

    # Shared symmetric y-range for honest cross-class comparison
    arr = np.concatenate([traj["lfc_2h"].to_numpy(), traj["lfc_4h"].to_numpy()])
    y_lim = max(float(np.nanquantile(np.abs(arr), 0.99)) * 1.15, 1.0) if arr.size else 1.0

    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.8), sharey=True)
    if n == 1:
        axes = [axes]

    try:
        from adjustText import adjust_text
        have_adjust = True
    except ImportError:
        have_adjust = False

    for ax, cls in zip(axes, classes):
        sub = traj[traj["class"] == cls]
        # Per-gene lines with median trajectory drawn on top
        for _, r in sub.iterrows():
            ax.plot([0, 1], [r["lfc_2h"], r["lfc_4h"]],
                    color=CLASS_COLORS[cls], lw=0.4, alpha=0.4)
        if len(sub) >= 3:
            med = [sub["lfc_2h"].median(), sub["lfc_4h"].median()]
            ax.plot([0, 1], med, color="black", lw=1.2, marker="o",
                    markersize=3, markerfacecolor="white", markeredgewidth=0.6)
        ax.axhline(0, color="black", lw=0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["2 h", "4 h"])
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_title(f"{cls.replace('_', ' ')}\n(n = {len(sub)})")

        # Label the most-extreme genes with adjustText repulsion
        if not sub.empty:
            extreme = sub.iloc[
                sub["lfc_4h"].abs().to_numpy().argsort()[::-1]
            ].head(label_top_n)
            texts = [
                ax.text(1.0, r["lfc_4h"], r["gene_name"], fontsize=5, va="center")
                for _, r in extreme.iterrows()
            ]
            if have_adjust and texts:
                adjust_text(
                    texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.3),
                    only_move={"points": "y", "texts": "y"},
                )

    axes[0].set_ylabel("log$_2$ fold change (CRE+CNO vs CRE+SAL)")
    fig.tight_layout()
    return fig


def venn4(sets: dict[str, set[str]], max_rows: int = 20) -> plt.Figure:
    """Approximate N-way overlap via UpSet-style bar chart (Venn4 is unreadable).

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
    """Heatmap of LFC across the 4 comparisons for top consensus genes."""
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
        d = (
            d.groupby("class", group_keys=False)
            .apply(lambda g: g.reindex(g["lfc_4h"].abs().sort_values(ascending=False).index)
                   .head(per_class))
        )

    mat = d[[
        "lfc_genetic_2h", "lfc_vehicle_2h",
        "lfc_genetic_4h", "lfc_vehicle_4h",
    ]].to_numpy()
    col_labels = ["2h vs WT", "2h vs SAL", "4h vs WT", "4h vs SAL"]

    # Symmetric, robust colour scale (99th percentile) so a single huge LFC
    # doesn't desaturate everything else.
    if mat.size:
        vmax = float(np.nanquantile(np.abs(mat), 0.99))
        vmax = max(vmax, 0.5)
    else:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(2.8, 0.13 * len(d) + 1.2))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")

    # Class separator lines + right-side class label bar
    classes_in_order = list(d["class"].astype(str))
    boundaries = [i for i in range(1, len(classes_in_order))
                  if classes_in_order[i] != classes_in_order[i - 1]]
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", lw=1.0)
        ax.axhline(b - 0.5, color="black", lw=0.4)

    # Vertical separator between 2h and 4h block
    ax.axvline(1.5, color="white", lw=1.0)
    ax.axvline(1.5, color="black", lw=0.4)

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

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.18)
    cbar.set_label("log$_2$FC", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.4)

    fig.tight_layout()
    return fig
