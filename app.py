"""Streamlit app: consensus DEG analysis across 2 h and 4 h CNO timepoints.

Upload the four DEG tables (genetic and vehicle controls at each
timepoint) via the sidebar; the app then produces consensus and
trajectory tables, eight tabs of figures, an XLSX bundle of every
table, and a downloadable plaintext debug log for bug reports.

Run with:

    streamlit run app.py

All heavy work (DEG parsing, sample-level QC, volcano PDF rendering)
goes through ``@st.cache_data`` so reruns triggered by widget changes
don't re-parse the upload bytes.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from debug_log import build_log
from analysis import (
    CLASS_ORDER,
    TimepointInputs,
    compute_pca,
    consensus_at_timepoint,
    is_tf,
    load_deg,
    load_samples,
    lookup_gene,
    sample_correlation,
    sig_set,
    tf_enrichment,
    to_excel_bytes,
    trajectories,
)
from plots import (
    fig_to_bytes,
    heatmap,
    lfc_scatter,
    pca_scatter,
    sample_corr_heatmap,
    tf_enrichment_dot,
    tf_lfc_panel,
    trajectory_lines,
    upset_plot,
    volcano,
)

st.set_page_config(page_title="Consensus DEG explorer", layout="wide")
st.title("Consensus DEG explorer — CRE+CNO chemogenetic activation")
st.caption(
    "Strict consensus = significant in BOTH genetic-control (vs WT+CNO) and "
    "vehicle-control (vs CRE+SAL) AND concordant log₂FC sign. "
    "Trajectory set = intersection of consensus genes at 2 h and 4 h."
)

with st.sidebar:
    st.header("Thresholds")
    padj_thresh = st.number_input("padj <", min_value=1e-6, max_value=0.5,
                                  value=0.05, step=0.01, format="%.4f")
    lfc_thresh = st.number_input("|log₂FC| ≥", min_value=0.0, max_value=5.0,
                                 value=0.0, step=0.25)
    st.header("Files")
    f_2h_g = st.file_uploader("2 h — CRE+CNO vs WT+CNO (genetic)", type=["xls", "tsv", "txt"])
    f_2h_v = st.file_uploader("2 h — CRE+CNO vs CRE+SAL (vehicle)", type=["xls", "tsv", "txt"])
    f_4h_g = st.file_uploader("4 h — CRE+CNO vs WT+CNO (genetic)", type=["xls", "tsv", "txt"])
    f_4h_v = st.file_uploader("4 h — CRE+CNO vs CRE+SAL (vehicle)", type=["xls", "tsv", "txt"])

if not all([f_2h_g, f_2h_v, f_4h_g, f_4h_v]):
    st.info("Upload all four DEG files in the sidebar to start.")
    st.stop()


@st.cache_data(show_spinner=False)
def _load(file_bytes: bytes, _cache_key: str) -> "tuple":
    """Load a DEG table; returns (df, list_of_warning_messages).

    The `_cache_key` argument differentiates uploads with identical bytes-but-
    different filenames; it isn't used inside the function.
    """
    from io import BytesIO
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        df = load_deg(BytesIO(file_bytes))
    return df, [str(w.message) for w in captured]


@st.cache_data(show_spinner=False)
def _qc_for_file(file_bytes: bytes, _cache_key: str,
                 a_label: str, b_label: str):
    """Load samples + run PCA + correlation in one cached call.

    Returns (metadata, pca_scores, var_explained, corr_df). Heavy outputs
    (the full expr matrix) are dropped before returning to keep the cache
    entry small.
    """
    from io import BytesIO
    sd = load_samples(BytesIO(file_bytes), a_label, b_label)
    scores, var_exp = compute_pca(sd.expr)
    corr = sample_correlation(sd.expr)
    return sd.metadata, scores, var_exp, corr


@st.cache_data(show_spinner=False, max_entries=32)
def _volcano_pdf(file_bytes: bytes, _cache_key: str,
                 padj: float, lfc: float,
                 highlight: tuple[str, ...], title: str) -> bytes:
    """Render a volcano panel to PDF bytes. Cached on the panel's inputs so
    download buttons don't re-render on every Streamlit rerun.
    """
    import matplotlib.pyplot as plt
    from io import BytesIO
    df = load_deg(BytesIO(file_bytes))
    fig = volcano(df, title, set(highlight), padj, lfc)
    out = fig_to_bytes(fig, "pdf")
    plt.close(fig)
    return out


with st.spinner("Loading DEG files…"):
    loaded = {
        label: _load(f.getvalue(), f.name)
        for label, f in [
            ("2 h genetic", f_2h_g), ("2 h vehicle", f_2h_v),
            ("4 h genetic", f_4h_g), ("4 h vehicle", f_4h_v),
        ]
    }
for label, (_, msgs) in loaded.items():
    for m in msgs:
        st.warning(f"{label}: {m}")
deg_2h_g, deg_2h_v = loaded["2 h genetic"][0], loaded["2 h vehicle"][0]
deg_4h_g, deg_4h_v = loaded["4 h genetic"][0], loaded["4 h vehicle"][0]

tp_2h = TimepointInputs(genetic=deg_2h_g, vehicle=deg_2h_v)
tp_4h = TimepointInputs(genetic=deg_4h_g, vehicle=deg_4h_v)

cons_2h = consensus_at_timepoint(tp_2h, padj_thresh, lfc_thresh)
cons_4h = consensus_at_timepoint(tp_4h, padj_thresh, lfc_thresh)
traj = trajectories(cons_2h, cons_4h)

# Background pool + TF enrichment computed once so the Tables tab can reuse it.
bg_union = (
    pd.concat([deg_2h_g, deg_2h_v, deg_4h_g, deg_4h_v], ignore_index=True)
    .drop_duplicates("gene_name")
)
tf_enrichment_df = (
    tf_enrichment(traj, bg_union, min_family_size=3) if not traj.empty else pd.DataFrame()
)

funnel_rows = []
for label, df in [
    ("2 h: vs WT",  deg_2h_g),
    ("2 h: vs SAL", deg_2h_v),
    ("4 h: vs WT",  deg_4h_g),
    ("4 h: vs SAL", deg_4h_v),
]:
    sig_mask = (df["padj"] < padj_thresh) & (df["log2FoldChange"].abs() >= lfc_thresh)
    sig = df[sig_mask]
    funnel_rows.append({
        "comparison": label,
        "tested": len(df),
        "significant": int(sig_mask.sum()),
        "up": int((sig["log2FoldChange"] > 0).sum()),
        "down": int((sig["log2FoldChange"] < 0).sum()),
    })
funnel = pd.DataFrame(funnel_rows)
class_counts_series = (
    traj["class"].value_counts().reindex(CLASS_ORDER, fill_value=0)
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Consensus @ 2 h", len(cons_2h))
c2.metric("Consensus @ 4 h", len(cons_4h))
c3.metric("Trajectory genes (2h ∩ 4h)", len(traj))
c4.metric("Reversed direction", int((traj["class"] == "reversed").sum()))

(
    tab_overview, tab_lookup, tab_volcano, tab_overlap, tab_traj,
    tab_heatmap, tab_tfs, tab_qc, tab_table,
) = st.tabs([
    "Overview", "Gene lookup", "Volcanoes", "Overlap", "Trajectories",
    "Heatmap", "TFs", "QC", "Tables",
])

with tab_overview:
    st.markdown(
        """
        **Pipeline**
        1. Load four DEG tables (2 h × {genetic, vehicle}, 4 h × {genetic, vehicle}).
        2. At each timepoint: keep genes with `padj < threshold` in both contrasts
           and concordant log₂FC sign → **consensus set**.
        3. Intersect 2 h and 4 h consensus sets → **trajectory set**.
        4. Classify each trajectory gene by direction at each timepoint:
           *sustained up / transient up / sustained down / transient down / reversed*.
        5. Effect size for plots = log₂FC from CRE+CNO vs CRE+SAL (direct chemogenetic).
        """
    )
    c1, c2 = st.columns([1.4, 1])
    c1.subheader("Per-comparison significance funnel")
    c1.dataframe(funnel, use_container_width=True, hide_index=True)
    c2.subheader("Trajectory class counts")
    c2.dataframe(class_counts_series.rename("genes").to_frame(),
                 use_container_width=False)

with tab_lookup:
    query = st.text_input(
        "Gene symbol",
        value="",
        placeholder="e.g. Fos, Jun, Atf3",
        help="Case-insensitive search across all four uploaded DEG tables. "
             "Significance flags use the current sidebar thresholds.",
    )
    if not query.strip():
        st.caption(
            "Type a gene symbol to see its values across the four contrasts "
            "and its membership in the consensus / trajectory sets."
        )
    else:
        lookup_contrasts = [
            ("2 h: vs WT",  deg_2h_g),
            ("2 h: vs SAL", deg_2h_v),
            ("4 h: vs WT",  deg_4h_g),
            ("4 h: vs SAL", deg_4h_v),
        ]
        result = lookup_gene(
            query, lookup_contrasts, cons_2h, cons_4h, traj,
            padj_thresh=padj_thresh, lfc_thresh=lfc_thresh,
        )
        if result.name is None:
            st.warning(f"`{result.query}` not found in any of the four DEG tables.")
            if result.suggestions:
                st.caption(
                    "Did you mean: "
                    + ", ".join(f"`{s}`" for s in result.suggestions)
                    + "?"
                )
        else:
            st.subheader(result.name)

            # Compact metadata strip + full description below.
            meta_bits = []
            a = result.annotations
            if "gene_id" in a:
                meta_bits.append(f"`{a['gene_id']}`")
            if "gene_biotype" in a:
                meta_bits.append(str(a["gene_biotype"]))
            tf = a.get("tf_family")
            if tf and str(tf).strip() != "-":
                meta_bits.append(f"TF family: **{tf}**")
            if all(k in a for k in ("gene_chr", "gene_start", "gene_end", "gene_strand")):
                meta_bits.append(
                    f"chr{a['gene_chr']}:"
                    f"{int(a['gene_start']):,}–{int(a['gene_end']):,} "
                    f"({a['gene_strand']})"
                )
            if meta_bits:
                st.caption(" · ".join(meta_bits))
            if "gene_description" in a:
                st.caption(str(a["gene_description"]))

            c1, c2 = st.columns([2, 1])
            c1.markdown("**Per-contrast values** (current thresholds applied)")
            display = result.per_contrast.copy()
            display["significant"] = display["significant"].map(
                {True: "✓", False: "·"}
            )
            c1.dataframe(
                display, use_container_width=True, hide_index=True,
                column_config={
                    "log2FC": st.column_config.NumberColumn(format="%+.3f"),
                    "padj": st.column_config.NumberColumn(format="%.2e"),
                },
            )

            c2.markdown("**Set membership**")
            c2.markdown(
                f"- {'✓' if result.in_consensus_2h else '·'} Consensus @ 2 h\n"
                f"- {'✓' if result.in_consensus_4h else '·'} Consensus @ 4 h\n"
                f"- {'✓' if result.in_trajectory else '·'} Trajectory set"
            )
            if result.trajectory_class:
                c2.markdown(
                    f"**Class:** `{result.trajectory_class.replace('_', ' ')}`"
                )
            if result.delta_lfc is not None:
                c2.markdown(f"**Δ lfc (4h − 2h):** {result.delta_lfc:+.3f}")

with tab_volcano:
    panels = [
        ("2 h: CRE+CNO vs WT+CNO",  deg_2h_g, f_2h_g),
        ("2 h: CRE+CNO vs CRE+SAL", deg_2h_v, f_2h_v),
        ("4 h: CRE+CNO vs WT+CNO",  deg_4h_g, f_4h_g),
        ("4 h: CRE+CNO vs CRE+SAL", deg_4h_v, f_4h_v),
    ]
    highlight_set = set(traj["gene_name"])
    highlight_key = tuple(sorted(highlight_set))  # hashable for cache_data
    cols = st.columns(2)
    for i, (title, df, fobj) in enumerate(panels):
        fig = volcano(df, title, highlight=highlight_set,
                      padj_thresh=padj_thresh, lfc_thresh=lfc_thresh)
        cols[i % 2].pyplot(fig, use_container_width=True)
        plt.close(fig)
        pdf_bytes = _volcano_pdf(
            fobj.getvalue(), fobj.name,
            float(padj_thresh), float(lfc_thresh),
            highlight_key, title,
        )
        cols[i % 2].download_button(
            f"Download PDF — {title}", pdf_bytes,
            file_name=f"volcano_{title.replace(' ', '_').replace('+', '')}.pdf",
            mime="application/pdf", key=f"vd{i}",
        )

with tab_overlap:
    sets = {
        "2h vs WT":  sig_set(deg_2h_g, padj_thresh),
        "2h vs SAL": sig_set(deg_2h_v, padj_thresh),
        "4h vs WT":  sig_set(deg_4h_g, padj_thresh),
        "4h vs SAL": sig_set(deg_4h_v, padj_thresh),
    }
    fig = upset_plot(sets)
    st.pyplot(fig, use_container_width=False)
    st.download_button("Download UpSet PDF", fig_to_bytes(fig, "pdf"),
                       file_name="upset.pdf", mime="application/pdf")

with tab_traj:
    if traj.empty:
        st.warning("No genes are consensus at both timepoints with current thresholds.")
    else:
        c1, c2 = st.columns([1, 1])
        f1 = lfc_scatter(traj)
        c1.pyplot(f1, use_container_width=True)
        c1.download_button("Scatter PDF", fig_to_bytes(f1, "pdf"),
                           file_name="lfc_scatter.pdf", mime="application/pdf")
        f2 = trajectory_lines(traj)
        c2.pyplot(f2, use_container_width=True)
        c2.download_button("Trajectories PDF", fig_to_bytes(f2, "pdf"),
                           file_name="trajectories.pdf", mime="application/pdf")

with tab_heatmap:
    if traj.empty:
        st.warning("No genes to plot.")
    elif len(traj) < 4:
        st.info(f"Only {len(traj)} consensus gene(s) — showing all.")
        fig = heatmap(traj, max_genes=len(traj))
        st.pyplot(fig, use_container_width=False)
        st.download_button("Heatmap PDF", fig_to_bytes(fig, "pdf"),
                           file_name="heatmap.pdf", mime="application/pdf")
    else:
        c1, c2 = st.columns([3, 1])
        ceiling = min(300, len(traj))
        default = min(60, len(traj))
        floor = min(4, ceiling)
        max_n = c1.slider("Max genes shown", floor, ceiling, default)
        cluster = c2.checkbox(
            "Hierarchical clustering",
            help="Reorder rows by similarity across the four contrasts "
                 "and draw a row dendrogram on the left. Class membership "
                 "is shown as a per-row colour band on the right.",
        )
        fig = heatmap(traj, max_genes=max_n, cluster=cluster)
        st.pyplot(fig, use_container_width=False)
        st.download_button(
            "Heatmap PDF", fig_to_bytes(fig, "pdf"),
            file_name=("heatmap_clustered.pdf" if cluster else "heatmap.pdf"),
            mime="application/pdf",
        )

with tab_tfs:
    if traj.empty:
        st.warning("No consensus genes — TF panel needs trajectory genes.")
    else:
        n_tf_traj = int(is_tf(traj).sum())
        n_tf_bg = int(is_tf(bg_union).sum())
        st.caption(
            f"{n_tf_traj} TF(s) in the trajectory set out of {len(traj)} genes "
            f"({n_tf_traj / max(len(traj), 1):.1%}); "
            f"background pool: {n_tf_bg} TFs in {len(bg_union)} genes "
            f"({n_tf_bg / max(len(bg_union), 1):.1%})."
        )

        c1, c2 = st.columns([1.1, 1])
        f_tf = tf_lfc_panel(traj)
        c1.pyplot(f_tf, use_container_width=True)
        c1.download_button("TF panel PDF", fig_to_bytes(f_tf, "pdf"),
                           file_name="tf_panel.pdf",
                           mime="application/pdf")

        if tf_enrichment_df.empty:
            c2.info("No TF families pass the size filter (≥3 in foreground).")
        else:
            f_enr = tf_enrichment_dot(tf_enrichment_df)
            c2.pyplot(f_enr, use_container_width=True)
            c2.download_button("TF enrichment PDF", fig_to_bytes(f_enr, "pdf"),
                               file_name="tf_enrichment.pdf",
                               mime="application/pdf")
            c2.dataframe(tf_enrichment_df.round(4),
                         use_container_width=True, height=240)

with tab_qc:
    qc_files = [
        ("2 h: CRE+CNO vs WT+CNO",  f_2h_g, "CRE+CNO_2h", "WT+CNO_2h"),
        ("2 h: CRE+CNO vs CRE+SAL", f_2h_v, "CRE+CNO_2h", "CRE+SAL_2h"),
        ("4 h: CRE+CNO vs WT+CNO",  f_4h_g, "CRE+CNO_4h", "WT+CNO_4h"),
        ("4 h: CRE+CNO vs CRE+SAL", f_4h_v, "CRE+CNO_4h", "CRE+SAL_4h"),
    ]
    qc_subtabs = st.tabs([t[0] for t in qc_files])
    for sub, (label, fobj, a_lbl, b_lbl) in zip(qc_subtabs, qc_files):
        with sub:
            try:
                metadata, scores, var_exp, corr = _qc_for_file(
                    fobj.getvalue(), fobj.name, a_lbl, b_lbl,
                )
            except Exception as e:
                st.error(f"Sample loading failed for {label}: {e}")
                continue

            c1, c2 = st.columns(2)
            f_pca = pca_scatter(scores, var_exp, metadata,
                                title=f"PCA · {label}")
            c1.pyplot(f_pca, use_container_width=True)
            c1.download_button(
                "PCA PDF", fig_to_bytes(f_pca, "pdf"),
                file_name=f"pca_{label.replace(' ', '_').replace(':', '')}.pdf",
                mime="application/pdf",
                key=f"pca_{label}",
            )
            f_corr = sample_corr_heatmap(corr, metadata,
                                         title=f"Sample correlation · {label}")
            c2.pyplot(f_corr, use_container_width=True)
            c2.download_button(
                "Correlation PDF", fig_to_bytes(f_corr, "pdf"),
                file_name=f"corr_{label.replace(' ', '_').replace(':', '')}.pdf",
                mime="application/pdf",
                key=f"corr_{label}",
            )

with tab_table:
    st.subheader("Consensus @ 2 h")
    st.dataframe(cons_2h, use_container_width=True, height=240)
    st.subheader("Consensus @ 4 h")
    st.dataframe(cons_4h, use_container_width=True, height=240)
    st.subheader("Trajectory set (consensus at both timepoints)")
    st.dataframe(traj, use_container_width=True, height=320)

    sheets = {
        "consensus_2h": cons_2h,
        "consensus_4h": cons_4h,
        "trajectories": traj,
    }
    if not tf_enrichment_df.empty:
        sheets["tf_enrichment"] = tf_enrichment_df
    st.download_button(
        "Download all tables (.xlsx)",
        to_excel_bytes(sheets),
        file_name="consensus_degs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

file_inputs = {
    "2 h genetic": f_2h_g, "2 h vehicle": f_2h_v,
    "4 h genetic": f_4h_g, "4 h vehicle": f_4h_v,
}
load_warnings = {label: msgs for label, (_, msgs) in loaded.items()}
debug_text = build_log(
    thresholds={"padj <": padj_thresh, "|log2FC| >=": lfc_thresh},
    files=file_inputs,
    load_warnings=load_warnings,
    counts={
        "consensus_2h": len(cons_2h),
        "consensus_4h": len(cons_4h),
        "trajectory": len(traj),
        "reversed": int((traj["class"] == "reversed").sum()),
        "bg_union": len(bg_union),
        "tf_enrichment_rows": len(tf_enrichment_df),
    },
    funnel=funnel,
    class_counts=class_counts_series,
    tf_enrichment=tf_enrichment_df,
)
st.sidebar.divider()
st.sidebar.download_button(
    "Download debug log",
    debug_text.encode("utf-8"),
    file_name="adipotrack_debug.log",
    mime="text/plain",
    help="Plaintext snapshot of inputs, settings, and pipeline counts. "
         "Attach this to bug reports.",
)
