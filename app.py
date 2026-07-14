"""Streamlit app: consensus DEG analysis across 20 min, 2 h and 4 h CNO timepoints.

The trajectory set is still the 2 h ∩ 4 h intersection; the 20 min timepoint is
carried as an earlier reference point (see ``add_earlier_reference``) and shown
in its own volcano, overlap, QC, and funnel views.

Run with:
    streamlit run app.py
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
    add_earlier_reference,
    clock_gene_panel,
    clock_gene_table,
    compute_pca,
    consensus_at_timepoint,
    is_tf,
    load_deg,
    load_samples,
    sample_correlation,
    sig_set,
    tf_enrichment,
    to_excel_bytes,
    trajectories,
)
from plots import (
    clock_bars,
    clock_dial,
    clock_network,
    clock_phase_heatmap,
    clock_phase_scatter,
    clock_trajectory,
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
    "Trajectory set = intersection of consensus genes at 2 h and 4 h; "
    "20 min is carried as an earlier reference point."
)

with st.sidebar:
    st.header("Thresholds")
    padj_thresh = st.number_input("padj <", min_value=1e-6, max_value=0.5,
                                  value=0.05, step=0.01, format="%.4f")
    lfc_thresh = st.number_input("|log₂FC| ≥", min_value=0.0, max_value=5.0,
                                 value=0.0, step=0.25)
    volcano_xlim = st.number_input(
        "Volcano |log₂FC| axis limit", min_value=1.0, max_value=30.0,
        value=8.0, step=1.0,
        help="Shared x-axis range for every volcano panel so timepoints are "
             "directly comparable. Genes beyond it are drawn as off-scale "
             "triangles at the boundary.",
    )
    st.header("Files")
    f_20m_g = st.file_uploader("20 min — CRE+CNO vs WT+CNO (genetic)", type=["xls", "tsv", "txt"])
    f_20m_v = st.file_uploader("20 min — CRE+CNO vs CRE+SAL (vehicle)", type=["xls", "tsv", "txt"])
    f_2h_g = st.file_uploader("2 h — CRE+CNO vs WT+CNO (genetic)", type=["xls", "tsv", "txt"])
    f_2h_v = st.file_uploader("2 h — CRE+CNO vs CRE+SAL (vehicle)", type=["xls", "tsv", "txt"])
    f_4h_g = st.file_uploader("4 h — CRE+CNO vs WT+CNO (genetic)", type=["xls", "tsv", "txt"])
    f_4h_v = st.file_uploader("4 h — CRE+CNO vs CRE+SAL (vehicle)", type=["xls", "tsv", "txt"])

if not all([f_20m_g, f_20m_v, f_2h_g, f_2h_v, f_4h_g, f_4h_v]):
    st.info("Upload all six DEG files in the sidebar to start.")
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
                 highlight: tuple[str, ...], title: str,
                 x_lim: float) -> bytes:
    """Render a volcano panel to PDF bytes. Cached on the panel's inputs so
    download buttons don't re-render on every Streamlit rerun.
    """
    import matplotlib.pyplot as plt
    from io import BytesIO
    df = load_deg(BytesIO(file_bytes))
    fig = volcano(df, title, set(highlight), padj, lfc, x_lim=x_lim)
    out = fig_to_bytes(fig, "pdf")
    plt.close(fig)
    return out


with st.spinner("Loading DEG files…"):
    loaded = {
        label: _load(f.getvalue(), f.name)
        for label, f in [
            ("20 min genetic", f_20m_g), ("20 min vehicle", f_20m_v),
            ("2 h genetic", f_2h_g), ("2 h vehicle", f_2h_v),
            ("4 h genetic", f_4h_g), ("4 h vehicle", f_4h_v),
        ]
    }
for label, (_, msgs) in loaded.items():
    for m in msgs:
        st.warning(f"{label}: {m}")
deg_20m_g, deg_20m_v = loaded["20 min genetic"][0], loaded["20 min vehicle"][0]
deg_2h_g, deg_2h_v = loaded["2 h genetic"][0], loaded["2 h vehicle"][0]
deg_4h_g, deg_4h_v = loaded["4 h genetic"][0], loaded["4 h vehicle"][0]

tp_20m = TimepointInputs(genetic=deg_20m_g, vehicle=deg_20m_v)
tp_2h = TimepointInputs(genetic=deg_2h_g, vehicle=deg_2h_v)
tp_4h = TimepointInputs(genetic=deg_4h_g, vehicle=deg_4h_v)

cons_20m = consensus_at_timepoint(tp_20m, padj_thresh, lfc_thresh)
cons_2h = consensus_at_timepoint(tp_2h, padj_thresh, lfc_thresh)
cons_4h = consensus_at_timepoint(tp_4h, padj_thresh, lfc_thresh)
# Trajectory set is unchanged (2 h ∩ 4 h); 20 min is attached as reference only.
traj = trajectories(cons_2h, cons_4h)
traj = add_earlier_reference(traj, tp_20m, label="20m")

# Background pool + TF enrichment computed once so the Tables tab can reuse it.
bg_union = (
    pd.concat([deg_20m_g, deg_20m_v, deg_2h_g, deg_2h_v, deg_4h_g, deg_4h_v],
              ignore_index=True)
    .drop_duplicates("gene_name")
)
tf_enrichment_df = (
    tf_enrichment(traj, bg_union, min_family_size=3) if not traj.empty else pd.DataFrame()
)

funnel_rows = []
for label, df in [
    ("20 min: vs WT",  deg_20m_g),
    ("20 min: vs SAL", deg_20m_v),
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

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Consensus @ 20 min", len(cons_20m))
c2.metric("Consensus @ 2 h", len(cons_2h))
c3.metric("Consensus @ 4 h", len(cons_4h))
c4.metric("Trajectory genes (2h ∩ 4h)", len(traj))
c5.metric("Reversed direction", int((traj["class"] == "reversed").sum()))

(
    tab_overview, tab_volcano, tab_overlap, tab_traj,
    tab_heatmap, tab_tfs, tab_clock, tab_qc, tab_table,
) = st.tabs([
    "Overview", "Volcanoes", "Overlap", "Trajectories",
    "Heatmap", "TFs", "Circadian", "QC", "Tables",
])

with tab_overview:
    st.markdown(
        """
        **Pipeline**
        1. Load six DEG tables (20 min / 2 h / 4 h, each × {genetic, vehicle}).
        2. At each timepoint: keep genes with `padj < threshold` in both contrasts
           and concordant log₂FC sign → **consensus set**.
        3. Intersect 2 h and 4 h consensus sets → **trajectory set** (20 min is
           carried as an earlier reference point, not a membership gate).
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

with tab_volcano:
    panels = [
        ("20 min: CRE+CNO vs WT+CNO",  deg_20m_g, f_20m_g),
        ("20 min: CRE+CNO vs CRE+SAL", deg_20m_v, f_20m_v),
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
                      padj_thresh=padj_thresh, lfc_thresh=lfc_thresh,
                      x_lim=float(volcano_xlim))
        cols[i % 2].pyplot(fig, use_container_width=True)
        plt.close(fig)
        pdf_bytes = _volcano_pdf(
            fobj.getvalue(), fobj.name,
            float(padj_thresh), float(lfc_thresh),
            highlight_key, title, float(volcano_xlim),
        )
        cols[i % 2].download_button(
            f"Download PDF — {title}", pdf_bytes,
            file_name=f"volcano_{title.replace(' ', '_').replace('+', '')}.pdf",
            mime="application/pdf", key=f"vd{i}",
        )

with tab_overlap:
    sets = {
        "20m vs WT":  sig_set(deg_20m_g, padj_thresh),
        "20m vs SAL": sig_set(deg_20m_v, padj_thresh),
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
        ceiling = min(300, len(traj))
        default = min(60, len(traj))
        floor = min(4, ceiling)
        max_n = st.slider("Max genes shown", floor, ceiling, default)
        fig = heatmap(traj, max_genes=max_n)
        st.pyplot(fig, use_container_width=False)
        st.download_button("Heatmap PDF", fig_to_bytes(fig, "pdf"),
                           file_name="heatmap.pdf", mime="application/pdf")

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

with tab_clock:
    st.markdown(
        """
        **Core circadian clock.** The molecular clock is a
        transcription–translation feedback loop: the **positive limb**
        (BMAL1/Arntl · CLOCK · NPAS2) activates E-box transcription of its own
        **repressive limb and outputs** (Per, Cry, Rev-erb, Dec, Dbp/Tef/Hlf),
        which cycle **antiphase** to *Bmal1*. When the loop is driven to one
        extreme, the positive limb goes up while the repressive limb / outputs
        collapse (or vice versa). Effect size = log₂FC of the selected contrast.
        """
    )

    veh_by_tp = {"20 min": deg_20m_v, "2 h": deg_2h_v, "4 h": deg_4h_v}
    gen_by_tp = {"20 min": deg_20m_g, "2 h": deg_2h_g, "4 h": deg_4h_g}

    cc1, cc2 = st.columns(2)
    contrast = cc1.radio(
        "Contrast", ["Vehicle (vs CRE+SAL)", "Genetic (vs WT+CNO)"],
        horizontal=False,
        help="Vehicle-control is the canonical chemogenetic effect size.",
    )
    snap_tp = cc2.radio("Snapshot timepoint", ["20 min", "2 h", "4 h"],
                        index=2, horizontal=True)

    deg_by_tp = veh_by_tp if contrast.startswith("Vehicle") else gen_by_tp
    contrast_short = "CRE+CNO vs CRE+SAL" if contrast.startswith("Vehicle") \
        else "CRE+CNO vs WT+CNO"

    clock_snap = clock_gene_table(deg_by_tp[snap_tp])
    clock_panel = clock_gene_panel(deg_by_tp)

    # Antiphase summary at the selected snapshot.
    found = clock_snap[clock_snap["found"]]
    pos = found.loc[found["family"] == "positive", "lfc"]
    rep = found.loc[found["family"] == "repressive", "lfc"]
    pos_mean = float(pos.mean()) if len(pos) else float("nan")
    rep_mean = float(rep.mean()) if len(rep) else float("nan")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Positive limb mean log₂FC", f"{pos_mean:+.2f}",
              help="Mean over detected activators (Bmal1, Npas2, Clock, ROR…).")
    m2.metric("Repressive/output mean log₂FC", f"{rep_mean:+.2f}",
              help="Mean over detected Per/Cry/Rev-erb/Dec/Dbp/Tef/Hlf.")
    if pos_mean == pos_mean and rep_mean == rep_mean:  # both non-NaN
        m3.metric("Antiphase separation", f"{pos_mean - rep_mean:+.2f}",
                  help="Positive-limb mean minus repressive-limb mean. Large "
                       "positive = classic antiphase (activators up, targets down).")
    m4.metric("Clock genes detected", f"{int(found.shape[0])} / {len(clock_snap)}")

    cL, cR = st.columns([1, 1])
    f_bars = clock_bars(clock_snap, padj_thresh,
                        title=f"Core clock @ {snap_tp}  ({contrast_short})")
    cL.pyplot(f_bars, use_container_width=True)
    cL.download_button(
        "Clock bars PDF", fig_to_bytes(f_bars, "pdf"),
        file_name=f"clock_bars_{snap_tp.replace(' ', '')}.pdf",
        mime="application/pdf", key="clock_bars_pdf",
    )

    f_ctraj = clock_trajectory(
        clock_panel, list(deg_by_tp.keys()),
        title=f"Clock trajectories ({contrast_short})",
    )
    cR.pyplot(f_ctraj, use_container_width=True)
    cR.download_button(
        "Clock trajectory PDF", fig_to_bytes(f_ctraj, "pdf"),
        file_name="clock_trajectory.pdf",
        mime="application/pdf", key="clock_traj_pdf",
    )

    st.divider()
    st.markdown(
        "##### Phase-based views\n"
        "Genes are positioned by their **reference peak phase** (ZT) — "
        "approximate mouse peripheral-clock acrophases from the literature "
        "(`CLOCK_GENES` in `analysis.py`), **not** a rhythm measured here. "
        "These views ask whether the perturbation is organised by a gene's "
        "normal peak time."
    )

    pc1, pc2 = st.columns([1, 1])
    f_dial = clock_dial(clock_snap, padj_thresh,
                        title=f"Clock-face @ {snap_tp}  ({contrast_short})")
    pc1.pyplot(f_dial, use_container_width=True)
    pc1.download_button(
        "Clock-face PDF", fig_to_bytes(f_dial, "pdf"),
        file_name=f"clock_dial_{snap_tp.replace(' ', '')}.pdf",
        mime="application/pdf", key="clock_dial_pdf",
    )

    f_psc = clock_phase_scatter(
        clock_snap, padj_thresh,
        title=f"Regulation by phase @ {snap_tp}  ({contrast_short})")
    pc2.pyplot(f_psc, use_container_width=True)
    pc2.download_button(
        "Phase-vs-log₂FC PDF", fig_to_bytes(f_psc, "pdf"),
        file_name=f"clock_phase_scatter_{snap_tp.replace(' ', '')}.pdf",
        mime="application/pdf", key="clock_psc_pdf",
    )

    pc3, pc4 = st.columns([1, 1])
    f_net = clock_network(clock_snap,
                          title=f"Clock TTFL @ {snap_tp}  ({contrast_short})")
    pc3.pyplot(f_net, use_container_width=True)
    pc3.download_button(
        "TTFL network PDF", fig_to_bytes(f_net, "pdf"),
        file_name=f"clock_network_{snap_tp.replace(' ', '')}.pdf",
        mime="application/pdf", key="clock_net_pdf",
    )

    f_pheat = clock_phase_heatmap(
        clock_panel, list(deg_by_tp.keys()),
        title=f"Clock genes by phase ({contrast_short})")
    pc4.pyplot(f_pheat, use_container_width=True)
    pc4.download_button(
        "Phase heatmap PDF", fig_to_bytes(f_pheat, "pdf"),
        file_name="clock_phase_heatmap.pdf",
        mime="application/pdf", key="clock_pheat_pdf",
    )

    st.subheader(f"Clock gene table ({contrast_short})")
    disp = clock_panel.drop(columns=["role"]).rename(
        columns={"role_label": "role"})
    st.dataframe(disp.round(3), use_container_width=True, height=360,
                 hide_index=True)
    st.download_button(
        "Download clock table (.xlsx)",
        to_excel_bytes({"clock_genes": clock_panel.drop(columns=["role"])}),
        file_name="clock_genes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="clock_xlsx",
    )

with tab_qc:
    qc_files = [
        ("20 min: CRE+CNO vs WT+CNO",  f_20m_g, "CRE+CNO_20m", "WT+CNO_20m"),
        ("20 min: CRE+CNO vs CRE+SAL", f_20m_v, "CRE+CNO_20m", "CRE+SAL_20m"),
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
    st.subheader("Consensus @ 20 min")
    st.dataframe(cons_20m, use_container_width=True, height=240)
    st.subheader("Consensus @ 2 h")
    st.dataframe(cons_2h, use_container_width=True, height=240)
    st.subheader("Consensus @ 4 h")
    st.dataframe(cons_4h, use_container_width=True, height=240)
    st.subheader("Trajectory set (consensus at 2 h and 4 h; 20 min as reference)")
    st.dataframe(traj, use_container_width=True, height=320)

    sheets = {
        "consensus_20m": cons_20m,
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
    "20 min genetic": f_20m_g, "20 min vehicle": f_20m_v,
    "2 h genetic": f_2h_g, "2 h vehicle": f_2h_v,
    "4 h genetic": f_4h_g, "4 h vehicle": f_4h_v,
}
load_warnings = {label: msgs for label, (_, msgs) in loaded.items()}
debug_text = build_log(
    thresholds={"padj <": padj_thresh, "|log2FC| >=": lfc_thresh},
    files=file_inputs,
    load_warnings=load_warnings,
    counts={
        "consensus_20m": len(cons_20m),
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
