# Consensus DEG explorer

Streamlit app and analysis library for finding genes that are robustly
regulated by chemogenetic activation. The pipeline takes six DESeq2 DEG
tables (three timepoints × two control comparisons), calls a strict
consensus per timepoint, intersects the 2 h and 4 h consensus sets to get a
trajectory set, and produces publication-style figures plus replicate-level
QC and TF-family enrichment. The 20 min timepoint is carried as an earlier
reference point on the trajectory (it does not gate membership).

## What it does

Given six DEG files from a chemogenetic experiment:

- `20 min: CRE+CNO vs WT+CNO`  (genetic control, 20 min timepoint)
- `20 min: CRE+CNO vs CRE+SAL` (vehicle control, 20 min timepoint)
- `2 h: CRE+CNO vs WT+CNO`   (genetic control, 2 h timepoint)
- `2 h: CRE+CNO vs CRE+SAL`  (vehicle control, 2 h timepoint)
- `4 h: CRE+CNO vs WT+CNO`   (genetic control, 4 h timepoint)
- `4 h: CRE+CNO vs CRE+SAL`  (vehicle control, 4 h timepoint)

the app produces:

1. **Strict consensus** per timepoint: genes significant in *both* control
   comparisons with concordant `log2FoldChange` sign.
2. **Trajectory set**: genes that are strict-consensus at *both* 2 h and 4 h.
   The 20 min fold changes are attached to these genes as an earlier
   reference point (leading point in the trajectory line plot, extra column
   block in the heatmap, extra columns in the exported tables).
3. Trajectory **classes** per gene: `sustained_up`, `transient_up`,
   `sustained_down`, `transient_down`, `reversed` (based on direction
   and magnitude change between 2 h and 4 h, using the vehicle-control
   LFC as the canonical effect size). "Sustained" means the response is
   maintained or strengthens at 4 h; "transient" means the response is
   present at 2 h but weakens by 4 h.
4. **Figures** (volcanoes, UpSet-style overlap, LFC-2h-vs-4h scatter,
   per-class trajectory lines, signed-LFC heatmap, TF panel,
   TF-family enrichment dot plot, a core **circadian clock** panel
   (antiphase bars, clock-face dial, phase-vs-log₂FC, TTFL network,
   phase-ordered heatmap), replicate PCA, sample correlation).
5. **Downloads**: every figure as PDF (vector, type-42 fonts, editable in
   Illustrator) and a single XLSX with the consensus, trajectory, and
   TF-enrichment tables.

## Quick start

```bash
# 1. Create a fresh virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload the six DEG files via
the sidebar — once all six are present, the analysis runs.

## Input file format

Despite the `.xls` extension, the DEG files are tab-separated UTF-8 text.
The loader uses `pandas.read_csv(sep="\t", encoding="utf-8-sig")`.

### Required columns

| Column           | Type   | Description                                  |
|------------------|--------|----------------------------------------------|
| `gene_id`        | str    | Ensembl mouse gene ID (`ENSMUSG…`)           |
| `gene_name`      | str    | Gene symbol used as the join key             |
| `log2FoldChange` | float  | DESeq2 LFC (CRE vs reference)                |
| `pvalue`         | float  | Raw p-value                                  |
| `padj`           | float  | BH-adjusted p-value (used for significance)  |

### Optional annotation columns (carried through if present)

`gene_biotype`, `gene_description`, `tf_family`, `gene_chr`, `gene_start`,
`gene_end`, `gene_strand`, `gene_length`. These appear in the consensus,
trajectory, and XLSX exports for biological context.

### Optional sample columns (used by the QC tab)

Per-replicate normalized counts named `Cre_Q###` or `Wt_Q###`. The loader
ignores anything ending in `_count` or `_fpkm` as well as group-mean
columns named like `CRE_CNO4h` or `Wt_CNO4h`.

**Group assignment** in `load_samples`:

- Genetic-control files (`Cre_*` and `Wt_*` both present): `Cre_*` →
  group A (CRE+CNO), `Wt_*` → group B (WT+CNO). Order of columns
  doesn't matter.
- Vehicle-control files (only `Cre_*` present): the loader assumes the
  first half of detected sample columns are CRE+CNO and the second half
  are CRE+SAL, matching the column-order convention in this dataset.

If neither rule applies the loader raises with a clear `ValueError`.

## Pipeline details

### Strict consensus per timepoint

A gene is consensus at timepoint *t* iff:

```
padj_genetic       < padj_thresh   AND
padj_vehicle       < padj_thresh   AND
|lfc_genetic|     ≥ lfc_thresh    AND
|lfc_vehicle|     ≥ lfc_thresh    AND
sign(lfc_genetic) = sign(lfc_vehicle)
```

The thresholds are configurable in the sidebar (`padj_thresh = 0.05`
and `lfc_thresh = 0` by default). Requiring concordant sign in both
controls subtracts off-target effects of CNO injection (the vehicle
comparison) and genetic background (the WT comparison) in a single
filter.

### Trajectory set

The trajectory set is the **intersection** of the 2 h and 4 h consensus
sets. The canonical effect size used for plots is
`log2FoldChange(CRE+CNO vs CRE+SAL)`, which is the most direct readout of
the chemogenetic activation.

The 20 min timepoint does **not** change trajectory membership. Its fold
changes are attached to the trajectory table by `add_earlier_reference`
(`analysis.py`) as reference-only columns (`lfc_20m`, `lfc_genetic_20m`,
`lfc_vehicle_20m`), pulled from the raw 20 min DEG tables so every tested
gene is covered. Trajectory genes not tested at 20 min carry `NaN` there —
their trajectory line simply starts at the 2 h point and their 20 min
heatmap cells are drawn blank. The 20 min timepoint still gets its own
strict consensus (shown in the Overview, Tables, volcano, and overlap
views) exactly like 2 h and 4 h.

### Trajectory classes

For each gene `g` in the trajectory set, with `lfc_2h` and `lfc_4h` from
the vehicle-control comparison:

| Class             | Rule                                                          |
|-------------------|---------------------------------------------------------------|
| `sustained_up`    | `lfc_2h > 0` and `lfc_4h > 0` and `|lfc_4h| ≥ |lfc_2h|`        |
| `transient_up`    | `lfc_2h > 0` and `lfc_4h > 0` and `|lfc_4h| < |lfc_2h|`        |
| `sustained_down`  | `lfc_2h < 0` and `lfc_4h < 0` and `|lfc_4h| ≥ |lfc_2h|`        |
| `transient_down`  | `lfc_2h < 0` and `lfc_4h < 0` and `|lfc_4h| < |lfc_2h|`        |
| `reversed`        | sign(`lfc_2h`) ≠ sign(`lfc_4h`) (rare given concordance gate) |

Class is stored as an ordered `pandas.Categorical` so downstream
groupby/sort operations preserve the canonical ordering.

### TF-family enrichment

The TFs tab runs a one-sided Fisher's exact test for each `tf_family`
that has at least three TFs in the foreground (the trajectory set). The
background pool is the union of all genes tested across the four DEG
files **with the foreground subtracted** — this avoids double-counting
the in-set TFs and biasing the test toward the null. p-values are
adjusted by Benjamini–Hochberg.

The dot plot caps `log2(odds_ratio)` at ±6 (≈ 64-fold). Off-scale rows
are drawn as `>` / `<` triangles at the cap; rows with undefined OR
(family entirely absent from the foreground or background) appear as
hollow rings at x = 0.

### Circadian clock module

A curated panel of core clock genes (`CLOCK_GENES` in `analysis.py`) is
looked up by symbol (case-insensitive, with aliases — e.g. `Bmal1→Arntl`,
`Dec1/2→Bhlhe40/41`, `Rev-erbβ→Nr1d2`, `Chrono→Ciart`) in whichever contrast
you select. Each gene carries a **role** in the transcription–translation
feedback loop (activators, Per/Cry, Rev-erb/ROR, Dec, PAR-bZip/D-box outputs,
accessory) and a **phase family** (`positive`, `repressive`, `accessory`).

- The **bar chart** (`clock_bars`) shows each gene's log₂FC at a chosen
  timepoint, grouped by role, coloured by sign, with non-significant bars
  faded. The antiphase state reads directly: the positive limb up while the
  repressive limb / outputs go down (or vice versa).
- The **trajectory** (`clock_trajectory`) draws each clock gene across
  20 min → 2 h → 4 h coloured by phase family, so you can watch the limbs
  pull apart over time.
- The **antiphase separation** metric is `mean(log₂FC of positive-limb
  genes) − mean(log₂FC of repressive-limb/output genes)` over the detected
  genes at the selected snapshot; a large positive value is the classic
  driven-antiphase signature.

**Phase-based views.** Each core clock gene also carries a reference peak
phase (`peak_zt`, ZT 0–24) — an approximate mouse peripheral-clock acrophase
from published atlases (e.g. Zhang 2014), **not** a rhythm measured in this
experiment. Genes with no robust rhythm (`Clock`, the post-translational
accessory factors) carry `None` and are dropped from these figures. Edit the
`peak_zt` column in `CLOCK_GENES` to use your own reference; every phase
figure reads straight from it. The views:

- **Clock-face dial** (`clock_dial`) — polar plot with each gene at its peak
  ZT (ZT0 top, clockwise), radius = `|log₂FC|`, colour = signed log₂FC, night
  (ZT12–24) shaded. A donut hole spreads unchanged genes around the ring; only
  moved genes (significant or `|log₂FC| ≥ 0.5`) are labelled.
- **Phase-vs-log₂FC** (`clock_phase_scatter`) — log₂FC against reference peak
  phase, with a least-squares **cosine fit** (`fit_phase_cosine`) summarising
  whether regulation is a coherent function of phase (reports the fitted peak
  ZT, amplitude, and R²). This is a fit of the perturbation effect *across
  genes by their reference phase*, not a time-series rhythmicity test.
- **TTFL network** (`clock_network`) — the core feedback-loop schematic
  (BMAL1/CLOCK → Per/Cry/Rev-erb/Dec/outputs, with ROR/Rev-erb acting back on
  *Bmal1*) with each module node coloured by the mean log₂FC of its genes;
  activation vs repression edges are distinguished.
- **Phase-ordered heatmap** (`clock_phase_heatmap`) — clock genes as rows
  sorted by peak phase × the three timepoints, with a cyclic phase swatch on
  the left.

Not included on purpose: circadian rhythmicity methods (JTK_CYCLE, RAIN,
MetaCycle, cosinor over time, actograms). Those need day-spanning sampling at
fixed circadian time, which this 20 min / 2 h / 4 h post-stimulus design does
not provide.

#### Clock-arrest evidence

Two analyses bear on the hypothesis that the oscillator has **stalled at the
Bmal1-high state** (as peripheral clocks do in torpor / hibernation):

- **Settling dynamics** (`clock_settling_table` / `clock_settling`) — compares
  each clock gene's log₂FC at the last two timepoints. Genes are classed
  `held` (plateau on the identity line), `amplifying` (still moving out),
  `reverting` (returning to baseline), or `reversed`. Arrest predicts the
  displacement **holds** and that repressive-limb targets **stay suppressed**
  rather than re-rising (a turning loop would start re-driving them). The tab
  reports a *persistence* fraction and a *targets-staying-suppressed* fraction.
- **Torpor concordance** (`torpor_concordance` / `torpor_concordance_plot`) —
  tests whether our clock-gene log₂FC signs match a curated torpor /
  hibernation peripheral-clock reference direction (`TORPOR_CLOCK_SIGNATURE`,
  editable, sourced from Revel 2007 / Williams 2012 and hibernation reviews;
  the torpor literature is heterogeneous). Reports concordant / total and a
  one-sided binomial p-value.

**These are necessary, not sufficient.** A 20 min → 4 h window cannot separate
*arrest* from a *phase shift* or a *damped oscillation* — all three look like a
sustained displacement. Deciding among them needs a **circadian time course ±
activation** (24–48 h, analysed with JTK/RAIN/MetaCycle), a **real-time
PER2::LUC reporter** in explants to watch amplitude damp, a **reversibility**
test, and **temperature control** (to separate a clock effect from the
temperature drop of torpor). The tab spells this out in an "Interpretation &
caveats" box.

Genes absent from a file are reported as "not detected" and left out of the
bar chart (kept as `NaN` in the table). The clock table downloads separately
as `clock_genes.xlsx`.

### Replicate QC

For each uploaded file the QC tab runs:

- **PCA**: log₂(normalized counts + 1) on the top 2 000 most-variable
  genes, samples centred, SVD-based PCA via `numpy.linalg.svd`. Genes
  with any non-finite value or zero variance are dropped before logging
  to keep NaNs from masquerading as zero-count genes. Returns variance
  explained per component.
- **Sample correlation**: pairwise Pearson r on log₂ counts, rendered as
  a heatmap with white separator lines between groups.

## App layout

| Tab           | What you'll see                                              |
|---------------|--------------------------------------------------------------|
| Overview      | Pipeline summary, per-comparison funnel (tested / sig / up / down at the current thresholds), and trajectory class counts. |
| Volcanoes     | One volcano per uploaded DEG file (six panels). Trajectory genes circled. Off-scale outliers shown as triangles at the boundary so they don't compress the panel. ↓/↑ counts in the corners. |
| Overlap       | UpSet-style bar + dot plot of significant-gene overlaps across the six contrasts. The all-6 intersection is always pinned. |
| Trajectories  | LFC-2h-vs-4h scatter (with Spearman ρ) and faceted per-class line plot. When 20 min reference values are present, each line starts at a leading 20 min point (20 m → 2 h → 4 h). The largest-|LFC| genes get an endpoint dot and a leader line to a vertically-spread label, so each label points unambiguously at its trajectory. |
| Heatmap       | Signed-LFC heatmap of consensus genes, with a leading 20 min column block (20m vs WT / 20m vs SAL) followed by the 2 h and 4 h contrasts, a class swatch on the right, and a horizontal colourbar at the bottom. |
| TFs           | Per-class TF bars (gene · tf_family) plus the TF-family enrichment dot plot and table. |
| Circadian     | Core clock panel: antiphase-separation summary, a diverging bar chart of clock-gene log₂FC grouped by TTFL role, a clock-gene trajectory across 20 min → 2 h → 4 h coloured by phase family, four phase-based views (clock-face dial, phase-vs-log₂FC with cosine fit, TTFL network, phase-ordered heatmap), and clock-arrest evidence (settling dynamics + torpor-signature concordance with a caveats box). Contrast + snapshot-timepoint selectors; downloadable clock-gene table. |
| QC            | Six sub-tabs (one per uploaded file) with PCA and pairwise correlation. |
| Tables        | Browsable consensus and trajectory tables; bundled XLSX download with consensus_20m, consensus_2h, consensus_4h, trajectories, and (when populated) tf_enrichment sheets. |

## Output

Every figure has its own `Download PDF` button. The Tables tab also
provides a single `consensus_degs.xlsx` with up to five sheets:

- `consensus_20m`    — strict consensus at 20 min with both LFCs and padj
- `consensus_2h`     — strict consensus at 2 h with both LFCs and padj
- `consensus_4h`     — strict consensus at 4 h
- `trajectories`     — intersection set with class, 2 h and 4 h LFCs, the 20 min reference LFCs, padj, delta, and any annotation columns
- `tf_enrichment`    — per-tf_family Fisher exact results (only present if any family met the size filter)

PDFs use type-42 fonts so all text is editable in Illustrator. Sans-serif
body type is Arial (falls back to Helvetica → DejaVu Sans).

## Module layout

| File              | Purpose                                                      |
|-------------------|--------------------------------------------------------------|
| `analysis.py`     | Loaders, consensus and trajectory logic, TF enrichment, sample-level QC helpers (`compute_pca`, `sample_correlation`). |
| `plots.py`        | Matplotlib plotting functions. Style is applied per-call via `apply_style()`; figures are returned (not saved) so the app can serve PDFs and inline previews. |
| `app.py`          | Streamlit UI. File uploaders, sliders, tabs. Caches loaded DEG and sample tables with `@st.cache_data`. |
| `requirements.txt`| Pinned-floor dependencies.                                   |

## Troubleshooting

**"DEG file missing required columns"** — the loader needs
`gene_id`, `gene_name`, `log2FoldChange`, `pvalue`, `padj`. Check for
typos or whitespace in the header line. The loader uses `utf-8-sig`
encoding so a BOM won't trip the first-column check.

**"Sample loader: no per-sample columns matched (Cre|Wt)_Q###"** —
the QC tab couldn't find any sample columns matching that pattern. If
your replicate IDs use a different scheme, the regex
`SAMPLE_COL_RE` in `analysis.py` is the place to widen.

**"Sample loader: single-prefix file with odd sample count"** — a
vehicle-control file has an odd number of `Cre_Q###` columns; the loader
can't infer where group A ends and group B begins. Either fix the file
or pre-split.

**Heatmap slider is missing** — when fewer than four trajectory genes
exist, the heatmap shows them all without a slider.

**`adjustText` not installed** — volcano gene labels fall back to no
repulsion (they may overlap). Install `adjustText` for nicer volcanoes. The
trajectory line plot does not use `adjustText`; its labels are placed with a
built-in deterministic leader-line layout.

**`scipy` not installed** — TF enrichment quietly returns an empty
DataFrame. Spearman ρ in the LFC scatter falls back to Pearson r.
Install `scipy` for the proper statistics.

## Defaults you can change

- `padj_thresh` and `lfc_thresh` in the sidebar.
- `min_family_size=3` in `tf_enrichment` (analysis.py) — minimum number
  of foreground TFs before a family is tested.
- `n_top_var=2000` in `compute_pca` — how many high-variance genes feed
  into PCA.
- `max_genes` slider in the heatmap tab.

## Conventions

- Significance uses `padj`, never raw `pvalue`.
- All `log2FoldChange` values are CRE relative to control (positive = up
  in CRE+CNO, negative = up in control). The vehicle-control LFC is the
  canonical effect size for trajectory plots.
- Class names use snake_case in code, "sustained up" etc. in figure titles.
