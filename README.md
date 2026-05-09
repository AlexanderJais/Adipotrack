# Consensus DEG explorer

Streamlit app and analysis library for finding genes that are robustly
regulated by chemogenetic activation. The pipeline takes four DESeq2 DEG
tables (two timepoints × two control comparisons), calls a strict
consensus per timepoint, intersects across timepoints to get a trajectory
set, and produces publication-style figures plus replicate-level QC and
TF-family enrichment.

## What it does

Given four DEG files from a chemogenetic experiment:

- `2 h: CRE+CNO vs WT+CNO`   (genetic control, 2 h timepoint)
- `2 h: CRE+CNO vs CRE+SAL`  (vehicle control, 2 h timepoint)
- `4 h: CRE+CNO vs WT+CNO`   (genetic control, 4 h timepoint)
- `4 h: CRE+CNO vs CRE+SAL`  (vehicle control, 4 h timepoint)

the app produces:

1. **Strict consensus** per timepoint: genes significant in *both* control
   comparisons with concordant `log2FoldChange` sign.
2. **Trajectory set**: genes that are strict-consensus at *both* 2 h and 4 h.
3. Trajectory **classes** per gene: `sustained_up`, `transient_up`,
   `sustained_down`, `transient_down`, `reversed` (based on direction
   and magnitude change between 2 h and 4 h, using the vehicle-control
   LFC as the canonical effect size). "Sustained" means the response is
   maintained or strengthens at 4 h; "transient" means the response is
   present at 2 h but weakens by 4 h.
4. **Figures** (volcanoes, UpSet-style overlap, LFC-2h-vs-4h scatter,
   per-class trajectory lines, signed-LFC heatmap, TF panel,
   TF-family enrichment dot plot, replicate PCA, sample correlation).
5. **Downloads**: every figure as PDF (vector, type-42 fonts, editable in
   Illustrator), a single XLSX with the consensus, trajectory, and
   TF-enrichment tables, and a plaintext debug-log snapshot for bug
   reports.

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

The app opens at `http://localhost:8501`. Upload the four DEG files via
the sidebar — once all four are present, the analysis runs.

## Input file format

Despite the `.xls` extension, the DEG files are tab-separated UTF-8 text.
The loader uses `pandas.read_csv(sep="\t", encoding="utf-8-sig")` and
sniffs the magic bytes first — a real binary `.xls` or `.xlsx` is
rejected with a clear message rather than a cryptic decode error.

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

Per-replicate normalized counts. Two naming schemes are recognized
(see `SAMPLE_COL_PATTERNS` in `analysis.py`):

- `Cre_Q###` / `Wt_Q###` — original lab convention.
- `<id>_<code><timepoint>` — current pipeline output, e.g. `S897_TC2`
  or `T296_TS2`. The single letters encode the experimental group:
  `T` = transgene (Cre), `W` = wildtype, `C` = CNO, `S` = saline. So
  `S897_TC2` is sample 897, transgene + CNO at 2 h; `T296_TS2` is
  sample 296, transgene + saline at 2 h.

Quantification suffixes (`*_count`, `*_fpkm`) and group-mean columns
(`PnocCreCno2`, `WtCno2`, …) are skipped because they don't end in
the trailing `<digits>` the pattern requires.

**Group assignment** (`load_samples`). Each detected column produces
an uppercase token (`TC`, `WC`, `TS`, `CRE`, `WT`, …). The label
strings passed in by the app (e.g. `"CRE+CNO_2h"` and `"WT+CNO_2h"`)
are mapped to the same token space, and each sample is assigned to
whichever label uniquely matches its token. This is order-independent
— interleaved files group correctly.

If token matching is ambiguous (legacy single-prefix file where every
column reads `Cre_*`), the loader falls back to a first-half /
second-half positional split. Files that resolve to neither raise
with a `ValueError` listing the columns it tried.

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

### Pathway / gene-set enrichment

The Pathways tab does the same one-sided Fisher + BH treatment but
with arbitrary gene sets you upload as a GMT file in the sidebar. The
universe is foreground ∪ background; gene-set members outside the
universe are dropped before testing (standard ORA convention).
Background = trajectory minus foreground, same as the TF tab.

GMT files are tab-separated:

```
SET_NAME<TAB>description<TAB>gene1<TAB>gene2<TAB>…
```

[MSigDB](https://www.gsea-msigdb.org/) ships hallmarks, GO BP, KEGG,
Reactome, TF target, and many other collections in this format. Use
the mouse-symbol version for mouse DEG tables. Lines starting with
`#` are skipped, descriptions are dropped, and duplicate set names
are merged.

Tunable in the tab: minimum set size (drops noisy small sets),
maximum set size (drops near-universe sets like "protein_coding"),
and how many top rows to plot. Sets with zero foreground overlap are
skipped automatically — they have OR = 0, no signal worth correcting
against.

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
| Gene lookup   | Type a gene symbol (case-insensitive) and see its log₂FC + padj across the four contrasts, the significance flag at the current sidebar thresholds, set membership (consensus 2 h, consensus 4 h, trajectory + class + Δlfc), and the carried annotations. Misses suggest close matches via `difflib`. |
| Volcanoes     | One volcano per uploaded DEG file. Trajectory genes circled. Off-scale outliers shown as triangles at the boundary so they don't compress the panel. ↓/↑ counts in the corners. |
| Overlap       | UpSet-style bar + dot plot of significant-gene overlaps across the four contrasts. The all-4 intersection is always pinned. |
| Trajectories  | LFC-2h-vs-4h scatter (with Spearman ρ) and faceted per-class line plot with one line per gene; the most extreme genes per class are labelled in a right-edge gutter (repelled with `adjustText`). |
| Heatmap       | Signed-LFC heatmap of consensus genes across the 4 contrasts, with class swatch on the right and a horizontal colourbar at the bottom. A `Hierarchical clustering` checkbox reorders rows by similarity (average linkage on euclidean distance over the 4-column LFC vector) and adds a row dendrogram on the left; class membership then appears as a per-row colour band plus a separate legend. |
| TFs           | Per-class TF bars (gene · tf_family) plus the TF-family enrichment dot plot and table. |
| Pathways      | Optional. Upload a `.gmt` gene-set file in the sidebar (MSigDB / GO / KEGG / Reactome) and the tab runs over-representation of each set in the trajectory foreground vs the union-minus-foreground background. Tunable min/max set size; results as a dot plot + sortable table with PDF + CSV downloads. |
| QC            | Four sub-tabs (one per uploaded file) with PCA and pairwise correlation. |
| Tables        | Browsable consensus and trajectory tables; bundled XLSX download with consensus_2h, consensus_4h, trajectories, and (when populated) tf_enrichment sheets. |

## Output

Every figure has its own `Download PDF` button. The Tables tab also
provides a single `consensus_degs.xlsx` with up to four sheets:

- `consensus_2h`     — strict consensus at 2 h with both LFCs and padj
- `consensus_4h`     — strict consensus at 4 h
- `trajectories`     — intersection set with class, both timepoints' LFCs, padj, delta, and any annotation columns
- `tf_enrichment`    — per-tf_family Fisher exact results (only present if any family met the size filter)

PDFs use type-42 fonts so all text is editable in Illustrator. Sans-serif
body type is Arial (falls back to Helvetica → DejaVu Sans).

A `Download debug log` button at the bottom of the sidebar produces
`adipotrack_debug.log` — a plaintext snapshot of Python and library
versions, the uploaded files (name, byte size, sha256 prefix, and any
loader warnings), the current thresholds, the pipeline counts, the
significance funnel, the trajectory class counts, and the top of the
TF-enrichment table. Attaching it to a bug report is enough to
reconstruct what the pipeline saw without sharing the original DEG
files.

## Module layout

| File              | Purpose                                                      |
|-------------------|--------------------------------------------------------------|
| `analysis.py`     | Loaders, consensus and trajectory logic, TF enrichment, sample-level QC helpers (`compute_pca`, `sample_correlation`). |
| `plots.py`        | Matplotlib plotting functions. Style is applied per-call via `apply_style()`; figures are returned (not saved) so the app can serve PDFs and inline previews. |
| `app.py`          | Streamlit UI. File uploaders, sliders, tabs. Caches loaded DEG and sample tables with `@st.cache_data`. |
| `debug_log.py`    | Builds the plaintext debug-log snapshot served by the sidebar download button. |
| `requirements.txt`| Pinned-floor dependencies.                                   |

## Troubleshooting

**"DEG file missing required columns"** — the loader needs
`gene_id`, `gene_name`, `log2FoldChange`, `pvalue`, `padj`. Check for
typos or whitespace in the header line. The loader uses `utf-8-sig`
encoding so a BOM won't trip the first-column check.

**"DEG file looks like a binary Excel workbook"** — you uploaded a real
`.xls` or `.xlsx` file rather than the tab-separated text the upstream
pipeline emits (which carries a `.xls` extension by convention but is
plain TSV). Re-export as TSV.

**"Sample loader: no per-sample count columns found. Headers seen: …"** —
none of the columns matched the patterns in `SAMPLE_COL_PATTERNS`
(`Cre_Q###` / `Wt_Q###` or `<id>_<code><tp>`). The full header is
included in the error so you can see what the loader saw; widen the
patterns in `analysis.py` if your replicate IDs use a different shape.

**"Sample loader: cannot resolve groups for [...] into 'CRE+CNO_2h' /
'CRE+SAL_2h'; odd sample count prevents a positional split"** —
token matching was ambiguous (e.g. every column reads `Cre_*`) and
the positional fallback can't split an odd number of columns. Either
rename the columns to encode the group, or fix the column count.

**Heatmap slider is missing** — when fewer than four trajectory genes
exist, the heatmap shows them all without a slider.

**`adjustText` not installed** — volcano labels and trajectory labels
fall back to either no repulsion (volcanoes) or a deterministic vertical
stacking pass (trajectories). Install `adjustText` for nicer figures.

**`scipy` not installed** — TF enrichment quietly returns an empty
DataFrame. Spearman ρ in the LFC scatter falls back to Pearson r.
Install `scipy` for the proper statistics.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the loaders (including both sample-column naming
schemes and the binary-Excel detection), consensus and trajectory
classification, TF enrichment schema invariants, BH adjustment,
PCA/correlation shapes, and a smoke pass over every figure function.
Two tests are explicit regression guards for the bugs the audit
caught: interleaved sample columns must group correctly via
`_column_token`, and `tf_enrichment_dot` must not double-render
off-scale points.

## Defaults you can change

- `padj_thresh` and `lfc_thresh` in the sidebar.
- `min_family_size=3` in `tf_enrichment` (`analysis.py`) — minimum
  number of foreground TFs before a family is tested.
- `min_set_size` / `max_set_size` controls in the Pathways tab (5 / 500
  by default) — gate which gene sets are tested.
- `n_top_var=2000` in `compute_pca` — how many high-variance genes
  feed into PCA. Note that `var_explained` is reported relative to the
  variance among these top-N genes, not the full transcriptome.
- `log_cap=6.0` in `tf_enrichment_dot` (`plots.py`) — clamps the
  `log₂(odds_ratio)` axis to ±6 (≈ 64-fold).
- `max_genes` slider and `Hierarchical clustering` toggle in the
  Heatmap tab. `linkage_method` and `distance_metric` keyword args on
  `heatmap()` (`"average"` and `"euclidean"` by default) let you swap in
  Ward / complete linkage or correlation distance from a notebook.

## Conventions

- Significance uses `padj`, never raw `pvalue`.
- All `log2FoldChange` values are CRE relative to control (positive = up
  in CRE+CNO, negative = up in control). The vehicle-control LFC is the
  canonical effect size for trajectory plots.
- Class names use snake_case in code, "sustained up" etc. in figure titles.
