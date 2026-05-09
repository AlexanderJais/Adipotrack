"""Tests for analysis.py.

Focus is on the loader edge cases the audit caught, plus enough
coverage of the consensus / trajectory / TF-enrichment paths that a
future change is likely to break a test before it ships.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pandas as pd
import pytest

from analysis import (
    _bh_adjust,
    _column_token,
    _label_tokens,
    TimepointInputs,
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


# ---- load_deg -------------------------------------------------------------


class TestLoadDeg:
    def test_happy_path(self, deg_genetic_2h_bytes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_deg(io.BytesIO(deg_genetic_2h_bytes))
        assert {"gene_id", "gene_name", "log2FoldChange", "pvalue", "padj"} <= set(df.columns)
        assert "tf_family" in df.columns  # annotation kept
        assert df["log2FoldChange"].dtype.kind == "f"
        assert df["padj"].dtype.kind == "f"

    def test_dedup_warning(self, deg_genetic_2h_bytes):
        # Fixture contains a duplicate gene_name (Fos appears twice). The
        # loader should keep the row with the smallest padj and warn.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            df = load_deg(io.BytesIO(deg_genetic_2h_bytes))
        assert (df["gene_name"] == "Fos").sum() == 1
        # The kept Fos should be the one with the lower padj.
        kept = df.loc[df["gene_name"] == "Fos", "padj"].iloc[0]
        assert kept == pytest.approx(1e-8)
        assert any("collapsed" in str(w.message) for w in captured)

    def test_missing_required_column(self):
        bad = b"gene_id\tgene_name\tlog2FoldChange\tpadj\nA\tFoo\t1.0\t0.05\n"
        with pytest.raises(ValueError, match="missing required columns"):
            load_deg(io.BytesIO(bad))

    def test_rejects_binary_xls(self):
        # Compound File Binary (Excel 97-2003) magic bytes.
        with pytest.raises(ValueError, match="binary Excel"):
            load_deg(io.BytesIO(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100))

    def test_rejects_binary_xlsx(self):
        # ZIP magic — xlsx is just a zip.
        with pytest.raises(ValueError, match="xlsx"):
            load_deg(io.BytesIO(b"PK\x03\x04" + b"\x00" * 100))

    def test_utf8_bom_tolerated(self):
        body = (
            "gene_id\tgene_name\tlog2FoldChange\tpvalue\tpadj\n"
            "A\tFoo\t1.0\t0.001\t0.01\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_deg(io.BytesIO(b"\xef\xbb\xbf" + body.encode()))
        assert df["gene_name"].iloc[0] == "Foo"


# ---- load_samples ---------------------------------------------------------


class TestLoadSamples:
    def test_legacy_mixed_prefix(self):
        header = "gene_id\tCre_Q1\tCre_Q2\tWt_Q3\tWt_Q4\tlog2FoldChange\tpvalue\tpadj\tgene_name"
        row = "g\t1\t2\t3\t4\t0.1\t0.01\t0.05\tFoo"
        sd = load_samples(io.BytesIO(f"{header}\n{row}".encode()),
                          "CRE+CNO_2h", "WT+CNO_2h")
        assert list(sd.metadata["sample"]) == ["Cre_Q1", "Cre_Q2", "Wt_Q3", "Wt_Q4"]
        assert list(sd.metadata["group"]) == [
            "CRE+CNO_2h", "CRE+CNO_2h", "WT+CNO_2h", "WT+CNO_2h",
        ]

    def test_legacy_single_prefix_uses_positional_split(self):
        header = "gene_id\tCre_Q1\tCre_Q2\tCre_Q3\tCre_Q4\tlog2FoldChange\tpvalue\tpadj\tgene_name"
        row = "g\t1\t2\t3\t4\t0.1\t0.01\t0.05\tFoo"
        sd = load_samples(io.BytesIO(f"{header}\n{row}".encode()),
                          "CRE+CNO_2h", "CRE+SAL_2h")
        assert list(sd.metadata["group"]) == [
            "CRE+CNO_2h", "CRE+CNO_2h", "CRE+SAL_2h", "CRE+SAL_2h",
        ]

    def test_legacy_single_prefix_odd_count_raises(self):
        header = "gene_id\tCre_Q1\tCre_Q2\tCre_Q3\tlog2FoldChange\tpvalue\tpadj\tgene_name"
        row = "g\t1\t2\t3\t0.1\t0.01\t0.05\tFoo"
        with pytest.raises(ValueError, match="odd sample count"):
            load_samples(io.BytesIO(f"{header}\n{row}".encode()),
                         "CRE+CNO_2h", "CRE+SAL_2h")

    def test_new_scheme_genetic(self, sample_genetic_bytes):
        sd = load_samples(io.BytesIO(sample_genetic_bytes),
                          "CRE+CNO_2h", "WT+CNO_2h")
        groups = sd.metadata.set_index("sample")["group"].to_dict()
        for s in ("S897_TC2", "T296_TC2", "S895_TC2", "S535_TC2"):
            assert groups[s] == "CRE+CNO_2h", f"{s} should be CRE+CNO"
        for s in ("S869_WC2", "S894_WC2", "S896_WC2", "S893_WC2"):
            assert groups[s] == "WT+CNO_2h", f"{s} should be WT+CNO"

    def test_new_scheme_interleaved_columns_regression(
        self, sample_vehicle_interleaved_bytes,
    ):
        """Audit regression: with interleaved columns, the OLD _column_token
        (anchored on 'S') would return None for T296_TS2 etc., dropping
        load_samples into the positional fallback. Positional split would
        then misgroup every other sample. The current token resolver
        handles arbitrary letter prefixes."""
        sd = load_samples(io.BytesIO(sample_vehicle_interleaved_bytes),
                          "CRE+CNO_2h", "CRE+SAL_2h")
        groups = sd.metadata.set_index("sample")["group"].to_dict()
        for s in ("S897_TC2", "S870_TC2", "S895_TC2", "S535_TC2"):
            assert groups[s] == "CRE+CNO_2h", (
                f"{s} should be CNO; if this fails, _column_token's "
                "regex regressed against arbitrary sample-id prefixes"
            )
        for s in ("T296_TS2", "T298_TS2", "T641_TS2", "T640_TS2"):
            assert groups[s] == "CRE+SAL_2h", f"{s} should be SAL"

    def test_quantification_suffixes_skipped(self):
        # Columns ending in _count / _fpkm / group means must NOT be picked
        # up as samples even though they are numeric.
        header = (
            "gene_id\tS897_TC2\tS869_WC2"
            "\tS897_TC2_count\tS869_WC2_count"
            "\tS897_TC2_fpkm\tS869_WC2_fpkm"
            "\tPnocCreCno2\tWtCno2"
            "\tlog2FoldChange\tpvalue\tpadj\tgene_name"
        )
        row = "g\t1\t2\t10\t20\t0.1\t0.2\t1.5\t1.6\t0.5\t0.001\t0.01\tFoo"
        sd = load_samples(io.BytesIO(f"{header}\n{row}".encode()),
                          "CRE+CNO_2h", "WT+CNO_2h")
        assert list(sd.metadata["sample"]) == ["S897_TC2", "S869_WC2"]

    def test_no_sample_columns_raises(self):
        header = "gene_id\tlog2FoldChange\tpvalue\tpadj\tgene_name"
        row = "g\t0.5\t0.001\t0.01\tFoo"
        with pytest.raises(ValueError, match="no per-sample count columns"):
            load_samples(io.BytesIO(f"{header}\n{row}".encode()),
                         "CRE+CNO_2h", "WT+CNO_2h")


# ---- _column_token / _label_tokens (private helpers) -----------------------


@pytest.mark.parametrize("col,expected", [
    ("S897_TC2", "TC"),
    ("T296_TS2", "TS"),
    ("Cre_Q927", "CRE"),
    ("Wt_Q919", "WT"),
    ("S897_TC2_count", None),  # quantification suffix — not a sample
    ("PnocCreCno2", None),     # group mean
    ("gene_id", None),
    ("baseMean", None),
])
def test_column_token(col, expected):
    assert _column_token(col) == expected


@pytest.mark.parametrize("label,must_contain", [
    ("CRE+CNO_2h", {"CRE", "TC", "CT"}),
    ("WT+CNO_4h",  {"WT", "WC", "CW"}),
    ("CRE+SAL_2h", {"CRE", "TS", "ST"}),
])
def test_label_tokens(label, must_contain):
    assert must_contain <= _label_tokens(label)


# ---- consensus_at_timepoint ------------------------------------------------


class TestConsensus:
    def _load_pair(self, gen_bytes, veh_bytes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return load_deg(io.BytesIO(gen_bytes)), load_deg(io.BytesIO(veh_bytes))

    def test_concordant_signs_kept(self, deg_genetic_2h_bytes, deg_vehicle_2h_bytes):
        g, v = self._load_pair(deg_genetic_2h_bytes, deg_vehicle_2h_bytes)
        cons = consensus_at_timepoint(TimepointInputs(genetic=g, vehicle=v),
                                      padj_thresh=0.05)
        # Five genes are sig + concordant in both files; Mxd1 is sig but
        # discordant; Sox2 is below padj.
        assert set(cons["gene_name"]) == {"Fos", "Jun", "Atf3", "Dnmt3a", "Foxp1"}

    def test_padj_threshold_filters(self, deg_genetic_2h_bytes, deg_vehicle_2h_bytes):
        g, v = self._load_pair(deg_genetic_2h_bytes, deg_vehicle_2h_bytes)
        cons = consensus_at_timepoint(TimepointInputs(genetic=g, vehicle=v),
                                      padj_thresh=1e-6)
        # Tighter threshold should drop Atf3 and Foxp1, which have larger padj.
        assert "Atf3" not in set(cons["gene_name"])
        assert "Foxp1" not in set(cons["gene_name"])

    def test_lfc_threshold_filters(self, deg_genetic_2h_bytes, deg_vehicle_2h_bytes):
        g, v = self._load_pair(deg_genetic_2h_bytes, deg_vehicle_2h_bytes)
        cons = consensus_at_timepoint(TimepointInputs(genetic=g, vehicle=v),
                                      padj_thresh=0.05, lfc_thresh=2.0)
        # Only Fos has |lfc| ≥ 2.0 in both contrasts.
        assert set(cons["gene_name"]) == {"Fos"}

    def test_canonical_lfc_is_vehicle(self, deg_genetic_2h_bytes, deg_vehicle_2h_bytes):
        g, v = self._load_pair(deg_genetic_2h_bytes, deg_vehicle_2h_bytes)
        cons = consensus_at_timepoint(TimepointInputs(genetic=g, vehicle=v))
        row = cons.set_index("gene_name").loc["Fos"]
        assert row["lfc"] == row["lfc_vehicle"]


# ---- trajectories ----------------------------------------------------------


class TestTrajectories:
    def _build(self, deg_genetic_2h_bytes, deg_vehicle_2h_bytes,
               deg_genetic_4h_bytes, deg_vehicle_4h_bytes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g2 = load_deg(io.BytesIO(deg_genetic_2h_bytes))
            v2 = load_deg(io.BytesIO(deg_vehicle_2h_bytes))
            g4 = load_deg(io.BytesIO(deg_genetic_4h_bytes))
            v4 = load_deg(io.BytesIO(deg_vehicle_4h_bytes))
        c2 = consensus_at_timepoint(TimepointInputs(genetic=g2, vehicle=v2))
        c4 = consensus_at_timepoint(TimepointInputs(genetic=g4, vehicle=v4))
        return trajectories(c2, c4)

    def test_class_assignments(
        self, deg_genetic_2h_bytes, deg_vehicle_2h_bytes,
        deg_genetic_4h_bytes, deg_vehicle_4h_bytes,
    ):
        traj = self._build(deg_genetic_2h_bytes, deg_vehicle_2h_bytes,
                           deg_genetic_4h_bytes, deg_vehicle_4h_bytes)
        cls = traj.set_index("gene_name")["class"].astype(str).to_dict()

        # Fos:    +2.2 → +2.6  (up, |4h| > |2h|)  -> sustained_up
        # Jun:    +1.5 → +0.9  (up, |4h| < |2h|)  -> transient_up
        # Atf3:   +1.0 → +0.6  (up, |4h| < |2h|)  -> transient_up
        # Dnmt3a: -1.3 → -1.7  (dn, |4h| > |2h|)  -> sustained_down
        # Foxp1:  -0.8 → +1.0  (sign flip)        -> reversed
        assert cls["Fos"] == "sustained_up"
        assert cls["Jun"] == "transient_up"
        assert cls["Atf3"] == "transient_up"
        assert cls["Dnmt3a"] == "sustained_down"
        assert cls["Foxp1"] == "reversed"

    def test_delta_lfc_computed(
        self, deg_genetic_2h_bytes, deg_vehicle_2h_bytes,
        deg_genetic_4h_bytes, deg_vehicle_4h_bytes,
    ):
        traj = self._build(deg_genetic_2h_bytes, deg_vehicle_2h_bytes,
                           deg_genetic_4h_bytes, deg_vehicle_4h_bytes)
        row = traj.set_index("gene_name").loc["Fos"]
        assert row["delta_lfc"] == pytest.approx(row["lfc_4h"] - row["lfc_2h"])


# ---- TF enrichment --------------------------------------------------------


class TestTfEnrichment:
    EXPECTED_COLS = [
        "tf_family", "n_fg", "n_fg_total", "n_bg", "n_bg_total",
        "odds_ratio", "p_value", "q_value",
    ]

    def _build_inputs(self):
        # Foreground: 3 bZIP + 3 Forkhead. Background has Forkhead at a
        # similar rate but bZIP rare, so bZIP should look enriched.
        fg = pd.DataFrame({
            "gene_name": ["Fos", "Jun", "Atf3", "Foxp1", "Foxp2", "Foxa1"],
            "tf_family": ["bZIP", "bZIP", "bZIP",
                          "Forkhead", "Forkhead", "Forkhead"],
        })
        bg = pd.DataFrame({
            "gene_name": [f"G{i}" for i in range(60)],
            "tf_family": (["-"] * 50 + ["Forkhead"] * 8 + ["bZIP"] * 2),
        })
        return fg, bg

    def test_columns_when_filter_excludes_everything(self):
        # min_family_size=10 → no family qualifies.
        fg, bg = self._build_inputs()
        out = tf_enrichment(fg, bg, min_family_size=10)
        assert list(out.columns) == self.EXPECTED_COLS
        assert len(out) == 0

    def test_missing_tf_family_column(self):
        fg = pd.DataFrame({"gene_name": ["A", "B"]})
        bg = pd.DataFrame({"gene_name": ["X", "Y"]})
        out = tf_enrichment(fg, bg)
        assert list(out.columns) == self.EXPECTED_COLS

    def test_happy_path(self):
        scipy = pytest.importorskip("scipy")  # skip if scipy not installed
        fg, bg = self._build_inputs()
        out = tf_enrichment(fg, bg, min_family_size=3)
        assert "bZIP" in set(out["tf_family"])
        # Foreground has 3 of 4 TF rows in bZIP → strong over-representation.
        bzip = out[out["tf_family"] == "bZIP"].iloc[0]
        assert bzip["n_fg"] == 3
        assert bzip["odds_ratio"] > 1.0
        assert 0 <= bzip["p_value"] <= 1
        assert 0 <= bzip["q_value"] <= 1


# ---- _bh_adjust -----------------------------------------------------------


def test_bh_adjust_known_values():
    # Reference: Benjamini-Hochberg of [0.01, 0.02, 0.03, 0.04, 0.05]
    # gives [0.05, 0.05, 0.05, 0.05, 0.05].
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    out = _bh_adjust(p)
    assert np.allclose(out, [0.05, 0.05, 0.05, 0.05, 0.05])


def test_bh_adjust_monotone_in_rank():
    # After BH, q-values are monotone non-decreasing in rank.
    rng = np.random.default_rng(0)
    p = np.sort(rng.uniform(size=50))
    q = _bh_adjust(p)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_bh_adjust_empty():
    assert len(_bh_adjust(np.array([]))) == 0


# ---- compute_pca / sample_correlation -------------------------------------


def test_compute_pca_shape_and_var_explained():
    rng = np.random.default_rng(0)
    expr = pd.DataFrame(
        rng.uniform(1, 1e3, size=(500, 6)),
        columns=[f"S{i}" for i in range(6)],
    )
    scores, var_exp = compute_pca(expr, n_top_var=200)
    assert scores.shape == (6, 2)
    assert var_exp.shape == (2,)
    assert 0.0 < var_exp.sum() <= 1.0 + 1e-9


def test_compute_pca_filters_nonfinite():
    expr = pd.DataFrame(
        np.array([[np.nan, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]]),
        columns=["S1", "S2", "S3"],
    )
    scores, var_exp = compute_pca(expr, n_top_var=10)
    assert scores.shape == (3, 2)


def test_sample_correlation_self_one():
    rng = np.random.default_rng(0)
    expr = pd.DataFrame(rng.uniform(1, 1e3, size=(100, 4)),
                        columns=[f"S{i}" for i in range(4)])
    corr = sample_correlation(expr)
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)


# ---- to_excel_bytes -------------------------------------------------------


def test_to_excel_bytes_collision_raises():
    sheets = {
        "consensus_genes_at_2_hour_timepoint": pd.DataFrame({"a": [1]}),
        "consensus_genes_at_2_hour_timepoint_extra": pd.DataFrame({"a": [2]}),
    }
    with pytest.raises(ValueError, match="collision"):
        to_excel_bytes(sheets)


def test_to_excel_bytes_happy_path():
    pytest.importorskip("openpyxl")
    sheets = {"foo": pd.DataFrame({"a": [1, 2]})}
    out = to_excel_bytes(sheets)
    assert out.startswith(b"PK\x03\x04")  # xlsx is a zip


# ---- Misc -----------------------------------------------------------------


def test_is_tf_handles_missing_column():
    df = pd.DataFrame({"gene_name": ["A", "B"]})
    mask = is_tf(df)
    assert mask.dtype == bool
    assert not mask.any()


def test_sig_set_filters_by_padj():
    df = pd.DataFrame({
        "gene_name": ["A", "B", "C"],
        "padj": [0.01, 0.5, 0.001],
    })
    s = sig_set(df, padj_thresh=0.05)
    assert s == {"A", "C"}
