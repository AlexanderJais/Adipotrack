"""Tests for debug_log.build_log."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pandas as pd
import pytest

from debug_log import build_log


def _fake_uploaded_file(name: str, content: bytes):
    """Stand-in for a Streamlit UploadedFile: needs .name and .getvalue()."""
    return SimpleNamespace(name=name, getvalue=lambda: content)


def test_build_log_minimal():
    text = build_log(
        thresholds={"padj <": 0.05, "|log2FC| >=": 0.0},
        files={"2 h genetic": None, "2 h vehicle": None,
               "4 h genetic": None, "4 h vehicle": None},
        load_warnings={},
        counts={"consensus_2h": 0, "consensus_4h": 0, "trajectory": 0},
        funnel=pd.DataFrame(),
        class_counts=None,
        tf_enrichment=None,
    )
    assert "# Adipotrack debug log" in text
    assert "## Environment" in text
    assert "## Thresholds" in text
    assert "## Input files" in text
    assert "<not uploaded>" in text  # all four files are None
    assert "## Pipeline counts" in text


def test_build_log_includes_file_metadata():
    f = _fake_uploaded_file("foo.xls", b"hello\tworld\n")
    text = build_log(
        thresholds={},
        files={"2 h genetic": f},
        load_warnings={"2 h genetic": ["collapsed 5 duplicate rows"]},
        counts={},
        funnel=pd.DataFrame(),
        class_counts=None,
        tf_enrichment=None,
    )
    assert "foo.xls" in text
    assert "size_bytes:  12" in text  # len(b"hello\tworld\n") == 12
    assert "sha256[:16]:" in text
    assert "collapsed 5 duplicate rows" in text


def test_build_log_tf_enrichment_sorted_by_q_value():
    enr = pd.DataFrame({
        "tf_family": ["a", "b", "c"],
        "p_value":   [0.01, 0.5, 0.001],
        "q_value":   [0.04, 0.5, 0.002],   # 'c' should come first
    })
    text = build_log(
        thresholds={}, files={}, load_warnings={},
        counts={}, funnel=pd.DataFrame(),
        class_counts=None, tf_enrichment=enr,
    )
    # The first family listed in the TF section should be the one with the
    # smallest q_value, regardless of input row order.
    section = text.split("## TF enrichment")[-1]
    # pandas to_string(index=False) right-justifies, so the family letter
    # has leading whitespace. Walk lines and record the order families
    # appear after the column-header line.
    order = []
    for line in section.splitlines():
        stripped = line.strip()
        token = stripped.split(" ", 1)[0] if stripped else ""
        if token in {"a", "b", "c"}:
            order.append(token)
    assert order == ["c", "a", "b"]


def test_build_log_handles_empty_tf_enrichment_with_columns():
    """Even when tf_enrichment is non-None but empty (the schema-preserving
    return path), build_log shouldn't crash."""
    empty = pd.DataFrame(columns=[
        "tf_family", "n_fg", "n_fg_total", "n_bg", "n_bg_total",
        "odds_ratio", "p_value", "q_value",
    ])
    text = build_log(
        thresholds={}, files={}, load_warnings={},
        counts={}, funnel=pd.DataFrame(),
        class_counts=None, tf_enrichment=empty,
    )
    assert "<none>" in text  # the empty branch fires
