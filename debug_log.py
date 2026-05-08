"""Build a snapshot debug log to download from the Streamlit app.

The log is generated on demand from the current state of the app — it is not
a persistent log of every action. The intent is that a user hitting an issue
can click "Download debug log" and attach the resulting text file to a bug
report so we can see the inputs, settings, library versions, and the shape of
the pipeline outputs without needing the original DEG files.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import platform
import sys
from io import StringIO
from typing import Any, Mapping, Sequence

import pandas as pd


def _version(mod_name: str) -> str:
    try:
        mod = __import__(mod_name)
    except Exception as e:
        return f"<not importable: {e.__class__.__name__}: {e}>"
    return getattr(mod, "__version__", "unknown")


def build_log(
    *,
    thresholds: Mapping[str, float],
    files: Mapping[str, Any],
    load_warnings: Mapping[str, Sequence[str]],
    counts: Mapping[str, int],
    funnel: pd.DataFrame,
    class_counts: pd.Series | None,
    tf_enrichment: pd.DataFrame | None,
) -> str:
    """Return a plaintext snapshot describing the current pipeline state."""
    out = StringIO()

    def w(line: str = "") -> None:
        out.write(line)
        out.write("\n")

    w("# Adipotrack debug log")
    w(f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    w()

    w("## Environment")
    w(f"  python:    {sys.version.split()[0]}")
    w(f"  platform:  {platform.platform()}")
    w(f"  executable:{sys.executable}")
    for pkg in (
        "streamlit", "pandas", "numpy", "scipy", "matplotlib",
        "statsmodels", "sklearn", "openpyxl", "adjustText",
    ):
        w(f"  {pkg:<10} {_version(pkg)}")
    w()

    w("## Thresholds")
    for k, v in thresholds.items():
        w(f"  {k}: {v}")
    w()

    w("## Input files")
    for label, f in files.items():
        if f is None:
            w(f"  [{label}] <not uploaded>")
            continue
        b = f.getvalue()
        h = hashlib.sha256(b).hexdigest()[:16]
        w(f"  [{label}]")
        w(f"    name:        {f.name}")
        w(f"    size_bytes:  {len(b):,}")
        w(f"    sha256[:16]: {h}")
        for msg in load_warnings.get(label, ()):
            w(f"    warning:     {msg}")
    w()

    w("## Pipeline counts")
    for k, v in counts.items():
        w(f"  {k}: {v}")
    w()

    w("## Per-comparison significance funnel")
    if funnel is not None and not funnel.empty:
        w(funnel.to_string(index=False))
    else:
        w("  <empty>")
    w()

    w("## Trajectory class counts")
    if class_counts is not None and len(class_counts):
        w(class_counts.to_string())
    else:
        w("  <empty>")
    w()

    w("## TF enrichment (top 25 by q-value)")
    if tf_enrichment is not None and not tf_enrichment.empty:
        df = tf_enrichment.copy()
        sort_col = next(
            (c for c in ("q_value", "qvalue", "padj") if c in df.columns),
            None,
        )
        if sort_col:
            df = df.sort_values(sort_col)
        w(df.head(25).round(4).to_string(index=False))
    else:
        w("  <none>")

    return out.getvalue()
