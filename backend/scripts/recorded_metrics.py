"""
Print the held-out test metrics that the TRAINING NOTEBOOK recorded in each *_metadata.json.

    python scripts/recorded_metrics.py                     # uses config.resolve_models_dir()
    python scripts/recorded_metrics.py /path/to/models --md recorded_metrics.md

These numbers were computed by the notebook on its own leakage-controlled test split (Section 10: group split, no
patient IDs available). This script only reads them back; it does not re-measure anything. To re-measure, re-run the
notebook with TRAIN_MODE = "load" for every stage (see ml/README.md).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

METRIC_KEYS = {
    "stage1": ["test_metrics"],
    "fdi": ["val_metrics", "test_metrics"],
    "localizer": ["test_metrics"],
    "gvb": ["test_metrics_gt_crops", "test_metrics"],
    "icdas": ["test_metrics", "test_ordinal", "lesion_window_test_metrics"],
}
SPLIT_KEYS = ("split_counts", "split_images", "split_fingerprint")


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def flatten(d, prefix=""):
    """Scalars as 'a.b.c = v'; per-class tables and confusion matrices are returned separately."""
    scalars, tables = [], []
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            tables.append((name, v))
        elif isinstance(v, dict):
            s, t = flatten(v, name + ".")
            scalars += s; tables += t
        elif isinstance(v, list) and v and isinstance(v[0], list):
            tables.append((name, v))
        else:
            scalars.append((name, v))
    return scalars, tables


def render_stage(stage, meta):
    L = [f"## {config.MODEL_TITLES[stage]}  (`{config.META_FILES[stage]}`)", ""]
    for k in SPLIT_KEYS:
        if k in meta:
            L.append(f"- {k}: {meta[k]}")
    found = False
    for key in METRIC_KEYS[stage]:
        m = meta.get(key)
        if not isinstance(m, dict) or not m:
            continue
        found = True
        L += ["", f"### {key}", ""]
        scalars, tables = flatten(m)
        L += [f"- {n} = {_fmt(v)}" for n, v in scalars]
        for name, t in tables:
            L += ["", f"**{name}**", ""]
            if isinstance(t, dict):
                cols = sorted({c for row in t.values() for c in row})
                L += ["| class | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
                L += [f"| {r} | " + " | ".join(_fmt(t[r].get(c, "")) for c in cols) + " |" for r in t]
            else:
                L += ["```"] + [" ".join(f"{x:>5}" for x in row) for row in t] + ["```"]
    if not found:
        L.append("- no test metrics recorded in this metadata file")
    return L + [""]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models_dir", nargs="?", default=None)
    ap.add_argument("--md", default=None, help="also write the report to this Markdown file")
    a = ap.parse_args(argv)
    folder = a.models_dir or config.resolve_models_dir()
    L = ["# Recorded test metrics (from the training notebook)", "",
         "Read back from the `*_metadata.json` files written by the notebook. NOT re-measured by this script.", ""]
    missing = []
    for stage in ("stage1", "fdi", "localizer", "gvb", "icdas"):
        p = os.path.join(folder, config.META_FILES[stage])
        if not os.path.isfile(p):
            missing.append(config.META_FILES[stage])
            continue
        with open(p) as f:
            meta = json.load(f)
        if meta.get("model") == "TEST STAND-IN":
            L += [f"## {config.MODEL_TITLES[stage]}", "", "- TEST STAND-IN metadata: no real metrics", ""]
            continue
        L += render_stage(stage, meta)
    if missing:
        L += ["## Missing metadata files", ""] + [f"- {m}" for m in missing]
    text = "\n".join(L) + "\n"
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text)
    return 1 if len(missing) == 5 else 0


if __name__ == "__main__":
    sys.exit(main())
