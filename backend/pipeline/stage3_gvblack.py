"""
Stage 3 — CNN 2, G.V. Black classification of each lesion crop (notebook Section 76, per-lesion block).

The output index is mapped to a class ONLY through metadata["class_names"] (e.g. index 0 -> "GVB_Class_I");
an index is never treated as a clinical class number.
"""
import numpy as np

from pipeline.preprocessing import tta_forward


def gvb_short(name):
    """'GVB_Class_II' -> 'Class II'."""
    return "Class " + str(name).split("_")[-1]


def classify_gvb(model, meta, crops):
    """crops: list of uint8 (S,S,3) context crops (from generate_cavity_crops) -> one result dict per crop."""
    if not len(crops):
        return []
    p = tta_forward(model, np.stack(crops), meta.get("tta", "none"))
    names, thr = meta["class_names"], meta["confidence_threshold"]
    out = []
    for row in p:
        k = int(np.argmax(row))
        out.append({"gvb_class": names[k], "gvb_display": gvb_short(names[k]), "gvb_confidence": float(row[k]),
                    "gvb_uncertain": bool(row[k] < thr),
                    "gvb_probabilities": {gvb_short(n): float(v) for n, v in zip(names, row)}})
    return out


def unavailable_gvb(n, reason):
    return [{"gvb_class": None, "gvb_display": "Unavailable", "gvb_confidence": None, "gvb_uncertain": None,
             "gvb_probabilities": {}, "gvb_note": reason} for _ in range(n)]
