"""
Stage 4 — CNN 3, ICDAS score of each lesion (notebook Section 76, per-lesion block).

Class coverage comes from metadata["scores"]. The trained model covers ICDAS 1–6 only (icdascl.zip has no ICDAS 0
folder): it never outputs ICDAS 0 and cannot rule out caries — that is Stage 1's job.
"""
import numpy as np

from pipeline.preprocessing import tta_forward

ICDAS_BASIS = "model prediction from a photograph — not clinically confirmed"


def classify_icdas(model, meta, crops):
    """crops: list of uint8 (S,S,3) tooth-scale windows (from generate_cavity_crops) -> one result dict per crop."""
    if not len(crops):
        return []
    p = tta_forward(model, np.stack(crops), meta.get("tta", "none"))
    scores, thr = meta["scores"], meta["confidence_threshold"]
    out = []
    for row in p:
        k = int(np.argmax(row))
        out.append({"icdas": int(scores[k]), "icdas_confidence": float(row[k]), "icdas_uncertain": bool(row[k] < thr),
                    "icdas_unreliable_class": int(scores[k]) in meta.get("unreliable_scores", []),
                    "icdas_probabilities": {f"ICDAS {s}": float(v) for s, v in zip(scores, row)},
                    "icdas_basis": ICDAS_BASIS})
    return out


def unavailable_icdas(n, reason):
    return [{"icdas": None, "icdas_confidence": None, "icdas_uncertain": None, "icdas_unreliable_class": False,
             "icdas_probabilities": {}, "icdas_basis": ICDAS_BASIS, "icdas_note": reason} for _ in range(n)]
