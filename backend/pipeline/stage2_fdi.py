"""
Stage 2 — FDI tooth detection + numbering, and the explicit lesion -> tooth association.
Ported from notebook Section 6.1 (`associate_lesions_with_teeth`) and Section 6.6 (`detect_teeth`).
The detector's boxes are TEETH, never lesions; lesions come from the separate localizer.
"""
from collections import defaultdict

import numpy as np

from pipeline.preprocessing import decode_fdi_detections, forward, loc_prepare, overlap_fraction, to_px


def detect_teeth(model, meta, rgb, score_threshold=None):
    """Stage 2 on a full RGB image -> list of teeth sorted left to right, each with box, detection score, FDI number and its
    probability. Number probability < class_confidence_threshold -> number uncertain; the same number predicted for two
    teeth -> the less confident one is uncertain too (each FDI number exists once per mouth)."""
    thr = meta["score_threshold"] if score_threshold is None else float(score_threshold)
    raw = forward(model, loc_prepare(rgb, meta["cache_side"], meta["input_size"]))
    boxes, obj, probs = decode_fdi_detections(raw, thr, meta["nms_iou"], meta["max_detections"])
    H, W = rgb.shape[:2]; names = meta["class_names"]; cthr = meta["class_confidence_threshold"]; teeth = []
    for b, s, p in zip(boxes, obj, probs):
        k = int(np.argmax(p)); top = np.argsort(-p)[:3]
        teeth.append({"box_norm": [float(v) for v in b], "box_xyxy_px": list(to_px(b, W, H)), "detection_score": float(s),
                      "fdi": names[k], "fdi_confidence": float(p[k]), "fdi_uncertain": bool(p[k] < cthr),
                      "fdi_unreliable_class": names[k] in meta.get("unreliable_classes", []),
                      "fdi_top3": {names[j]: float(p[j]) for j in top},
                      "fdi_note": f"number probability {p[k]:.0%} < {cthr:.0%}" if p[k] < cthr else None})
    by_num = defaultdict(list)
    for i, t in enumerate(teeth): by_num[t["fdi"]].append(i)
    for num, idx in by_num.items():
        best = max(idx, key=lambda i: teeth[i]["fdi_confidence"])
        for i in idx:
            if i != best:
                teeth[i]["fdi_uncertain"] = True; teeth[i]["fdi_note"] = f"FDI {num} is also predicted for a more confident tooth"
    teeth.sort(key=lambda t: (t["box_norm"][0] + t["box_norm"][2]) / 2)
    for i, t in enumerate(teeth, 1):
        t["tooth_id"] = i
        t["fdi_display"] = t["fdi"] if not t["fdi_uncertain"] else f"Unknown / uncertain (best guess {t['fdi']})"
    return teeth


def associate_lesions_with_teeth(lesion_boxes, teeth, min_overlap, ambiguity_ratio):
    """Explicit lesion -> tooth link (lesion boxes and tooth boxes come from two different models).
    The tooth covering the largest share of the lesion wins if it covers >= min_overlap. Status:
    ASSIGNED | AMBIGUOUS (a 2nd tooth covers >= ambiguity_ratio x as much) | TOOTH_NUMBER_UNCERTAIN | NO_TOOTH.
    Only ASSIGNED gives an FDI number; everything else is reported as 'Unknown / uncertain' (best guesses kept separately)."""
    tb = np.array([t["box_norm"] for t in teeth], np.float32).reshape(-1, 4); out = []
    for b in lesion_boxes:
        cov = overlap_fraction(b, tb) if len(tb) else np.zeros(0, np.float32)
        if not len(cov) or cov.max() < min_overlap:
            out.append({"fdi_tooth": None, "fdi_display": "Unknown / uncertain", "fdi_status": "NO_TOOTH", "tooth_id": None,
                        "fdi_confidence": None, "fdi_overlap": float(cov.max()) if len(cov) else 0.0, "fdi_also_overlaps": [],
                        "fdi_note": f"no detected tooth covers >= {min_overlap:.0%} of the lesion"})
            continue
        order = np.argsort(-cov); j = int(order[0]); t = teeth[j]
        alts = [teeth[k] for k in order[1:] if cov[k] >= max(min_overlap, ambiguity_ratio * cov[j])]
        rec = {"tooth_id": t.get("tooth_id"), "fdi_confidence": float(t["fdi_confidence"]), "fdi_overlap": float(cov[j]),
               "fdi_best_guess": t["fdi"], "fdi_also_overlaps": [a["fdi"] for a in alts]}
        if t["fdi_uncertain"]:
            rec.update(fdi_tooth=None, fdi_status="TOOTH_NUMBER_UNCERTAIN", fdi_display=f"Unknown / uncertain (best guess {t['fdi']})",
                       fdi_note="the covering tooth was detected but its FDI number is uncertain")
        elif alts:
            rec.update(fdi_tooth=None, fdi_status="AMBIGUOUS",
                       fdi_display="Uncertain: " + " or ".join([t["fdi"]] + [a["fdi"] for a in alts]),
                       fdi_note="the lesion lies across more than one detected tooth")
        else:
            rec.update(fdi_tooth=t["fdi"], fdi_status="ASSIGNED", fdi_display=t["fdi"], fdi_note=None)
        out.append(rec)
    return out


def unavailable_links(n, reason="Stage 2 unavailable"):
    """Per-lesion placeholder used by the notebook when the FDI detector is not loaded (no number is invented)."""
    return [{"fdi_tooth": None, "fdi_display": "Unavailable", "fdi_status": "STAGE2_UNAVAILABLE", "tooth_id": None,
             "fdi_confidence": None, "fdi_overlap": None, "fdi_also_overlaps": [], "fdi_note": reason} for _ in range(n)]
