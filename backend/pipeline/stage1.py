"""Stage 1 — CNN 1, Healthy vs Caries (notebook Section 6.6 `stage1_probability`, `stage1_decide`)."""
from pipeline.preprocessing import forward, s1_prepare


def stage1_probability(model, meta, rgb):
    """Sigmoid output = P(caries) (metadata: class_mapping {0: Healthy, 1: Caries})."""
    return float(forward(model, s1_prepare(rgb, meta["image_size"])).ravel()[0])


def stage1_decide(p, meta, confidence_threshold=None):
    thr = meta["threshold"]; cthr = meta["confidence_threshold"] if confidence_threshold is None else confidence_threshold
    raw = "CARIES" if p >= thr else "HEALTHY"
    conf = p if raw == "CARIES" else 1 - p
    reasons = []
    if abs(p - thr) < meta["uncertain_margin"]:
        reasons.append(f"P(caries) {p:.3f} is within ±{meta['uncertain_margin']:.2f} of the validation threshold {thr:.3f}")
    if conf < cthr:
        reasons.append(f"confidence {conf * 100:.1f}% < {cthr * 100:.0f}%")
    return {"prediction": "UNCERTAIN" if reasons else raw, "leaning": raw, "confidence": float(conf), "p_caries": float(p),
            "threshold": float(thr), "reason": "; ".join(reasons) or None}
