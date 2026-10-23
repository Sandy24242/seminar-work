"""
Full cascade on one image — notebook Section 76 `run_pipeline`, with real per-step progress callbacks.

FULL DENTAL IMAGE -> Stage 1 (healthy vs caries) -> HEALTHY: stop | UNCERTAIN: stop (unless continue_if_uncertain) | CARIES
-> Stage 2 FDI detector (UNAVAILABLE if not loaded) -> lesion localization -> lesion -> tooth association
-> for EACH lesion: Stage 3 G.V. Black + Stage 4 ICDAS.

With all five models loaded, the returned dict has exactly the keys and values of the notebook's run_pipeline (verified
by tests/test_notebook_parity.py). Website-only additions: graceful UNAVAILABLE / FAILED handling when a model is missing
or a stage raises (the notebook would stop with an exception), and `steps` (timing + status of each step).
Every value comes from a model output; nothing is filled in for a stage that did not run.
"""
import time

import numpy as np

from pipeline import lesion_localization as L
from pipeline import stage2_fdi as S2
from pipeline.stage1 import stage1_decide, stage1_probability
from pipeline.stage3_gvblack import classify_gvb, unavailable_gvb
from pipeline.stage4_icdas import classify_icdas, unavailable_icdas

STEP_ORDER = ["validation", "models", "stage1", "stage2", "localization", "association", "stage3", "stage4", "report"]
STEP_TITLES = {
    "validation": "Image validation", "models": "Model loading", "stage1": "Stage 1 — Healthy vs Caries",
    "stage2": "Stage 2 — FDI tooth detection", "localization": "Lesion localization",
    "association": "Tooth–lesion association", "stage3": "Stage 3 — G.V. Black classification",
    "stage4": "Stage 4 — ICDAS classification", "report": "Report generation",
}


def _noop(*_a, **_k):
    pass


def run_pipeline(registry, rgb, score_threshold=None, s1_confidence_threshold=None, continue_if_uncertain=False,
                 keep_crops=False, progress=None):
    """progress(step, state, message): state in {'running', 'done', 'skipped', 'unavailable', 'failed', 'uncertain'}."""
    cb = progress or _noop
    steps = {}

    def step(name, state, msg, t0=None):
        steps[name] = {"state": state, "message": msg, "seconds": round(time.time() - t0, 2) if t0 else None}
        cb(name, state, msg)

    res = {"stage1": None, "stage2": {"status": "NOT RUN"}, "localization": {"status": "NOT RUN"},
           "stage3": {"status": "NOT RUN"}, "stage4": {"status": "NOT RUN"}, "teeth": [], "lesions": []}

    def skip_rest(names, reason, state="skipped"):
        for k in names:
            if k in res and isinstance(res[k], dict) and res[k].get("status") == "NOT RUN":
                res[k]["reason"] = reason
            step(k, state, reason)

    # ---- Stage 1 ----
    if not registry.has("stage1"):
        reason = "Stage 1 model unavailable — " + registry.unavailable.get("stage1", "not loaded")
        res.update(overall="NOT ANALYSED — Stage 1 model unavailable", error=reason)
        step("stage1", "unavailable", reason)
        skip_rest(["stage2", "localization", "association", "stage3", "stage4"], "Stage 1 did not run")
        res["steps"] = steps
        return res
    cb("stage1", "running", "classifying the full image")
    t0 = time.time()
    try:
        m1 = registry.meta["stage1"]
        d = stage1_decide(stage1_probability(registry.models["stage1"], m1, rgb), m1, s1_confidence_threshold)
    except Exception as e:
        reason = f"Stage 1 inference failed ({type(e).__name__}: {e})"
        res.update(overall="NOT ANALYSED — Stage 1 failed", error=reason)
        step("stage1", "failed", reason, t0)
        skip_rest(["stage2", "localization", "association", "stage3", "stage4"], "Stage 1 failed")
        res["steps"] = steps
        return res
    res["stage1"] = d
    step("stage1", "uncertain" if d["prediction"] == "UNCERTAIN" else "done",
         f"{d['prediction']} (confidence {d['confidence'] * 100:.1f}%)", t0)
    if d["prediction"] == "HEALTHY" or (d["prediction"] == "UNCERTAIN" and not continue_if_uncertain):
        res["overall"] = ("HEALTHY — NO CARIES DETECTED" if d["prediction"] == "HEALTHY" else
                          "UNCERTAIN — Stage 1 is not confident; retake a clear, well-lit photo or see a dentist")
        reason = f"Stage 1 = {d['prediction']} (caries-specific cascade stopped)"
        for k in ("stage2", "localization", "stage3", "stage4"):
            res[k]["reason"] = reason
        for k in ("stage2", "localization", "association", "stage3", "stage4"):
            step(k, "skipped", reason)
        res["steps"] = steps
        return res

    # ---- Stage 2: teeth + FDI numbers ----
    teeth, mf = None, None
    if registry.has("fdi"):
        cb("stage2", "running", "detecting teeth and FDI numbers")
        t0 = time.time()
        try:
            mf = registry.meta["fdi"]
            teeth = S2.detect_teeth(registry.models["fdi"], mf, rgb)
            res["teeth"] = teeth
            res["stage2"] = {"status": "RAN", "n_teeth": len(teeth), "n_number_uncertain": int(sum(t["fdi_uncertain"] for t in teeth)),
                             "score_threshold": float(mf["score_threshold"]),
                             "class_confidence_threshold": float(mf["class_confidence_threshold"])}
            step("stage2", "done", f"{len(teeth)} teeth detected ({res['stage2']['n_number_uncertain']} with an uncertain number)", t0)
        except Exception as e:
            teeth = None
            res["stage2"] = {"status": "FAILED", "reason": f"FDI detector inference failed ({type(e).__name__}: {e})"}
            step("stage2", "failed", res["stage2"]["reason"], t0)
    else:
        res["stage2"] = {"status": "UNAVAILABLE", "reason": registry.unavailable.get(
            "fdi", "Stage 2 FDI detector is not loaded (train it in Section 24, then run load_all_models())")}
        step("stage2", "unavailable", res["stage2"]["reason"])

    # ---- lesion localization (validated threshold unless the caller lowers it) ----
    if not registry.has("localizer"):
        res["localization"] = {"status": "UNAVAILABLE", "reason": registry.unavailable.get("localizer", "not loaded")}
        step("localization", "unavailable", res["localization"]["reason"])
        crops = []
        res["overall"] = "CARIES DETECTED (Stage 1) — lesion localization unavailable"
        skip_rest(["association", "stage3", "stage4"], "lesion localization unavailable — no lesion crops")
        if d["prediction"] == "UNCERTAIN":
            res["overall"] = "UNCERTAIN (Stage 1) — cascade continued on request: " + res["overall"].lower()
        res["steps"] = steps
        return res
    cb("localization", "running", "localizing lesions")
    t0 = time.time()
    ml = registry.meta["localizer"]
    try:
        validated = float(ml["score_threshold"]); used = validated if score_threshold is None else float(score_threshold)
        boxes, scores = L.localize(registry.models["localizer"], ml, rgb, used)
        crops = L.generate_cavity_crops(rgb, boxes, scores, registry.meta.get("gvb"), registry.meta.get("icdas"), ml)
    except Exception as e:
        res["localization"] = {"status": "FAILED", "reason": f"localizer inference failed ({type(e).__name__}: {e})"}
        step("localization", "failed", res["localization"]["reason"], t0)
        res["overall"] = "CARIES DETECTED (Stage 1) — lesion localization failed"
        skip_rest(["association", "stage3", "stage4"], "lesion localization failed — no lesion crops")
        res["steps"] = steps
        return res
    res["localization"] = {"status": "RAN", "n_regions": int(len(boxes)), "n_valid_crops": len(crops),
                           "score_threshold_used": used, "validated_threshold": validated}
    step("localization", "done", f"{len(boxes)} region(s) at score ≥ {used:.2f}; {len(crops)} valid lesion crop(s)", t0)

    # ---- explicit lesion -> tooth association ----
    t0 = time.time()
    if teeth is not None:
        links = S2.associate_lesions_with_teeth([c["box_norm"] for c in crops], teeth, mf["assoc_min_overlap"],
                                                mf["assoc_ambiguity_ratio"])
        if crops:
            n_ok = sum(lk["fdi_status"] == "ASSIGNED" for lk in links)
            step("association", "done", f"{n_ok} of {len(crops)} lesion(s) linked to one FDI tooth", t0)
        else:
            step("association", "skipped", "no lesion to associate")
    else:
        links = S2.unavailable_links(len(crops))
        step("association", "unavailable" if crops else "skipped",
             "Stage 2 unavailable — lesions are not linked to tooth numbers" if crops else "no lesion to associate")

    # ---- Stage 3 + Stage 4 on EACH lesion crop (same preprocessing and validation-chosen TTA as the test evaluation) ----
    n = len(crops)
    stage_out = {}
    for key, fn, unavail, title in (("gvb", classify_gvb, unavailable_gvb, "stage3"), ("icdas", classify_icdas, unavailable_icdas, "stage4")):
        if not n:
            stage_out[key] = []
            continue
        if not registry.has(key):
            reason = f"model unavailable — {registry.unavailable.get(key, 'not loaded')}"
            stage_out[key] = unavail(n, reason); res[title] = {"status": "UNAVAILABLE", "reason": reason}
            step(title, "unavailable", reason)
            continue
        cb(title, "running", f"classifying {n} lesion crop(s)")
        t0 = time.time()
        try:
            stage_out[key] = fn(registry.models[key], registry.meta[key], [c[f"{key}_crop"] for c in crops])
            res[title] = {"status": "RAN", "n_lesions": n}
            unc = sum(bool(r.get(f"{key}_uncertain")) for r in stage_out[key])
            step(title, "uncertain" if unc else "done", f"{n} lesion(s) classified" + (f", {unc} uncertain" if unc else ""), t0)
        except Exception as e:
            reason = f"inference failed ({type(e).__name__}: {e})"
            stage_out[key] = unavail(n, reason); res[title] = {"status": "FAILED", "reason": reason}
            step(title, "failed", reason, t0)

    validated = res["localization"]["validated_threshold"]
    for i, (c, lk) in enumerate(zip(crops, links)):
        res["lesions"].append({
            "lesion": i + 1, "box_xyxy_px": c["box_xyxy_px"], "box_norm": c["box_norm"], "localizer_score": c["localizer_score"],
            "below_validated_threshold": bool(c["localizer_score"] < validated), "icdas_window_px": c["icdas_window_px"],
            **lk, **stage_out["gvb"][i], **stage_out["icdas"][i]})
    if not n:
        for k in ("stage3", "stage4"):
            res[k] = {"status": "NOT RUN", "reason": "no valid lesion crop (no lesion localized above the score threshold)"}
            step(k, "skipped", res[k]["reason"])
    res["overall"] = "CARIES DETECTED" if res["lesions"] else "CARIES DETECTED — but no lesion localized above the threshold"
    if d["prediction"] == "UNCERTAIN":
        res["overall"] = "UNCERTAIN (Stage 1) — cascade continued on request: " + res["overall"].lower()
    if keep_crops:
        res["_crops"] = [{"gvb_crop": c["gvb_crop"], "icdas_crop": c["icdas_crop"]} for c in crops]
    res["steps"] = steps
    return res


def ui_status(res, key):
    """Website status label for a result section: Completed | Uncertain | Unavailable | Skipped | Failed."""
    if key == "stage1":
        s1 = res.get("stage1")
        if s1 is None:
            return "Failed" if "failed" in str(res.get("overall", "")).lower() else "Unavailable"
        return "Uncertain" if s1["prediction"] == "UNCERTAIN" else "Completed"
    st = (res.get(key) or {}).get("status", "NOT RUN")
    if st == "RAN":
        flag = {"stage3": "gvb_uncertain", "stage4": "icdas_uncertain"}.get(key)
        if key == "stage2" and res.get("stage2", {}).get("n_number_uncertain"):
            return "Uncertain"
        if flag and any(l.get(flag) for l in res.get("lesions", [])):
            return "Uncertain"
        return "Completed"
    return {"UNAVAILABLE": "Unavailable", "FAILED": "Failed"}.get(st, "Skipped")


def strip_private(res):
    """JSON-safe copy (drops crops / numpy arrays kept for display)."""
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items() if not str(k).startswith("_")}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    return conv(res)
