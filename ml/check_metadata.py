"""
Check the trained models' *_metadata.json files against what the website's pipeline reads — no weights needed.

    python ml/check_metadata.py /content/drive/MyDrive/dentalcaries/pipeline_stage1_to_4_corrected

Exit code 0 = every required key is present with a usable type; 1 = something the backend would need is missing.
"""
import json
import os
import sys

TTA_SUPPORTED = {"none", "hflip", "zoom", "hflip+zoom"}

CONTRACT = {   # metadata file -> {key: check}
    "stage1_caries_model_metadata.json": {
        "image_size": int, "threshold": float, "uncertain_margin": float, "confidence_threshold": float,
        "class_names": list},
    "stage2_fdi_detector_metadata.json": {
        "class_names": list, "input_size": int, "cache_side": int, "score_threshold": float,
        "class_confidence_threshold": float, "nms_iou": float, "max_detections": int, "assoc_min_overlap": float,
        "assoc_ambiguity_ratio": float},
    "cavity_localizer_model_metadata.json": {
        "input_size": int, "cache_side": int, "score_threshold": float, "nms_iou": float, "max_detections": int},
    "stage3_gvblack_model_metadata.json": {
        "class_names": list, "image_size": int, "context_pad": float, "confidence_threshold": float},
    "stage4_icdas_model_metadata.json": {
        "scores": list, "image_size": int, "confidence_threshold": float, "inference_window": dict},
}


def _ok_type(v, t):
    return isinstance(v, (int, float)) and not isinstance(v, bool) if t is float else isinstance(v, t)


def check(folder):
    problems, summary = [], {}
    for fname, keys in CONTRACT.items():
        p = os.path.join(folder, fname)
        if not os.path.isfile(p):
            problems.append(f"{fname}: missing")
            continue
        with open(p) as f:
            m = json.load(f)
        for k, t in keys.items():
            if k not in m:
                problems.append(f"{fname}: key '{k}' missing")
            elif not _ok_type(m[k], t):
                problems.append(f"{fname}: key '{k}' has type {type(m[k]).__name__}, expected {t.__name__}")
        if m.get("tta", "none") not in TTA_SUPPORTED:
            problems.append(f"{fname}: tta '{m['tta']}' not supported by pipeline/preprocessing.tta_views")
        if fname.startswith("stage4"):
            w = m.get("inference_window", {})
            for k in ("mult", "min_frac_w", "min_frac_h"):
                if k not in w:
                    problems.append(f"{fname}: inference_window.{k} missing")
            if len(m.get("scores", [])) != len(m.get("class_names", m.get("scores", []))):
                problems.append(f"{fname}: scores and class_names differ in length")
        summary[fname] = {k: m.get(k) for k in ("architecture", "backbone", "class_names", "scores", "threshold",
                                                 "score_threshold", "tta", "mapping_source") if k in m}
    return problems, summary


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DCAI_MODELS_DIR", "backend/models")
    probs, summ = check(folder)
    print(json.dumps(summ, indent=2))
    print("\nOK — metadata matches the website contract" if not probs else "\nPROBLEMS:\n- " + "\n- ".join(probs))
    sys.exit(1 if probs else 0)
