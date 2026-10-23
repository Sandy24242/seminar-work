"""
Validate the REAL trained models and run genuine predictions with them.

    python scripts/validate_real_models.py                              # contract checks only
    python scripts/validate_real_models.py --images D:\\photos           # + real cascade on every JPG/PNG in the folder
    python scripts/validate_real_models.py --models-dir X --images a.jpg b.png --out report.json

1. Loads every model with the same ModelRegistry the website uses (config.resolve_models_dir() unless --models-dir).
2. Refuses test stand-in models (metadata "model": "TEST STAND-IN", written only by tests/standin_models.py).
3. Checks each model's input/output shapes against its own *_metadata.json (image size, grid, class list length).
4. Optionally runs the real cascade (pipeline.runner.run_pipeline) on the given images and prints/saves the outputs.

Nothing here computes accuracy: the images given are not assumed to be labelled or held out. For held-out metrics see
scripts/recorded_metrics.py (numbers the notebook recorded) and ml/README.md (re-evaluating with TRAIN_MODE="load").

Exit code 0 = every loaded model passed its checks and all required models loaded; 1 otherwise.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from pipeline.model_loader import ModelRegistry  # noqa: E402

STANDIN_MARKER = "TEST STAND-IN"
IMG_EXT = (".jpg", ".jpeg", ".png")


def _shape(s):
    s = s[0] if isinstance(s, list) else s
    return tuple(s)


def check_contract(stage, model, meta):
    """List of (check, ok, detail) comparing the loaded model with its metadata."""
    inp, out = _shape(model.input_shape), _shape(model.output_shape)
    checks = [("input is (None, H, W, 3)", len(inp) == 4 and inp[-1] == 3, f"input {inp}")]

    def eq(name, a, b):
        checks.append((name, a == b, f"{a!r} vs {b!r}"))

    names = meta.get("class_names")
    if stage == "stage1":
        eq("input side == metadata image_size", inp[1], meta.get("image_size"))
        eq("single sigmoid output", out[-1], 1)
        mapping = {str(k): v for k, v in (meta.get("class_mapping") or {}).items()}
        eq("class_mapping 0/1", mapping, {"0": "Healthy", "1": "Caries"})
        thr = meta.get("threshold")
        checks.append(("threshold in (0, 1)", isinstance(thr, (int, float)) and 0 < thr < 1, f"{thr!r}"))
    elif stage == "localizer":
        eq("input side == metadata input_size", inp[1], meta.get("input_size"))
        eq("grid == input_size / stride", out[1], (meta.get("input_size") or 0) // (meta.get("stride") or 1))
        eq("5 output channels (objectness + l,t,r,b)", out[-1], 5)
    elif stage == "fdi":
        eq("input side == metadata input_size", inp[1], meta.get("input_size"))
        eq("output channels == 5 + len(class_names)", out[-1], 5 + len(names or []))
        if "num_classes" in meta:
            eq("num_classes == len(class_names)", meta["num_classes"], len(names or []))
    elif stage in ("gvb", "icdas"):
        if meta.get("image_size"):
            eq("input side == metadata image_size", inp[1], meta["image_size"])
        eq("softmax units == len(class_names)", out[-1], len(names or []))
        if stage == "icdas" and meta.get("scores") is not None:
            eq("len(scores) == len(class_names)", len(meta["scores"]), len(names or []))
    return checks, {"input_shape": list(inp), "output_shape": list(out), "class_names": names}


def list_images(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for dp, _dn, fn in os.walk(p):
                out += [os.path.join(dp, f) for f in sorted(fn) if f.lower().endswith(IMG_EXT)]
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f"  !! not found: {p}")
    return out


def summarise(res):
    s1 = res.get("stage1") or {}
    return {
        "overall": res.get("overall"),
        "stage1": {k: s1.get(k) for k in ("prediction", "leaning", "confidence", "p_caries", "threshold", "reason")} if s1 else None,
        "stage2": res.get("stage2", {}).get("status"),
        "localization": res.get("localization", {}).get("status"),
        "n_teeth": len(res.get("teeth") or []),
        "lesions": [{k: l.get(k) for k in ("box_xyxy_px", "localizer_score", "fdi_tooth", "fdi_status", "gvb_class",
                                            "gvb_confidence", "icdas", "icdas_confidence")} for l in res.get("lesions") or []],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--images", nargs="*", default=[], help="image files and/or folders")
    ap.add_argument("--continue-if-uncertain", action="store_true")
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    ap.add_argument("--allow-standins", action="store_true", help=argparse.SUPPRESS)   # tests only
    a = ap.parse_args(argv)

    folder = a.models_dir or config.resolve_models_dir()
    print(f"Model folder: {folder}")
    reg = ModelRegistry(folder).load_all(progress=lambda k: print(f"  loading {k} ...", flush=True))
    report = {"models_dir_name": os.path.basename(os.path.normpath(folder)), "versions": reg.versions,
              "models": {}, "predictions": [], "standin_models_used": False}
    ok = True
    for k in ("stage1", "fdi", "localizer", "gvb", "icdas"):
        entry = {"file": config.MODEL_FILES[k], "required": k in config.REQUIRED_MODELS, "loaded": reg.has(k)}
        if not reg.has(k):
            entry["reason"] = reg.unavailable.get(k, "not loaded")
            print(f"\n[{k}] UNAVAILABLE - {entry['reason']}")
            ok &= k not in config.REQUIRED_MODELS
            report["models"][k] = entry
            continue
        meta = reg.meta[k]
        standin = meta.get("model") == STANDIN_MARKER
        entry["test_standin"] = standin
        if standin:
            report["standin_models_used"] = True
            print(f"\n[{k}] !! this is a TEST STAND-IN model (untrained), not the trained model")
            if not a.allow_standins:
                ok = False
        checks, info = check_contract(k, reg.models[k], meta)
        entry.update(info, checks=[{"check": c, "ok": bool(o), "detail": d} for c, o, d in checks],
                     split_fingerprint=meta.get("split_fingerprint"), architecture=meta.get("backbone") or meta.get("architecture"))
        print(f"\n[{k}] {config.MODEL_TITLES[k]} - loaded in {reg.load_seconds.get(k)} s")
        print(f"    input {info['input_shape']}  output {info['output_shape']}")
        if info["class_names"]:
            print(f"    classes ({len(info['class_names'])}): {info['class_names']}")
        for c, o, d in checks:
            print(f"    {'PASS' if o else 'FAIL'}  {c}  ({d})")
            ok &= bool(o)
        report["models"][k] = entry
    if reg.missing_required:
        ok = False

    images = list_images(a.images)
    if images and not reg.has("stage1"):
        print("\nStage 1 is not loaded, so no prediction can be run.")
        ok = False
    elif images:
        from pipeline.image_validation import validate_image_bytes
        from pipeline.runner import run_pipeline
        label = "TEST STAND-IN outputs (meaningless)" if report["standin_models_used"] else "trained-model outputs"
        print(f"\nRunning the cascade on {len(images)} image(s) - {label}")
        for p in images:
            with open(p, "rb") as f:
                rgb, info, err = validate_image_bytes(f.read(), os.path.basename(p))
            row = {"image": os.path.basename(p)}
            if err:
                row["error"] = err
            else:
                t0 = time.time()
                row.update(summarise(run_pipeline(reg, rgb, continue_if_uncertain=a.continue_if_uncertain)),
                           seconds=round(time.time() - t0, 2), width=info["width"], height=info["height"])
            report["predictions"].append(row)
            s1 = row.get("stage1") or {}
            print(f"  {row['image']:40} " + (f"ERROR {row['error']}" if "error" in row else
                  f"{s1.get('prediction')} p_caries={s1.get('p_caries'):.3f} lesions={len(row['lesions'])} | {row['overall']}"))
            for les in row.get("lesions", []):
                print(f"      lesion box={les['box_xyxy_px']} tooth={les['fdi_tooth']} gvb={les['gvb_class']} "
                      f"({les['gvb_confidence']}) icdas={les['icdas']} ({les['icdas_confidence']})")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport written to {a.out}")
    print("\nRESULT:", "all checks passed" if ok else "problems found (see FAIL / UNAVAILABLE above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
