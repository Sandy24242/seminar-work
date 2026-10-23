"""
Real-weights smoke test: loads the TRAINED models (never stand-ins) and runs the full cascade on real photographs.

    python scripts/smoke_test_real_models.py --models <folder with the 5 .keras + metadata> --images <photo or folder>

Refuses to run if the folder contains test stand-in metadata. Prints the per-image result and exits 1 if a required
model fails to load or inference raises. It reports what the models output; it does not judge clinical correctness.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from pipeline.image_validation import validate_image_bytes  # noqa: E402
from pipeline.model_loader import ModelRegistry  # noqa: E402
from pipeline.runner import run_pipeline, strip_private  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get("DCAI_MODELS_DIR") or config.resolve_models_dir())
    ap.add_argument("--images", required=True)
    a = ap.parse_args()

    reg = ModelRegistry(a.models).load_all()
    if any(str(m.get("model", "")).startswith("TEST STAND-IN") for m in reg.meta.values()):
        sys.exit("refusing: this folder contains test stand-in models, not trained weights")
    print("versions:", reg.versions)
    for k in ("stage1", "fdi", "localizer", "gvb", "icdas"):
        print(f"  {k:9s} {'LOADED' if reg.has(k) else 'UNAVAILABLE: ' + reg.unavailable.get(k, '')}"
              + (f" ({reg.load_seconds[k]} s)" if k in reg.load_seconds else ""))
    if reg.missing_required:
        print("required models missing:", reg.missing_required)
        sys.exit(1)

    paths = [a.images] if os.path.isfile(a.images) else sorted(
        os.path.join(a.images, f) for f in os.listdir(a.images) if f.lower().endswith(config.ALLOWED_UPLOAD_EXT))
    failed = 0
    for p in paths:
        with open(p, "rb") as f:
            rgb, info, err = validate_image_bytes(f.read(), p)
        if err:
            print(f"\n{p}: invalid image — {err}")
            continue
        res = strip_private(run_pipeline(reg, rgb))
        failed += any(s["state"] == "failed" for s in res.get("steps", {}).values())
        print(f"\n{os.path.basename(p)} -> {res['overall']}")
        print(json.dumps({"stage1": res["stage1"], "n_teeth": len(res["teeth"]),
                          "lesions": [{k: l.get(k) for k in ("lesion", "localizer_score", "fdi_tooth", "gvb_display",
                                                              "gvb_confidence", "icdas", "icdas_confidence")}
                                      for l in res["lesions"]]}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
