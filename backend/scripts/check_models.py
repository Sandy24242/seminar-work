"""
Check that the trained models are present and loadable (run before launching the website).

    python scripts/check_models.py                 # uses config.resolve_models_dir()
    python scripts/check_models.py /path/to/folder

Exit code 0 = all required models loaded, 1 = something required is missing.
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from pipeline.model_loader import ModelRegistry  # noqa: E402


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else config.resolve_models_dir()
    print(f"Model folder: {folder}")
    if not os.path.isdir(folder):
        print("  !! folder does not exist")
        return 1
    for k in ("stage1", "fdi", "localizer", "gvb", "icdas"):
        for f in (config.MODEL_FILES[k], config.META_FILES[k]):
            p = os.path.join(folder, f)
            print(f"  {'OK ' if os.path.isfile(p) else 'MISSING'}  {f}" + (f"  ({os.path.getsize(p) / 1e6:.1f} MB)" if os.path.isfile(p) else ""))
    print("\nLoading models (this can take 1–2 minutes)…")
    reg = ModelRegistry(folder).load_all(progress=lambda k: print(f"  loading {k} …", flush=True))
    print(f"\nTensorFlow {reg.versions.get('tensorflow')} | Keras {reg.versions.get('keras')}")
    for r in reg.summary():
        print(f"  [{r['Status']:11}] {r['Model']:42} {r['Details']}")
    if reg.missing_required:
        print("\nRESULT: required model(s) unavailable:", ", ".join(reg.missing_required))
        return 1
    print("\nRESULT: all required models loaded" + ("" if reg.has("fdi") else " (Stage 2 FDI detector unavailable — optional)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
