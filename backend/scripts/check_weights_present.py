"""
Fail fast if the model folder holds Git LFS pointer files instead of the trained weights (used by the Dockerfile).

    python scripts/check_weights_present.py [models_dir]      # default: $DCAI_MODELS_DIR
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

LFS_HEADER = b"version https://git-lfs"


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DCAI_MODELS_DIR", "")
    expected = list(config.MODEL_FILES.values())
    missing = [f for f in expected if not os.path.isfile(os.path.join(folder, f))]
    pointers = []
    for f in sorted(glob.glob(os.path.join(folder, "*.keras"))):
        with open(f, "rb") as fh:
            if fh.read(len(LFS_HEADER)) == LFS_HEADER:
                pointers.append(os.path.basename(f))
    if missing or pointers:
        sys.exit(f"{folder}: missing {missing or 'none'}; Git LFS pointer files instead of weights: {pointers or 'none'}. "
                 "Run `git lfs pull` before `docker build`, or build with --build-arg REQUIRE_MODELS=0 and mount the models.")
    print("weights present:", ", ".join(f"{f} {os.path.getsize(os.path.join(folder, f)) // 2**20} MB" for f in expected))


if __name__ == "__main__":
    main()
