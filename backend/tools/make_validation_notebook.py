"""Writes colab/Validate_Real_Models_and_Data.ipynb.   Run: python tools/make_validation_notebook.py

A Colab notebook that, with the user's own Drive mounted, (1) loads the REAL trained models and checks them against their
metadata, (2) runs genuine predictions on real photographs, also through the FastAPI backend, (3) audits the dataset
ZIPs read-only, (4) prints the test metrics the training notebook recorded, and (5) saves the text reports (no images,
no weights) to Drive so they can be shared. It never trains and never writes into the dataset or model folders.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INTRO = """# Dental Caries AI — validate the real trained models and the datasets

Runs entirely on **your** Google Drive: nothing is uploaded anywhere. Nothing is trained, and the dataset ZIPs and the model
folder are only read.

**Before you start:** `Runtime → Change runtime type → T4 GPU` (CPU works too, slower). Then run the cells top to bottom.

Outputs (text only) are written to `MyDrive/dentalcaries/dcai_validation_reports/`:
`check_models.txt`, `model_validation.json`, `api_e2e.json`, `dataset_audit.md/json`, `recorded_metrics.md`.

**What this does NOT do:** it does not compute new accuracy figures. The sample photos come from the training archives and
may be in the training split. Held-out metrics are (a) read back from the metadata the training notebook wrote, and
(b) re-measured by re-running the training notebook with `TRAIN_MODE = "load"` (last section)."""

CELLS = [
    ("Mount Google Drive", '''import os, subprocess
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "No GPU (CPU runtime)")
from google.colab import drive
drive.mount("/content/drive")
DRIVE_PROJECT = "/content/drive/MyDrive/dentalcaries"              # edit if your folder lives elsewhere
MODELS_DRIVE  = f"{DRIVE_PROJECT}/pipeline_stage1_to_4_corrected"
REPORTS = f"{DRIVE_PROJECT}/dcai_validation_reports"
os.makedirs(REPORTS, exist_ok=True)
print("project folder:", os.path.isdir(DRIVE_PROJECT), "| model folder:", os.path.isdir(MODELS_DRIVE))'''),

    ("Get the code from GitHub and install the API packages", '''import subprocess, shutil, os
REPO_URL = "https://github.com/Sandy24242/seminar-work.git"
REPO_BRANCH = "claude/validate-real-models"     # branch with the validation scripts; "" = repository default branch
shutil.rmtree("/content/seminar-work", ignore_errors=True)
args = ["git", "clone", "--depth", "1"] + (["-b", REPO_BRANCH] if REPO_BRANCH else []) + [REPO_URL, "/content/seminar-work"]
if subprocess.run(args).returncode != 0:
    print("branch not found - cloning the default branch"); subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/seminar-work"], check=True)
APP_DIR = "/content/seminar-work/backend"
assert os.path.exists(f"{APP_DIR}/scripts/validate_real_models.py"), "this branch has no validation scripts - set REPO_BRANCH"
!pip install -q -r {APP_DIR}/requirements-colab.txt "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.9"
%cd $APP_DIR'''),

    ("Copy the real models to local disk and check that they load", '''import glob, shutil
LOCAL_MODELS = f"{APP_DIR}/models"
os.makedirs(LOCAL_MODELS, exist_ok=True)
for f in glob.glob(f"{MODELS_DRIVE}/*.keras") + glob.glob(f"{MODELS_DRIVE}/*metadata.json"):
    dst = f"{LOCAL_MODELS}/{os.path.basename(f)}"
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(f)):
        shutil.copy(f, dst); print("copied", os.path.basename(f))
os.environ["DCAI_MODELS_DIR"] = LOCAL_MODELS
!python scripts/check_models.py "$DCAI_MODELS_DIR" 2>&1 | tee {REPORTS}/check_models.txt'''),

    ("Pick real sample photos (healthy + caries) from the dataset ZIPs", '''import zipfile, random
SAMPLES = "/content/dcai_samples"; shutil.rmtree(SAMPLES, ignore_errors=True); os.makedirs(SAMPLES)
N_PER_ARCHIVE = 4                                # small, just to exercise the real models
random.seed(0)
for name in ["healthy.zip", "caries.zip"]:
    p = f"{DRIVE_PROJECT}/{name}"
    if not os.path.isfile(p):
        print("not found:", p); continue
    with zipfile.ZipFile(p) as z:
        imgs = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")) and "__MACOSX" not in n]
        for n in random.sample(imgs, min(N_PER_ARCHIVE, len(imgs))):
            out = f"{SAMPLES}/{name[:-4]}__{os.path.basename(n)}"
            open(out, "wb").write(z.read(n))
shutil.copy(f"{APP_DIR}/assets/clinical_caries_example.jpg", f"{SAMPLES}/notebook_example__clinical_caries_example.jpg")
print(sorted(os.listdir(SAMPLES)))
print("NOTE: these photos come from the training archives; they may be in the training split. Not an evaluation.")'''),

    ("Validate the models against their metadata and run genuine predictions", '''!python scripts/validate_real_models.py --models-dir "$DCAI_MODELS_DIR" --images /content/dcai_samples --out {REPORTS}/model_validation.json'''),

    ("End-to-end through the real FastAPI backend", '''import subprocess, time, json, urllib.request, requests
subprocess.run("pkill -f 'uvicorn api.main:app' || true", shell=True)
api_log = open("/content/api.log", "w")
api = subprocess.Popen(["python", "-m", "uvicorn", "api.main:app", "--port", "8000"], cwd=APP_DIR,
                       env={**os.environ, "DCAI_MODELS_DIR": LOCAL_MODELS}, stdout=api_log, stderr=subprocess.STDOUT)
for _ in range(180):
    try:
        if requests.get("http://127.0.0.1:8000/health", timeout=2).ok: break
    except Exception: time.sleep(1)
out = {"health": requests.get("http://127.0.0.1:8000/health").json(),
       "models_status": requests.get("http://127.0.0.1:8000/models/status").json(), "predict": []}
print("ready:", out["models_status"]["ready"], "| missing:", out["models_status"]["missing_required"])
for f in sorted(os.listdir("/content/dcai_samples")):
    with open(f"/content/dcai_samples/{f}", "rb") as fh:
        r = requests.post("http://127.0.0.1:8000/predict", files={"file": (f, fh)})
    b = r.json(); b.pop("annotated_jpeg_base64", None)
    out["predict"].append({"image": f, "http": r.status_code, "overall": b.get("overall"), "stage1": b.get("stage1"),
                           "stage_status": b.get("stage_status"),
                           "lesions": [{k: l.get(k) for k in ("box_xyxy_px", "localizer_score", "fdi_tooth", "gvb_class",
                                        "gvb_confidence", "icdas", "icdas_confidence")} for l in b.get("lesions", [])]})
    print(f"{f:60} HTTP {r.status_code}  {b.get('overall')}")
bad = requests.post("http://127.0.0.1:8000/predict", files={"file": ("notes.txt", b"hello")})
out["invalid_upload"] = {"http": bad.status_code, "detail": bad.json().get("detail")}
print("invalid upload ->", bad.status_code, bad.json().get("detail"))
json.dump(out, open(f"{REPORTS}/api_e2e.json", "w"), indent=2, default=str)
api.terminate()'''),

    ("Audit the dataset ZIPs (read-only)", '''ZIPS = [f"{DRIVE_PROJECT}/{n}" for n in ["healthy.zip", "caries.zip", "lowoccyolo (4).zip", "icdascl.zip",
                                           "caries_FDI_dataset_final.zip"] if os.path.isfile(f"{DRIVE_PROJECT}/{n}")]
print(ZIPS)
# --near-dup compares every image with every other (dHash), as the notebook's split does; it adds a few minutes.
!python scripts/audit_datasets.py {" ".join(f'"{z}"' for z in ZIPS)} --near-dup --out {REPORTS}/dataset_audit.json --md {REPORTS}/dataset_audit.md
print(open(f"{REPORTS}/dataset_audit.md").read())'''),

    ("Test metrics recorded by the training notebook", '''!python scripts/recorded_metrics.py "$DCAI_MODELS_DIR" --md {REPORTS}/recorded_metrics.md'''),

    ("Re-measure the held-out metrics (optional, long)", '''# The notebook does not save its per-image split, so held-out metrics can only be re-measured by re-running its own
# leakage-controlled split (Section 10) and evaluation sections with the saved models:
#   1. Open notebooks/Dental_Caries_Pipeline_Stage1_to_Stage4_with_FDI_Stage2.ipynb in Colab (GPU runtime).
#   2. In Section 3 set: TRAIN_MODE = {"stage1": "load", "localizer": "load", "gvb": "load", "icdas": "load", "fdi": "load"}
#   3. Run all. With "load" nothing is retrained. Section 6.6 `decide()` compares each model's saved split_fingerprint
#      with the regenerated split; if they differ it only PRINTS "WARNING: saved <stage> model was trained on a different
#      split". If you see that warning for a stage, its printed test metrics are NOT held-out results - do not report them.
#   4. File → Save a copy in Drive keeps the outputs (confusion matrices, classification reports) for your report.
print("See the comments in this cell.")'''),

    ("List the reports (share these text files, not the images or weights)", '''for f in sorted(os.listdir(REPORTS)):
    print(f"{f:28} {os.path.getsize(f'{REPORTS}/{f}')/1e3:8.1f} kB")'''),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


def main():
    cells = [md(INTRO)]
    for n, (title, src) in enumerate(CELLS, 1):
        cells += [md(f"## {n}. {title}"), code(src)]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = os.path.join(ROOT, "colab", "Validate_Real_Models_and_Data.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
