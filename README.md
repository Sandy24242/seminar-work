# Dental Caries AI

AI-assisted analysis of intraoral photographs, built as an academic proof of concept. The system runs the trained
multi-CNN cascade from the project notebook:

**Stage 1** healthy vs caries → **Stage 2** FDI tooth numbers (optional) → **lesion localization** → per lesion:
**Stage 3** G.V. Black class and **Stage 4** ICDAS score.

> ⚠️ Results are experimental model predictions, not a professional dental diagnosis.

## Repository layout

```
frontend/   React + Vite UI (upload, model status, results, detection overlay)
backend/    Python: FastAPI REST API (api/main.py) + inference pipeline (pipeline/) + optional Streamlit UI
            (streamlit_app.py), tests/, scripts/check_models.py, models/ (put trained weights here; not in Git)
notebooks/  Training notebook: Dental_Caries_Pipeline_Stage1_to_Stage4_with_FDI_Stage2.ipynb (code-only copy)
ml/         Training / export instructions (ml/README.md)
docs/       Colab guide (PDF), previous test report, screenshots, Streamlit README
```

## Status of each stage

| Stage | Code | Trained weights | Verified with real weights in this repo's CI/session? |
|---|---|---|---|
| Stage 1 — Healthy vs Caries | ✅ | On your Google Drive (`stage1_caries_model.keras`, 45 MB) | ❌ — Drive download blocked in the build environment |
| Stage 2 — FDI tooth numbers | ✅ (optional stage) | On Drive (`stage2_fdi_detector.keras`, 30 MB) | ❌ same |
| Lesion localizer | ✅ | On Drive (`cavity_localizer_model.keras`, 29 MB) | ❌ same |
| Stage 3 — G.V. Black | ✅ | On Drive (`stage3_gvblack_model.keras`, 272 MB) | ❌ same |
| Stage 4 — ICDAS | ✅ | On Drive (`stage4_icdas_model.keras`, 50 MB) | ❌ same |

The weights are **not** in Git: they total about 426 MB, and the G.V. Black model alone exceeds GitHub's 100 MB file
limit. Without them, `/models/status` reports every model as unavailable and `/predict` returns HTTP 503. The website
never invents a prediction.

On 23 Oct 2026 the Drive folder `pipeline_stage1_to_4_corrected/` was listed (names and sizes only). All five `.keras` files
and their five `*_metadata.json` files are there under the exact names the code expects, plus `pipeline_metadata.json`.
Downloading them into the cloud build environment was blocked by its network policy, so **no prediction from the real
weights has been run or reported in this repository yet**. Run the validation in Colab (below) to do that.

## Setup (Windows PowerShell, Python 3.12, Node 18+)

```powershell
git clone https://github.com/Sandy24242/seminar-work.git
cd seminar-work

# 1) Backend
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt          # includes requirements.txt (TensorFlow 2.21.0 / Keras 3.13.2)
```

### Put the trained models in place

From Google Drive → `dentalcaries/pipeline_stage1_to_4_corrected/`, download these 10 files into `backend\models\`:
`stage1_caries_model.keras`, `stage2_fdi_detector.keras`, `cavity_localizer_model.keras`, `stage3_gvblack_model.keras`,
`stage4_icdas_model.keras`, plus each file's `*_metadata.json`. If Drive renamed a file (for example
`stage1_caries_model (1).keras`), rename it back. Alternatively, keep the files elsewhere and set
`$env:DCAI_MODELS_DIR="D:\path\to\folder"`.

```powershell
python scripts\check_models.py               # loads every model and prints what is missing
```

Optional: if the files are shared as "Anyone with the link can view", `scripts\fetch_models.py` can download them. It
reads a git-ignored manifest `backend\models\drive_ids.json` (`{"stage1_caries_model.keras": "<Drive file ID>", ...}`)
and rejects anything that is not a valid `.keras` ZIP or valid JSON (for example a Drive HTML warning page):
`python scripts\fetch_models.py models\drive_ids.json`.

### Run (two terminals)

```powershell
# Terminal 1 — API on http://127.0.0.1:8000
cd backend; .\.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — website on http://127.0.0.1:5173
cd frontend
npm install
copy .env.example .env.local                 # VITE_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```

Optional alternative UI (the original Streamlit app, which includes the questionnaire and PDF reports):
`streamlit run streamlit_app.py` from `backend\`.

To run in Google Colab instead, open `backend/colab/Run_Dental_Caries_AI_Website.ipynb`. Step 3 now clones this GitHub
repository (entry point `backend/streamlit_app.py`). It falls back to `Dental_Caries_AI_Website.zip` on Drive only if
GitHub cannot be reached. `docs/Dental_Caries_AI_Colab_Guide.pdf` still describes the ZIP route; its other steps are
unchanged.

## Validate the real models and the datasets

These scripts never train anything, never modify the datasets, and refuse the test-only stand-in models. All
inference goes through the same `pipeline/` code as the website.

| Script (in `backend/`) | What it does |
|---|---|
| `scripts/check_models.py [folder]` | Lists the 10 files and loads every model. |
| `scripts/validate_real_models.py --images <files/folders> [--out r.json]` | Checks each model's input/output shape against its metadata (image size, detector grid, number of classes, Stage 1 class mapping and threshold), then runs the real cascade on the images and prints Stage 1–4 outputs. Exits 1 if stand-in models are found. |
| `scripts/audit_datasets.py <zips/folders> [--near-dup] [--md a.md]` | Read-only dataset audit: formats, counts per folder or class, unreadable files, MD5 duplicates within and across datasets, dHash near-duplicates, YOLO label checks (class ids, malformed lines, out-of-range boxes, unmatched images/labels), COCO JSON, ICDAS class folders, and whether any patient ID exists. Ends with which tasks the data supports. |
| `scripts/recorded_metrics.py [folder] [--md m.md]` | Prints the held-out test metrics the training notebook stored in each metadata file. It reads them back without re-measuring. |

**Easiest route: Google Colab with your Drive mounted.** Open `backend/colab/Validate_Real_Models_and_Data.ipynb` and run
all cells. It copies the models from `MyDrive/dentalcaries/pipeline_stage1_to_4_corrected/` and runs the checks. It then
runs genuine predictions on a few healthy and caries photos from your archives, both directly and through the FastAPI
`/predict` endpoint. Finally it audits the dataset ZIPs and writes text-only reports to
`MyDrive/dentalcaries/dcai_validation_reports/`. Those sample photos come from the training archives, so their outputs
show that the models run; they are **not** an evaluation.

**Held-out evaluation.** The notebook does not store its per-image split, so held-out metrics can only be re-measured by
re-running its leakage-controlled group split (Section 10). Set `TRAIN_MODE` to `"load"` for every stage and run the
notebook. If it prints `WARNING: saved <stage> model was trained on a different split`, that stage's printed test metrics
are not held-out results. See [`ml/README.md`](ml/README.md).

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Process health: `{"status":"ok",...}` |
| GET | `/models/status` | `ready`, `missing_required`, and a per-model `loaded` / `reason` / `class_names` entry; upload limits |
| POST | `/predict` | multipart field `file` (JPG/PNG, ≤ 20 MB, ≥ 64 px), optional `continue_if_uncertain=true` |

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/models/status
curl.exe -F "file=@backend\assets\clinical_caries_example.jpg" http://127.0.0.1:8000/predict
```

`/predict` responses:
- **200**: `overall`, `stage1` (prediction, confidence, p_caries, threshold), `stage2`, `localization`, `stage3`, `stage4`,
  `lesions[]` (box in pixels, localizer score, FDI tooth, G.V. Black and ICDAS with confidences), `stage_status`, and
  `annotated_jpeg_base64`.
- **422**: invalid file (wrong type, not a real image, empty, too small).
- **413**: file too large.
- **503**: the Stage 1 model is not loaded.

Stages whose model is missing have `status: "UNAVAILABLE"` and a `reason`.

## Training

See [`ml/README.md`](ml/README.md). Training runs in Google Colab (GPU) with the notebook, using the datasets on your
Drive. The notebook writes the `.keras` + metadata files that the backend loads.

## Tests

```powershell
cd backend; .\.venv\Scripts\Activate.ps1
pytest -q
cd ..\frontend; npm run build
```

Observed on 23 Oct 2026 (Linux, Python 3.13, TF 2.21.0, Keras 3.13.2): `pytest -q` → **64 passed**. That is the 44
original tests (including notebook parity), 11 API tests and 9 tests for the validation scripts. `npm run build`
succeeded. The React site was also driven in headless Chromium against the API: status panel, caries and healthy
uploads, the detection overlay, and rejection of `.txt` and fake-PNG uploads. Stand-in models were used, and the UI
labelled them as such.

Model-dependent tests use **test-only stand-in models** (`backend/tests/standin_models.py`). These are untrained Keras
models with the real file names, shapes and metadata keys. They test the code paths only and are never used by the app.
To check parity against your real weights:

```powershell
$env:DCAI_PARITY_MODELS_DIR="backend\models"; $env:DCAI_PARITY_IMAGES="D:\photos"; pytest -q tests\test_notebook_parity.py
```

`/models/status` reports `test_standin` per model and `test_standins_loaded`; `/predict` returns `test_standins_used`.
These are true only for the test fixtures, and the website then shows a warning on both the status panel and the result.

## Known limitations

- **No real-weight results yet.** Nothing in this repository has been run with the trained weights; see the status
  table above. The Colab validation notebook is the next step.
- **The split is not patient-level.** The notebook states that no dataset contains patient IDs. Its split groups exact
  and near-duplicate photos and linked crops, so it is group-independent but not guaranteed patient-independent.
- **ICDAS 0 is not modelled.** Stage 4 outputs ICDAS 1–6 only, and only for lesions that Stage 1 and the localizer
  passed on.
- **Stage 2 (FDI) is optional.** If it is unavailable, lesions are reported without tooth numbers.
- **Training ran in the notebook only.** It was not extracted into standalone scripts (see `ml/README.md`).
- **Not a diagnostic device.** This is an academic proof of concept and must not be used for clinical decisions.
