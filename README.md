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

| Stage | Trained weights (`models/`, Git LFS) | Verified with the real weights (23 Oct 2026) |
|---|---|---|
| Stage 1 — Healthy vs Caries | `stage1_caries_model.keras` (45 MB) | ✅ loads; real inference |
| Stage 2 — FDI tooth numbers (optional) | `stage2_fdi_detector.keras` (30 MB) | ✅ loads; real inference |
| Lesion localizer | `cavity_localizer_model.keras` (29 MB) | ✅ loads; real inference |
| Stage 3 — G.V. Black | `stage3_gvblack_model.keras` (272 MB) | ✅ loads; real inference (class mapping is owner-confirmed, see docs) |
| Stage 4 — ICDAS | `stage4_icdas_model.keras` (50 MB) | ✅ loads; real inference (ICDAS 4 unreliable, no ICDAS 0) |

"Verified" means the trained weights load under TensorFlow 2.21.0 / Keras 3.13.2, and the full cascade runs on a real
photograph through the API and the React UI. It also means the website's output equals the training notebook's own code
on the same weights and image (`test_notebook_parity.py`). It says nothing about clinical accuracy. The accuracy figures
are the notebook's recorded test metrics in [`docs/DATA_AND_MODELS.md`](docs/DATA_AND_MODELS.md); they were not
re-measured here.

The weights are stored with **Git LFS**. After cloning, run `git lfs install` and `git lfs pull`; otherwise the
`.keras` files are 133-byte pointer files and every model reports *unavailable*. Without the weights, `/predict`
returns HTTP 503. The website never invents a prediction.

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

```powershell
git lfs install
git lfs pull                                 # downloads models\*.keras (~426 MB)
cd backend
python scripts\check_models.py               # loads every model and prints what is missing
```

The backend uses the first of these folders that contains `stage1_caries_model.keras`:
1. `$env:DCAI_MODELS_DIR`.
2. `backend\models\`.
3. `models\` at the repository root, which holds the Git LFS weights.
4. The Drive folder `dentalcaries/pipeline_stage1_to_4_corrected` (Colab).

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

To run in Google Colab instead, see `docs/Dental_Caries_AI_Colab_Guide.pdf` and
`backend/colab/Run_Dental_Caries_AI_Website.ipynb`.

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

## Deploying the backend (Docker)

The backend needs TensorFlow and about 1 GB of RAM at run time with all five models loaded, so it does **not** fit
Vercel's standard serverless functions (see [`docs/VERCEL_DEPLOYMENT.md`](docs/VERCEL_DEPLOYMENT.md)). Run it as a container instead. The `Dockerfile` at the repository root bakes in the
weights from `models/`:

```bash
git lfs install && git lfs pull                 # the build fails if models/*.keras are LFS pointer files
docker build -t dental-caries-api .
docker run --rm -p 8000:8000 -e DCAI_CORS_ORIGINS=https://<your-frontend>.vercel.app dental-caries-api
curl http://127.0.0.1:8000/models/status        # "ready": true
```

- **Hosts:** Render, Railway, Fly.io, Hugging Face Spaces (Docker SDK, `app_port: 8000`) or any VM can build from this
  `Dockerfile`. The container listens on `$PORT` (default 8000). Pick a plan with at least 1.5 GB of RAM. The host must
  fetch Git LFS files when it clones the repository; otherwise the build stops with a message naming the pointer files.
- **Without weights in the image:** build with `--build-arg REQUIRE_MODELS=0`, then mount the models at run time with
  `-v /path/to/models:/app/models:ro`.
- **Frontend:** keep it on Vercel. Set `VITE_API_BASE_URL` to the backend's public URL in the Vercel project settings,
  then redeploy.
- **Image contents:** the image installs only `backend/requirements.txt`, the API-only dependency set without Streamlit. It runs as a non-root user
  with one worker, and has a health check on `/health`.

Verified on 23 Oct 2026 with Docker 29.9.1:
- The image builds, and the build fails with a clear message when the weights are LFS pointers.
- The container starts in about 20 s, and `/models/status` reports all 5 models loaded.
- `/predict` gives the same real-model result as the local run, and a `.txt` upload gets HTTP 422.
- CORS headers are correct, the process runs as uid 1000, the health check reports `healthy`, and memory use is about
  1 GB after inference.

## Datasets (kept on Google Drive, never in Git)

The datasets and weights live in the project Drive folder. [`docs/DATA_AND_MODELS.md`](docs/DATA_AND_MODELS.md) has
the verified layout, which task each dataset supports, and the caveats.

| Dataset folder on Drive | Used for |
|---|---|
| `healthy/` + `caries/` | Stage 1 healthy vs caries (subfolders of `caries/` are photographic **views**, not classes) |
| `caries_FDI_dataset_final/` | Stage 2 FDI detection (YOLO `.txt`, 26 FDI classes) |
| lesion YOLO boxes (notebook Section 25/29) | localizer + Stage 3 G.V. Black (class mapping is owner-confirmed, not provable from the files) |
| `icdascl/ICDAS_1_to_6_Classified/` | Stage 4 ICDAS 1–6 |
| `pipeline_stage1_to_4_corrected/` | the trained `.keras` + `*_metadata.json` the website loads |

Configure paths with environment variables. The defaults match the notebook.

| Variable | Default | Meaning |
|---|---|---|
| `DCAI_DATA_ROOT` | `/content/drive/MyDrive/dentalcaries` | folder holding the dataset folders / ZIPs |
| `DCAI_<KEY>_DIR` | `<root>/<folder>` | override one dataset (`FDI`, `CARIES_VIEWS`, `HEALTHY`, `ICDAS`, `LOWER_OCCLUSAL_YOLO`) |
| `DCAI_MODELS_DIR` | `backend/models`, then the Drive folder | trained models for the backend |

If the folder is shared with you rather than owned, add a shortcut to it in *My Drive* so that Colab can reach it
under `/content/drive/MyDrive/...`.

Tools (run them where the data is mounted, normally Colab):

```bash
python ml/inventory_datasets.py --out dataset_inventory --check-zips   # counts, labels, corrupt files, duplicates, views
python ml/check_metadata.py "$DCAI_MODELS_DIR"                          # metadata vs website contract (no weights needed)
python backend/scripts/smoke_test_real_models.py --models "$DCAI_MODELS_DIR" --images <held-out photos>
```

The Colab notebook [`ml/colab/Dataset_Inventory_and_Model_Check.ipynb`](ml/colab/Dataset_Inventory_and_Model_Check.ipynb)
runs all of these against your Drive, followed by the notebook-parity test on the real weights.

## Training

See [`ml/README.md`](ml/README.md). Training runs in Google Colab (GPU) with the notebook, using the datasets on your
Drive. The notebook writes the `.keras` + metadata files that the backend loads.

## Tests

```powershell
cd backend; .\.venv\Scripts\Activate.ps1
pytest -q
cd ..\frontend; npm run build
```

Observed on 23 Oct 2026 (Linux, Python 3.13, TF 2.21.0, Keras 3.13.2): `pytest -q` → **54 passed**. That is the 44
original tests, including notebook parity, plus 10 API tests. `npm run build` succeeded.

**Which tests use real weights.** None of the automated tests above use the trained weights. Real-weight checks are
`scripts/check_models.py`, `scripts/smoke_test_real_models.py` (it refuses stand-ins) and `test_notebook_parity.py`
with `DCAI_PARITY_MODELS_DIR` set. Results with the real weights on 23 Oct 2026:
- `smoke_test_real_models.py --models ..\models`: all 5 models loaded and the cascade completed.
- `test_notebook_parity.py` with `DCAI_PARITY_MODELS_DIR=..\models`: **4 passed**.
- The live API returned `/models/status` `ready: true` with all models loaded, a real `/predict` result, and HTTP 422
  for a `.txt` upload.
- The React UI showed the same result (`docs/screenshots/react_real_models.png`).

The only real photograph available in the repository is `backend/assets/clinical_caries_example.jpg`, a crop of the
notebook's preview of `UFV72.jpg`, so these runs show the models work, not how accurate they are. `ml/tests` (7 tests) checks the dataset utilities on small generated fixtures.

Model-dependent tests use **test-only stand-in models** (`backend/tests/standin_models.py`). These are untrained Keras
models with the real file names, shapes and metadata keys. They test the code paths only and are never used by the app.
To check parity against your real weights:

```powershell
$env:DCAI_PARITY_MODELS_DIR="backend\models"; $env:DCAI_PARITY_IMAGES="D:\photos"; pytest -q tests\test_notebook_parity.py
```
