# Dental Caries AI Assessment System

*AI-assisted dental image analysis and caries assessment.* Academic proof of concept, not a clinical diagnosis.

This is a Streamlit website around the trained Stage 1–4 pipeline from
`Dental_Caries_Pipeline_Stage1_to_Stage4_with_FDI_Stage2.ipynb`. It collects patient details and the 7-question oral-habits
questionnaire, then runs the real models on an uploaded intraoral photograph:

| Stage | Model file (from the notebook) | What it does |
|---|---|---|
| 1 | `stage1_caries_model.keras` | Healthy vs caries (EfficientNetB0, sigmoid). HEALTHY → stop. UNCERTAIN → stop (optional: continue). |
| 2 | `stage2_fdi_detector.keras` | Tooth boxes + two-digit FDI numbers. Optional: reported as Unavailable when missing. |
| — | `cavity_localizer_model.keras` | Class-agnostic lesion boxes, each linked to the tooth box covering most of it. |
| 3 | `stage3_gvblack_model.keras` | G.V. Black Class I–VI per lesion (context crop). |
| 4 | `stage4_icdas_model.keras` | ICDAS 1–6 per lesion (tooth-scale window). ICDAS 0 is not modelled. |

Each `.keras` file needs its `*_metadata.json`. All thresholds, class mappings, crop sizes, padding, TTA and the ICDAS
window are read from the metadata, exactly as the notebook does. Reports are available as PDF, TXT, JSON, CSV, the
annotated JPEG, or a ZIP of all five.

## Run in Google Colab

Follow **`docs/Dental_Caries_AI_Colab_Guide.pdf`**, or open **`colab/Run_Dental_Caries_AI_Website.ipynb`** in Colab and run
the cells from top to bottom. In short:

1. Upload `Dental_Caries_AI_Website.zip` to `MyDrive/dentalcaries/`.
2. Make sure the models are in `MyDrive/dentalcaries/pipeline_stage1_to_4_corrected/`.
3. Run Steps 1–6 of the notebook. Step 7 gives you the website link:
   - **Option A:** Colab's private proxy.
   - **Option B:** a public Cloudflare quick tunnel.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-streamlit.txt
# put the 5 .keras files + 5 *_metadata.json files into models/  (or: export DCAI_MODELS_DIR=/path/to/folder)
python scripts/check_models.py
streamlit run app.py                                     # opens http://localhost:8501
```

### Where the app looks for the models

The first folder that contains `stage1_caries_model.keras` is used:

1. The `DCAI_MODELS_DIR` environment variable.
2. `models/` inside this project.
3. `/content/drive/MyDrive/dentalcaries/pipeline_stage1_to_4_corrected` (Colab with Drive mounted).

**The trained models are not inside this ZIP.** They total about 426 MB; the G.V. Black model alone is 272 MB. Download
them from the Drive folder `dentalcaries/pipeline_stage1_to_4_corrected`. The app never substitutes random or placeholder
models. A missing model is shown as **Unavailable**, and its stage is never reported as completed.

## Project structure

```
app.py                      Streamlit entry point (Home, Assessment, Results, Models & About)
config.py                   model file names, model-folder search order, upload limits, notebook fall-backs
pipeline/
  preprocessing.py          notebook Section 6.1 preprocessing contract + 6.7 TTA (unchanged)
  image_validation.py       upload checks (format, decode, EXIF, size) — notebook Section 82
  model_loader.py           loads .keras + metadata (keras.models.load_model, compile=False)
  stage1.py                 Stage 1 probability + decision rule (threshold, uncertainty band, confidence)
  stage2_fdi.py             FDI tooth detection + lesion→tooth association
  lesion_localization.py    localizer + per-lesion crop generation
  stage3_gvblack.py         G.V. Black per lesion (class names from metadata)
  stage4_icdas.py           ICDAS per lesion (scores from metadata)
  runner.py                 full cascade = notebook Section 76 run_pipeline (+ progress, graceful failures)
  annotation.py             annotated image = notebook draw_result
  questionnaire.py          7 questions, points, risk levels, advice = notebook Section 82
  report_generator.py       PDF (ReportLab), TXT, JSON, CSV, JPEG, ZIP
components/                 Streamlit UI pieces (styles, questionnaire, image viewer, result cards)
assets/                     logo, favicon, hero, tooth anatomy, stage icons, workflow diagram, fonts (DejaVu)
models/                     put the trained model files here for a local run (see models/README.md)
scripts/check_models.py     verifies the model files and loads them
colab/                      Colab notebook that installs and launches the website
docs/                       Colab guide (PDF) and test report
reference/                  code-only copy of the training notebook (used by the parity test)
tests/                      pytest suite + optional browser end-to-end check
tools/                      generators for assets, the Colab notebook and the PDF guide
```

## Questionnaire scoring

These rules come from notebook Section 82. They match the project specification.

| Question | Answers → points |
|---|---|
| Brush twice a day? | yes 0 · no 2 |
| Junk food / sugary foods or drinks | rarely 0 · once a day 1 · several times a day 3 |
| Sugary between meals / after night brushing | no 0 · sometimes 1 · often 2 |
| Fluoride toothpaste | yes 0 · not sure 1 · no 1 |
| Last dental visit | within 6 months 0 · 6-12 months ago 1 · more than 1 year / never 2 |
| Pain or sensitivity | no 0 · yes 2 |
| Smoke or chew tobacco | no 0 · yes 1 |

The maximum score is 13. **LOW** is 0–3, **MODERATE** is 4–7, **HIGH** is 8–13. The questionnaire score is reported
separately and never changes an AI prediction.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q                                   # 44 tests
python tests/e2e_browser_check.py           # optional: real browser run (needs `pip install playwright` + Chromium)
```

The tests use **test-only stand-in models** (`tests/standin_models.py`). These are hand-built Keras models with the real
file names, input/output shapes and metadata keys. They are not trained and are never used by the app.

`tests/test_notebook_parity.py` executes the notebook's own code from `reference/` (Sections 3, 6.1, 6.6, 6.7, 76 and 82).
It checks that the website returns identical results on the same models and images.

To repeat the parity check with **your trained models** and photographs:

```bash
DCAI_PARITY_MODELS_DIR=/path/to/pipeline_stage1_to_4_corrected DCAI_PARITY_IMAGES=/path/to/photos pytest -q tests/test_notebook_parity.py
```

## Privacy

- Uploaded photos are processed in memory and are never written to disk.
- Results live only in the browser session.
- Report files are generated in memory for download.
- Avoid entering more patient information than necessary; the ID field is optional.

## Disclaimer

This website is an academic AI proof of concept. Its outputs are model predictions from a photograph, not a clinical
diagnosis, and it is not a substitute for examination by a qualified dentist.

Decorative images in `assets/` are original illustrations generated by `tools/make_assets.py`.
`assets/clinical_caries_example.jpg` is the upload preview recorded in the notebook's own example run.
The DejaVu fonts are redistributed under their licence (`assets/fonts/DejaVu_LICENSE.txt`).
