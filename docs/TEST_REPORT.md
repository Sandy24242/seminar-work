# Test report — Dental Caries AI Website v1.0.0

Date: 21 Oct 2026. Environment: Linux, Python 3.13, TensorFlow 2.21.0, Keras 3.13.2, Streamlit 1.66.0, ReportLab 5.0.1.
These are the same TensorFlow/Keras versions your notebook printed in Colab.

## Source inspection

- **Notebook:** `Dental_Caries_Pipeline_Stage1_to_Stage4_with_FDI_Stage2.ipynb`, 203 cells. I inspected the configuration, preprocessing (6.1), prediction (6.6), TTA (6.7), detector architectures (24.4, 25.2), loading (75), cascade (76) and questionnaire/report (82).
- **Google Drive folder `dentalcaries`:** listed through the Drive connector. `pipeline_stage1_to_4_corrected/` contains all 5 trained models and 5 metadata files. I read `stage1_caries_model_metadata.json`: EfficientNetB0, 224 px, threshold 0.6029, uncertainty margin 0.07, confidence threshold 0.60.
- **Other stages' settings:** taken from the notebook's own Section 75 output:
  - Localizer: score ≥ 0.40, input 416.
  - G.V. Black: ResNet50, Classes I–VI, pad 0.5.
  - ICDAS: EfficientNetB0, scores 1–6, window ×1.5 / min scale 1.0.
  - FDI: 26 classes, score ≥ 0.55, number confidence ≥ 0.50.
- **Questionnaire:** identical to the specification. Max score 13; LOW 0–3, MODERATE 4–7, HIGH 8–13. No discrepancy.

## Automated tests: `pytest -q` → 44 passed

| File | What is verified |
|---|---|
| `test_questionnaire.py` | Questions, options and points equal the specification. All 648 answer combinations, risk boundaries, the notebook example (9/13 HIGH), invalid answers, advice. |
| `test_validation.py` | PNG/JPEG accepted. Empty file, wrong extension, fake content, GIF renamed to .png, truncated JPEG and images under 64 px rejected. EXIF orientation applied. RGBA/grey converted to RGB. Safe filenames. |
| `test_pipeline.py` | Loading through `keras.models.load_model`. Caries cascade, HEALTHY stop, UNCERTAIN stop and continue-if-uncertain, Stage 1 decision rule. FDI association: ASSIGNED, AMBIGUOUS, NO_TOOTH. No-lesion handling, lower-threshold flag, real progress callbacks. Missing FDI / localizer / classifiers / Stage 1, missing metadata, corrupted model file, empty folder, uncertainty thresholds and class mappings from metadata, JSON serialisation. |
| `test_reports.py` | PDF (text-extracted and checked), TXT, JSON, CSV, annotated JPEG and ZIP. Healthy report does not claim downstream stages. |
| `test_notebook_parity.py` | The notebook's own code (Sections 3, 6.1, 6.6, 6.7, 76, 82), executed from `reference/`, compared with the website. Preprocessing functions, decoders, `run_pipeline` results on 8 images × 3 option sets, annotation pixels, lesion/teeth tables, report text, questionnaire (648 combinations) and upload validation are all **identical**. |
| `test_app.py` | Streamlit AppTest: home page, Start Assessment navigation, form validation, questionnaire submission (9/13 HIGH), models page (loaded and missing cases), results page with real pipeline output and all 6 download buttons. |

## End-to-end browser check: `tests/e2e_browser_check.py` → passed

- The real `streamlit run app.py` server passed its health check.
- Headless Chromium completed the full flow: home → Start Assessment → form + 7 answers → 9/13 HIGH → upload → Analyze → results with all stage sections → PDF download, verified as a valid PDF.
- The mobile-width page was also rendered. The screenshots are in `docs/screenshots/`.

## Simulated Colab run

The cells of `colab/Run_Dental_Caries_AI_Website.ipynb` ran locally with these adaptations:

- `/content` was mapped to a temp folder.
- The Drive layout was simulated.
- `google.colab` calls and `pip` were skipped.

Steps 2–6 and 8 worked: ZIP found, extraction checks OK, models copied, `check_models.py` reported all required models loaded, Streamlit started, health OK, and the page was served over HTTP.

## Not verified (environment limits)

- **Real model weights:** Google Drive downloads are blocked by this environment's network policy, so the trained weights (≈ 426 MB) could not be downloaded. All model-dependent tests therefore used **test-only stand-in models** (`tests/standin_models.py`). These have the same file names, input/output shapes and metadata keys, and they exercise the real loading and inference code. Predictions from the real models were **not** run here.
- **Google Colab itself:** Colab could not be launched. Not executed:
  - `drive.mount`
  - the Colab proxy (Option A)
  - the Cloudflare quick tunnel (Option B)
  - Colab's preinstalled package set
- **Stage 3/4 input sizes and TTA variants:** these are read from the real metadata at runtime. Only the Stage 1 metadata file was read in full here.

To complete verification with the trained models, run the Colab guide Step 5 (`check_models.py`) and the Section 7 example (UFV72.jpg). Optionally, also run:

```bash
DCAI_PARITY_MODELS_DIR=<model folder> DCAI_PARITY_IMAGES=<photos> pytest -q tests/test_notebook_parity.py
```
