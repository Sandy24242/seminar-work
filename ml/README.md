# ML pipeline — training and model export

## Source of truth

`notebooks/Dental_Caries_Pipeline_Stage1_to_Stage4_with_FDI_Stage2.ipynb` (203 cells) is the most complete version of the
project. It trains, evaluates, saves and reloads every model. All website preprocessing, thresholds and class mappings
come from this notebook and from the `*_metadata.json` files it writes. `backend/tests/test_notebook_parity.py` executes
the notebook's own code (Sections 3, 6.1, 6.6, 6.7, 76, 82) and checks that the website gives identical results.

| Notebook part | Model file written | Architecture (from notebook / metadata) | Classes |
|---|---|---|---|
| Part 5, §8–23 | `stage1_caries_model.keras` | EfficientNetB0 (ImageNet) + GAP + Dense(256) + sigmoid, 224 px, raw RGB 0–255 | 0 Healthy, 1 Caries; threshold from Youden's J on validation (0.6029), uncertainty band ±0.07 |
| Part 6, §24 | `stage2_fdi_detector.keras` | custom Keras grid detector, 26 FDI classes | two-digit FDI numbers |
| Part 7, §25–27 | `cavity_localizer_model.keras` | custom Keras grid detector (not YOLO; the YOLO `.txt` files are only the annotation format) | class-agnostic lesion boxes |
| Part 8–10, §28–48 | `stage3_gvblack_model.keras` | ResNet50 on lesion context crops (pad 0.5) | G.V. Black Class I–VI |
| Part 11–13, §49–74 | `stage4_icdas_model.keras` | EfficientNetB0 on tooth-scale windows | ICDAS 1–6 (ICDAS 0 is not modelled) |

All five `*_metadata.json` files were read from Google Drive on 23 Oct 2026; see `docs/DATA_AND_MODELS.md`
for the verified dataset counts, splits and caveats.

Dataset tools: `ml/dcai_data/` (paths + validation), `ml/inventory_datasets.py`, `ml/check_metadata.py`, `ml/colab/`.

Older notebooks (`Dental_Caries_Final_Pipeline_Stage1_to_Stage3*.ipynb`, `..._Stage1_to_Stage4_FINAL_MemorySafe`,
`GVBlack_*`) were **not available in this session**. According to the project history, the Stage 1–3 version skipped FDI
and ICDAS and used a MobileNetV2 G.V. Black classifier. The Stage 1–4 notebook above supersedes it.

## Training (Google Colab, GPU)

The training code was **not** extracted into standalone scripts. Rewriting about 200 notebook cells without the
datasets to re-run them would risk silently changing model semantics. Train with the notebook itself:

1. Upload the dataset archives to `MyDrive/dentalcaries/`. The notebook's Section 3 lists the expected names, e.g.
   `caries_2_FDI_250_*.zip`, `lowoccyolo*.zip` and `icdascl.zip`.
2. Open the notebook in Colab, then select **Runtime → Change runtime type → GPU**.
3. Run all cells. The models and metadata are written to `MyDrive/dentalcaries/pipeline_stage1_to_4_corrected/`.
4. Download the 5 `.keras` files and 5 `*_metadata.json` files into `backend/models/`, or point `DCAI_MODELS_DIR` to
   them. Then run `python backend/scripts/check_models.py`.

The datasets must not be committed to Git. Keep them on Drive or on a local disk outside the repository; `.gitignore`
excludes `*.zip`, `data/` and `datasets/`.

## Evaluation on the held-out test split

Every `*_metadata.json` stores the test metrics the notebook measured on its untouched test split: `test_metrics`
(Stage 1, FDI, localizer, ICDAS), `test_metrics_gt_crops` (G.V. Black on ground-truth lesion crops), `test_ordinal` and
`lesion_window_test_metrics` (ICDAS). Each file also stores the split sizes and a `split_fingerprint`. Print them with:

```bash
python backend/scripts/recorded_metrics.py backend/models --md recorded_metrics.md
```

These numbers come from the notebook; the script does not re-measure them. To re-measure:

1. Open the notebook in Colab with the same dataset ZIPs on Drive.
2. In Section 3 set `TRAIN_MODE = {"stage1": "load", "localizer": "load", "gvb": "load", "icdas": "load", "fdi": "load"}`.
3. Run all. Nothing is retrained. The notebook rebuilds the group split (Section 10), loads the saved models and runs
   its evaluation sections (19–22, 39–46, 63–72).
4. `decide()` (Section 6.6) compares each model's saved `split_fingerprint` with the rebuilt split. On a mismatch it
   only prints `WARNING: saved <stage> model was trained on a different split — its test metrics may be optimistic.`
   Treat any stage with that warning as **not** evaluated on held-out data.

Limits that apply to every metric: no dataset contains patient IDs, so the split is group-independent (duplicates,
near-duplicates and linked crops are kept together), not patient-independent. Stage 4 has no ICDAS 0 class.

To check the data itself first, run the read-only audit on the ZIPs:

```bash
python backend/scripts/audit_datasets.py healthy.zip caries.zip "lowoccyolo (4).zip" icdascl.zip caries_FDI_dataset_final.zip --near-dup --md dataset_audit.md
```

Or run `backend/colab/Validate_Real_Models_and_Data.ipynb`, which does all of the above in Colab, except the long
re-evaluation run.

