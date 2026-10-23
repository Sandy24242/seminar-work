# Datasets and trained models — verified inventory

Source: Google Drive folder `1TGQXX8UzBXUkTg-0iARgvq_vklJFS_Lq` (college account), inspected on 23 Oct 2026 through the
Drive connector. The connector exposes folder listings and small files only. The large archives and `.keras` weights
could not be downloaded into the build environment (Drive downloads are denied by its network policy). Therefore:
- **Verified here:** folder structure, README, `classes.txt`, `data.yaml`, and all five `*_metadata.json` files.
- **Taken from the training notebook's own reports inside the metadata:** image counts, splits, class distributions and
  test metrics. These were not recomputed here.
- **Not verified here:** per-file corruption and duplicate checks, and loading the weights. Run
  `ml/colab/Dataset_Inventory_and_Model_Check.ipynb` to do this against the real files.

## Drive layout

```
<root>/
  pipeline_stage1_to_4_corrected/   trained models used by the website (5 × .keras + 5 × *_metadata.json + pipeline_metadata.json)
  caries_FDI_dataset_final/ (+ .zip)  caries_FDI_dataset/{images,labels}/<View>/, classes.txt, data.yaml, README.txt, "only fdi"/
  caries/ (+ caries.zip)            "Comprehensive visual index of carious lesions"/<Upper|Lower> <Front|Left|Right|Occlusal> View/
  healthy/ (+ healthy.zip)          "Mendeley dataset 19.4.2024"/"Teeth or Dental Image Dataset"/
  icdascl/ (+ icdascl.zip)          ICDAS_1_to_6_Classified/ICDAS_1..ICDAS_6/, crop_manifest.csv, images_without_caries.txt
  lowoccyolo (4).zip                lower-occlusal YOLO set (not opened)
  pipeline_stage1_to_4/, pipeline_stage1_to_3/, combined_pipeline*/, models/, stage1_*.keras   older, superseded runs
```

**Views vs labels.** Folders and filename codes for the eight photographic views (UFV, ULV, URV, UOV, LFV, LLV, LRV,
LOV) say *which part of the mouth was photographed*. They are **not** disease classes. Disease labels come only from:
- the healthy/caries dataset roots (Stage 1),
- YOLO class IDs in `.txt` files (FDI numbers, or lesion classes),
- `ICDAS_<k>` folder names (Stage 4).

**Patient identifiers.** No dataset documents patient IDs. The notebook builds leakage-safe splits by grouping
near-duplicate images with perceptual hashes (`GROUP_HAMMING` 6–8), not by patient.

## Which data supports which task

| Task | Data | Facts (from the notebook's metadata) | Verdict |
|---|---|---|---|
| Stage 1 healthy vs caries | `healthy/` + `caries/` | split 9184 / 1804 / 1522; 0 = Healthy, 1 = Caries; `images_without_caries.txt` list excluded | Supported, trained |
| Stage 2 FDI tooth detection | `caries_FDI_dataset` | 869 annotated / 870 images, YOLO txt, 0 unreadable, 0 orphan labels; 26 classes (11–16, 21–26, 31–37, 41–47); only certain teeth labelled; split 660/119/90 | Supported, trained. Classes 14, 15, 16, 24, 25, 26 have 13–16 boxes each and are flagged `unreliable_classes` |
| Lesion localization | lesion YOLO boxes (917 images) | class-agnostic; split 730/135/52 | Supported, trained (custom MobileNetV2 grid detector, **not YOLO**) |
| Stage 3 G.V. Black | the same lesion boxes, as context crops | 2,191 crops: I 542, II 404, III 511, IV 110, V 563, VI 61 | Trained, **with caveat:** the metadata states the class-ID → G.V. Black mapping is the project owner's confirmed mapping, "NOT provable from the data alone" (no mapping file in the ZIP). Classes IV and VI are thin |
| Stage 4 ICDAS | `icdascl` class folders | files: ICDAS 1 1348, 2 343, 3 2297, 4 31, 5 374, 6 1143; 11 conflicting near-duplicates removed; ICDAS 0 absent | Supported for scores 1–6; score 4 is flagged unreliable |

## Trained models (`pipeline_stage1_to_4_corrected/`)

| File | Size | Architecture (metadata) | Input / preprocessing | Notebook test-set result (as recorded) |
|---|---|---|---|---|
| `stage1_caries_model.keras` | 45.5 MB | EfficientNetB0 + Dense(256), sigmoid | 224 px raw RGB; threshold 0.6029 ± 0.07 | acc 0.938, ROC-AUC 0.975 (n = 1522) |
| `stage2_fdi_detector.keras` | 29.5 MB | MobileNetV2 grid detector + FDI head | 640 → 512 px, MobileNetV2 preprocess; score ≥ 0.55 | mAP50 0.840 (90 images) |
| `cavity_localizer_model.keras` | 29.0 MB | MobileNetV2 grid detector | 520 → 416 px; score ≥ 0.40 | AP50 0.373, F1 @ IoU 0.5 0.466 (52 images) |
| `stage3_gvblack_model.keras` | 272.0 MB | ResNet50 + Dense(256), softmax(6) | 224 px context crop, pad 0.5; TTA none | acc 0.731, macro-F1 0.641 (134 GT crops) |
| `stage4_icdas_model.keras` | 50.4 MB | EfficientNetB0 + Dense(256), softmax(6) | 224 px letterbox; TTA hflip+zoom; window ×1.5 | acc 0.747, macro-F1 0.532 (395 crops); on localized lesion windows acc 0.506 (89) |

These numbers were copied from the metadata files the notebook wrote. They were **not** reproduced in this repository.

## Compatibility with the website

Every metadata key that `backend/pipeline` reads is present with a usable type, including the TTA variants
(`none`, `hflip+zoom`) and the ICDAS `inference_window`. Run `python ml/check_metadata.py <models folder>` to re-check.
**Real weights verified (23 Oct 2026).** The five `.keras` files in `models/` (Git LFS) passed these checks:
- `git lfs fsck` is OK, and every archive passes a ZIP CRC check.
- All five load under TensorFlow 2.21.0 / Keras 3.13.2, taking 1.3–5.1 s each on CPU.
- They run the full cascade.
- `test_notebook_parity.py` with the real weights passes: the website's results equal the training notebook's own code
  on the same weights and image.

On `backend/assets/clinical_caries_example.jpg` (a 767×464 crop of the notebook's preview of `UFV72.jpg`), the outputs were:
- Stage 1 CARIES (P = 1.000).
- 3 teeth detected (11, an uncertain one, 21).
- 2 lesions:
  - L1 on tooth 21: G.V. Black Class III (0.60), ICDAS 3 (0.61).
  - L2 with no tooth number: Class IV (1.00), ICDAS 5 (0.62).

The notebook's saved figure for the original `UFV72.jpg` shows the same teeth and lesion positions, but L1 as Class IV /
ICDAS 5. L1's two confidences here are only about 0.6. Since the code paths are proven identical, the difference comes
from the input: the bundled image is a re-encoded screenshot crop, not the original photograph. To compare exactly, run
the smoke test on the original `UFV72.jpg`.
