# models/

This folder is intentionally empty in Git and in the ZIP. The trained models total about 426 MB, so they were not packaged.
The app never uses placeholder or random models.

For a local run, copy these 10 files here from Google Drive → `dentalcaries/pipeline_stage1_to_4_corrected/`:

| Model file | Metadata file | Size (Drive) |
|---|---|---|
| `stage1_caries_model.keras` | `stage1_caries_model_metadata.json` | 45.5 MB |
| `stage2_fdi_detector.keras` | `stage2_fdi_detector_metadata.json` | 29.5 MB |
| `cavity_localizer_model.keras` | `cavity_localizer_model_metadata.json` | 29.0 MB |
| `stage3_gvblack_model.keras` | `stage3_gvblack_model_metadata.json` | 272.0 MB |
| `stage4_icdas_model.keras` | `stage4_icdas_model_metadata.json` | 50.4 MB |

Then check that they load:

```bash
python scripts/check_models.py
```

In Google Colab you do not need to copy anything by hand. Step 5 of the Colab notebook/guide copies the files from Drive
into this folder for you.

Alternatively, share the files as "Anyone with the link can view", list their Drive file IDs in `drive_ids.json` in this
folder (git-ignored), and run `python scripts/fetch_models.py models/drive_ids.json`. Then check the models against
their metadata and run real predictions:

```bash
python scripts/validate_real_models.py --images /path/to/photos
```

