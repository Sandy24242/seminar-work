# backend/models/

This folder holds the real `*_metadata.json` files (tracked in Git). The trained weights are stored with Git LFS in the
**repository-root `models/` folder**, and the backend finds them there automatically after `git lfs pull`.

Alternatively, copy the five `.keras` files here from Google Drive → `dentalcaries/pipeline_stage1_to_4_corrected/`,
or set `DCAI_MODELS_DIR`. `.keras` files in this folder are git-ignored. Then check that the models load:

```bash
python scripts/check_models.py
```
