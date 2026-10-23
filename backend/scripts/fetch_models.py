"""
Download the trained models + metadata from Google Drive into backend/models/ (or DCAI_MODELS_DIR).

    python scripts/fetch_models.py models/drive_ids.json

drive_ids.json maps each file name to its Google Drive file ID (the part after /d/ in the file's share link):

    {"stage1_caries_model.keras": "1AbC...", "stage1_caries_model_metadata.json": "1XyZ...", ...}

The manifest is git-ignored on purpose: with "Anyone with the link" sharing, the IDs are enough to download the weights.
Each file must be shared as "Anyone with the link can view". Downloads go to <name>.part first and are only renamed
when complete; .keras files must be valid ZIP archives and metadata files valid JSON, otherwise the file is rejected.
Files already present with a valid structure are skipped (use --force to download again).

If you can use Google Colab instead, no download is needed: the Colab notebooks read the models from the mounted Drive.
"""
import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"


def valid(path):
    if path.endswith(".keras"):
        if not zipfile.is_zipfile(path):
            return "not a ZIP archive (Drive may have returned an HTML page: check the sharing setting)"
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            names = set(z.namelist())
        if bad:
            return f"damaged archive member {bad}"
        if not {"config.json", "metadata.json"} & names:
            return "ZIP does not look like a Keras 3 .keras file"
        return None
    try:
        with open(path) as f:
            json.load(f)
    except ValueError as e:
        return f"not valid JSON ({e})"
    return None


def download(file_id, dst):
    tmp = dst + ".part"
    with urllib.request.urlopen(URL.format(id=file_id), timeout=120) as r, open(tmp, "wb") as f:
        if "text/html" in (r.headers.get("Content-Type") or ""):
            raise RuntimeError("Drive returned an HTML page instead of the file (file not shared, or download blocked)")
        shutil.copyfileobj(r, f, length=1 << 20)
    return tmp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("--dest", default=os.environ.get("DCAI_MODELS_DIR") or os.path.join(config.PROJECT_ROOT, "models"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    with open(a.manifest) as f:
        ids = json.load(f)
    expected = set(config.MODEL_FILES.values()) | set(config.META_FILES.values())
    unknown = sorted(set(ids) - expected - {"pipeline_metadata.json"})
    if unknown:
        print("Ignoring unexpected names in the manifest:", unknown)
    os.makedirs(a.dest, exist_ok=True)
    failed = []
    for name in sorted(set(ids) - set(unknown)):
        dst = os.path.join(a.dest, name)
        if os.path.isfile(dst) and not a.force and valid(dst) is None:
            print(f"  skip   {name} (already present, {os.path.getsize(dst) / 1e6:.1f} MB)")
            continue
        print(f"  fetch  {name} ...", flush=True)
        try:
            tmp = download(ids[name], dst)
            why = valid(tmp)
            if why:
                os.remove(tmp)
                raise RuntimeError(why)
            os.replace(tmp, dst)
            print(f"         ok, {os.path.getsize(dst) / 1e6:.1f} MB")
        except Exception as e:
            failed.append(name)
            print(f"  FAILED {name}: {type(e).__name__}: {e}")
    missing = sorted(expected - set(ids))
    if missing:
        print("Not in the manifest:", missing)
    print("\nNext: python scripts/validate_real_models.py --images <photos>")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
