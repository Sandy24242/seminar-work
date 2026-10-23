"""scripts/validate_real_models.py, audit_datasets.py, recorded_metrics.py, fetch_models.py.

Model-dependent tests use the TEST-ONLY stand-in models from standin_models.py; the point is that the validator REFUSES
them unless explicitly told otherwise, and that its contract checks catch a metadata/model mismatch."""
import io
import json
import os
import sys
import zipfile

import numpy as np
import pytest
from PIL import Image

from standin_models import make_synthetic_photo, write_standin_models

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import audit_datasets  # noqa: E402
import fetch_models  # noqa: E402
import recorded_metrics  # noqa: E402
import validate_real_models  # noqa: E402


def _png(seed=None, rgb=None):
    if rgb is None:
        rgb = np.random.default_rng(seed).integers(0, 255, (80, 90, 3), dtype=np.uint8)
    b = io.BytesIO(); Image.fromarray(rgb).save(b, "PNG"); return b.getvalue()


# ------------------------------------------------------------------ validate_real_models
def test_validator_refuses_standins(standin_dir, tmp_path):
    out = tmp_path / "r.json"
    assert validate_real_models.main(["--models-dir", standin_dir, "--out", str(out)]) == 1
    rep = json.loads(out.read_text())
    assert rep["standin_models_used"] is True
    assert all(m["test_standin"] for m in rep["models"].values())


def test_validator_runs_cascade_and_labels_standin_outputs(standin_dir, tmp_path, capsys):
    img = tmp_path / "c.png"; img.write_bytes(_png(rgb=make_synthetic_photo("caries")))
    bad = tmp_path / "bad.png"; bad.write_bytes(b"nope")
    out = tmp_path / "r.json"
    code = validate_real_models.main(["--models-dir", standin_dir, "--images", str(tmp_path), "--out", str(out), "--allow-standins"])
    assert code == 0
    assert "TEST STAND-IN outputs (meaningless)" in capsys.readouterr().out
    preds = {p["image"]: p for p in json.loads(out.read_text())["predictions"]}
    assert preds["c.png"]["stage1"]["prediction"] == "CARIES" and preds["c.png"]["lesions"]
    assert "error" in preds["bad.png"]


def test_validator_contract_catches_metadata_mismatch(tmp_path):
    d = write_standin_models(str(tmp_path / "m"), meta_overrides={"gvb": {"image_size": 256},
                                                                   "icdas": {"class_names": ["a", "b"]}})
    out = tmp_path / "r.json"
    assert validate_real_models.main(["--models-dir", d, "--out", str(out), "--allow-standins"]) == 1
    rep = json.loads(out.read_text())["models"]
    failed = {(k, c["check"]) for k, m in rep.items() for c in m.get("checks", []) if not c["ok"]}
    assert ("gvb", "input side == metadata image_size") in failed
    assert ("icdas", "softmax units == len(class_names)") in failed
    assert not any(k in ("stage1", "localizer", "fdi") for k, _ in failed)


def test_validator_missing_models(tmp_path):
    out = tmp_path / "r.json"
    assert validate_real_models.main(["--models-dir", str(tmp_path), "--out", str(out)]) == 1
    assert not any(m["loaded"] for m in json.loads(out.read_text())["models"].values())


# ------------------------------------------------------------------ audit_datasets
@pytest.fixture()
def dataset_zips(tmp_path):
    with zipfile.ZipFile(tmp_path / "healthy.zip", "w") as z:
        z.writestr("healthy/a.png", _png(1)); z.writestr("healthy/b.png", _png(2))
        z.writestr("healthy/a_copy.png", _png(1)); z.writestr("healthy/broken.jpg", b"\xff\xd8 not really")
        z.writestr("__MACOSX/healthy/._a.png", b"junk")
    with zipfile.ZipFile(tmp_path / "caries.zip", "w") as z:
        z.writestr("caries/same_as_healthy_b.png", _png(2)); z.writestr("caries/c.png", _png(3))
    with zipfile.ZipFile(tmp_path / "lowoccyolo.zip", "w") as z:
        z.writestr("images/p1.png", _png(4)); z.writestr("images/p3.png", _png(5))
        z.writestr("labels/p1.txt", "0 0.5 0.5 0.2 0.2\n3 0.1 0.1 0.05 0.05\n")
        z.writestr("labels/p2.txt", "1 0.5 0.5 1.4 0.2\n2 0.4 0.4 x 0.1\n")
        z.writestr("classes.txt", "Class_I\nClass_II\nClass_III\nClass_IV\n")
    with zipfile.ZipFile(tmp_path / "icdascl.zip", "w") as z:
        for k in (1, 2, 5):
            z.writestr(f"icdascl/ICDAS_{k}/c{k}.png", _png(10 + k))
        z.writestr("icdascl/crop_manifest.csv", "crop,source_photo,score\nc1.png,UFV1.jpg,1\n")
    return tmp_path


def test_audit_reports_what_is_in_the_files(dataset_zips, tmp_path):
    out, mdp = tmp_path / "a.json", tmp_path / "a.md"
    paths = [str(dataset_zips / n) for n in ("healthy.zip", "caries.zip", "lowoccyolo.zip", "icdascl.zip")]
    assert audit_datasets.main(paths + ["--near-dup", "--out", str(out), "--md", str(mdp)]) == 0
    rep = json.loads(out.read_text())
    by = {r["dataset"]: r for r in rep["datasets"]}
    h = by["healthy.zip"]
    assert h["n_images"] == 4                                      # __MACOSX junk ignored
    assert h["n_unreadable"] == 1 and h["unreadable"][0]["file"] == "healthy/broken.jpg"
    assert h["exact_duplicate_extra_copies"] == 1
    assert rep["cross_dataset_exact_duplicates"] == {"caries.zip & healthy.zip": 1}
    assert rep["near_duplicates"]["pairs_by_dataset"]["caries.zip & healthy.zip"] >= 1
    y = by["lowoccyolo.zip"]["yolo"]
    assert y["class_id_counts"] == {"0": 1, "1": 1, "3": 1} and y["n_boxes"] == 3
    assert y["malformed_lines"] == 1 and y["boxes_out_of_range"] == 1
    assert y["images_without_label_file"] == 1 and y["label_files_without_image"] == 1
    assert y["class_names_file"] == ["Class_I", "Class_II", "Class_III", "Class_IV"]
    assert by["icdascl.zip"]["icdas_class_folders"] == {"ICDAS_1": 1, "ICDAS_2": 1, "ICDAS_5": 1}
    assert by["icdascl.zip"]["csv_files"][0]["columns"] == ["crop", "source_photo", "score"]
    ts = rep["task_support"]
    assert ts["healthy_vs_caries"]["supported"] and ts["icdas_classification"]["supported"]
    assert ts["lesion_localization"]["supported"] and not ts["fdi_tooth_detection"]["supported"]
    assert ts["patient_level_split_possible"] is False
    assert "# Dataset audit" in mdp.read_text()
    with zipfile.ZipFile(dataset_zips / "healthy.zip") as z:      # read-only: archive unchanged
        assert len(z.namelist()) == 5


def test_audit_detects_patient_ids_and_coco(tmp_path):
    coco = {"images": [{"id": 1, "file_name": "a.png", "patient_id": "P1"}], "categories": [{"id": 1, "name": "11"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}]}
    with zipfile.ZipFile(tmp_path / "caries_FDI.zip", "w") as z:
        z.writestr("a.png", _png(7)); z.writestr("ann.json", json.dumps(coco))
    out = tmp_path / "a.json"
    audit_datasets.main([str(tmp_path / "caries_FDI.zip"), "--no-decode", "--out", str(out)])
    r = json.loads(out.read_text())
    j = r["datasets"][0]["json_files"][0]
    assert j["format"] == "COCO" and j["annotations_per_category"] == {"11": 1}
    assert r["datasets"][0]["patient_ids"]["found"] is True
    assert r["task_support"]["fdi_tooth_detection"]["supported"] is True


# ------------------------------------------------------------------ recorded_metrics
def test_recorded_metrics_reads_back_metadata(tmp_path, capsys):
    meta = {"model": "cavity / lesion localizer (class-agnostic)", "split_images": {"train": 7, "val": 2, "test": 3},
            "split_fingerprint": "abc", "test_metrics": {"AP50": 0.25, "iou0.5": {"precision": 0.5, "recall": 0.4}}}
    (tmp_path / "cavity_localizer_model_metadata.json").write_text(json.dumps(meta))
    (tmp_path / "stage1_caries_model_metadata.json").write_text(json.dumps({"model": "TEST STAND-IN"}))
    md = tmp_path / "m.md"
    assert recorded_metrics.main([str(tmp_path), "--md", str(md)]) == 0
    text = md.read_text()
    assert "NOT re-measured" in text
    assert "AP50 = 0.2500" in text and "iou0.5.recall = 0.4000" in text and "'test': 3" in text
    assert "TEST STAND-IN metadata: no real metrics" in text
    assert "stage4_icdas_model_metadata.json" in text            # listed as missing


def test_recorded_metrics_renders_per_class_and_confusion(tmp_path):
    meta = {"test_metrics": {"accuracy": 0.5, "per_class": {"ICDAS 1": {"precision": 1.0, "support": 2}},
                             "confusion_matrix": [[1, 1], [0, 2]]}}
    (tmp_path / "stage4_icdas_model_metadata.json").write_text(json.dumps(meta))
    md = tmp_path / "m.md"
    recorded_metrics.main([str(tmp_path), "--md", str(md)])
    text = md.read_text()
    assert "| ICDAS 1 | 1.0000 | 2 |" in text and "    1     1" in text


# ------------------------------------------------------------------ fetch_models (no network: integrity checks only)
def test_fetch_validity_checks(standin_dir, tmp_path):
    assert fetch_models.valid(os.path.join(standin_dir, "stage1_caries_model.keras")) is None
    assert fetch_models.valid(os.path.join(standin_dir, "stage1_caries_model_metadata.json")) is None
    html = tmp_path / "x.keras"; html.write_text("<html>Google Drive - virus scan warning</html>")
    assert "not a ZIP" in fetch_models.valid(str(html))
    other = tmp_path / "y.keras"
    with zipfile.ZipFile(other, "w") as z:
        z.writestr("readme.txt", "hi")
    assert "Keras 3" in fetch_models.valid(str(other))
    bad = tmp_path / "z_metadata.json"; bad.write_text("{not json")
    assert "not valid JSON" in fetch_models.valid(str(bad))
