"""Unit tests for ml/dcai_data and ml/check_metadata on tiny GENERATED fixtures (not the real datasets)."""
import json
import os
import sys
import zipfile

import numpy as np
from PIL import Image

ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)

import check_metadata  # noqa: E402
from dcai_data.validate import (find_duplicates, inventory_class_folders, inventory_yolo, parse_yolo_file,  # noqa: E402
                                view_of, zip_integrity)


def _img(path, seed=0, size=(80, 60)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), np.uint8)).save(path)


def test_view_is_not_a_label():
    assert view_of("images/Upper Occlusal View/a.jpg") == "UOV"
    assert view_of("crops/IMG_LFV_0012.jpg") == "LFV"
    assert view_of("ICDAS_3/x.jpg") is None


def test_parse_yolo_reports_problems(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.2\n3 1.4 0.5 0.1 0.1\nbad line\n")
    boxes, probs = parse_yolo_file(str(p))
    assert len(boxes) == 2 and len(probs) == 2


def test_inventory_yolo(tmp_path):
    root = tmp_path / "fdi"
    (root / "classes.txt").parent.mkdir(parents=True)
    (root / "classes.txt").write_text("11\n12\n")
    _img(str(root / "images/Upper Front View/a.jpg"), 1)
    _img(str(root / "images/Upper Front View/b.jpg"), 2)
    _img(str(root / "images/only fdi/c.jpg"), 3)
    os.makedirs(root / "labels/Upper Front View")
    (root / "labels/Upper Front View/a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    (root / "labels/Upper Front View/orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "images/Upper Front View/broken.jpg").write_bytes(b"not a jpeg")
    r = inventory_yolo(str(root), exclude_dirs=("only fdi",))
    assert r["n_images"] == 3 and r["n_labelled"] == 1 and r["n_unlabelled"] == 2
    assert r["boxes_per_class"] == {"11": 1, "12": 1}
    assert r["images_per_view"] == {"UFV": 3}
    assert len(r["corrupt_images"]) == 1 and r["n_orphan_labels"] == 1


def test_inventory_yolo_without_mapping(tmp_path):
    _img(str(tmp_path / "images/a.jpg"))
    os.makedirs(tmp_path / "labels")
    (tmp_path / "labels/a.txt").write_text("4 0.5 0.5 0.2 0.2\n")
    r = inventory_yolo(str(tmp_path))
    assert r["class_names_source"].startswith("MISSING") and r["boxes_per_class"] == {"id_4": 1}


def test_class_folders_and_duplicates(tmp_path):
    for k, seed in (("ICDAS_1", 1), ("ICDAS_1", 2), ("ICDAS_6", 3)):
        _img(str(tmp_path / k / f"{seed}.png"), seed)
    _img(str(tmp_path / "misc" / "x.png"), 9)
    r = inventory_class_folders(str(tmp_path))
    assert r["images_per_class"] == {"1": 2, "6": 1} and r["images_outside_class_folders"] == {"misc": 1}
    Image.open(tmp_path / "ICDAS_1" / "1.png").save(tmp_path / "ICDAS_6" / "dup.png")
    groups = find_duplicates([str(p) for p in tmp_path.rglob("*.png")])
    assert len(groups) == 1 and len(groups[0]) == 2


def test_zip_integrity(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("x.txt", "hello")
    assert zip_integrity(str(z)) == (True, 1, None)


def test_check_metadata_contract(tmp_path):
    """Fixture metadata with the key layout read from the real Drive files (values here are fixtures)."""
    meta = {
        "stage1_caries_model_metadata.json": {"image_size": 224, "threshold": 0.6, "uncertain_margin": 0.07,
                                              "confidence_threshold": 0.6, "class_names": ["Healthy", "Caries"]},
        "stage2_fdi_detector_metadata.json": {"class_names": ["11"], "input_size": 512, "cache_side": 640,
                                              "score_threshold": 0.55, "class_confidence_threshold": 0.5, "nms_iou": 0.45,
                                              "max_detections": 40, "assoc_min_overlap": 0.3, "assoc_ambiguity_ratio": 0.8},
        "cavity_localizer_model_metadata.json": {"input_size": 416, "cache_side": 520, "score_threshold": 0.4,
                                                 "nms_iou": 0.4, "max_detections": 10},
        "stage3_gvblack_model_metadata.json": {"class_names": ["GVB_Class_I"], "image_size": 224, "context_pad": 0.5,
                                               "confidence_threshold": 0.5, "tta": "none"},
        "stage4_icdas_model_metadata.json": {"scores": [1, 2], "class_names": ["ICDAS 1", "ICDAS 2"], "image_size": 224,
                                             "confidence_threshold": 0.5, "tta": "hflip+zoom",
                                             "inference_window": {"mult": 1.5, "min_frac_w": 0.33, "min_frac_h": 0.55}},
    }
    for k, v in meta.items():
        (tmp_path / k).write_text(json.dumps(v))
    assert check_metadata.check(str(tmp_path))[0] == []
    meta["stage4_icdas_model_metadata.json"]["tta"] = "rot90"
    del meta["cavity_localizer_model_metadata.json"]["nms_iou"]
    for k, v in meta.items():
        (tmp_path / k).write_text(json.dumps(v))
    probs = check_metadata.check(str(tmp_path))[0]
    assert any("rot90" in p for p in probs) and any("nms_iou" in p for p in probs)
