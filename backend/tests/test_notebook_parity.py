"""
Prediction consistency with the ORIGINAL notebook.

The notebook's own code cells (Section 3 configuration, 6.1 preprocessing contract, 6.6 prediction functions,
6.7 TTA, 76 run_pipeline / draw_result, 82 questionnaire + report) are executed from
reference/Dental_Caries_Pipeline_Stage1_to_Stage4_with_FDI_Stage2.ipynb. Both implementations are then given the SAME
loaded models and the SAME images, and their outputs must be identical. With the trained models this test can be re-run
by pointing DCAI_PARITY_MODELS_DIR at the real model folder (see README).
"""
import contextlib
import glob
import io
import json
import os

import numpy as np
import pytest

import config
from pipeline import report_generator as RG
from pipeline.annotation import draw_result
from pipeline.image_validation import validate_image_bytes
from pipeline.model_loader import ModelRegistry
from pipeline.questionnaire import RISK_QUESTIONS, score_answers
from pipeline.runner import run_pipeline
from standin_models import make_synthetic_photo

NB = os.path.join(os.path.dirname(config.PROJECT_ROOT), "notebooks", "Dental_Caries_Pipeline_Stage1_to_Stage4_with_FDI_Stage2.ipynb")
HEADERS = ["SECTION 3 — CONFIGURATION", "SECTION 6.1 —", "SECTION 6.6 —", "SECTION 6.7 —", "SECTION 76 —", "SECTION 82 —"]


@pytest.fixture(scope="module")
def nb(tmp_path_factory, request):
    import cv2, hashlib, re, time  # noqa: E401
    from collections import Counter, defaultdict
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess
    from PIL import Image, ImageOps

    cells = ["".join(c["source"]) for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
    tmp = str(tmp_path_factory.mktemp("nb"))
    ns = dict(os=os, sys=__import__("sys"), io=io, json=json, re=re, time=time, hashlib=hashlib, math=__import__("math"),
              np=np, pd=pd, cv2=cv2, tf=tf, keras=keras, layers=layers, Image=Image, ImageOps=ImageOps,
              mobilenet_v2_preprocess=mobilenet_v2_preprocess, Counter=Counter, defaultdict=defaultdict,
              EarlyStopping=EarlyStopping, ReduceLROnPlateau=ReduceLROnPlateau, ModelCheckpoint=ModelCheckpoint,
              IN_COLAB=False, MODEL_PATHS={}, MODEL_META_PATHS={"fdi": os.path.join(tmp, "none.json")},
              METADATA_PATH=os.path.join(tmp, "pipeline_metadata.json"), plt=None)
    for h in HEADERS:
        src = next(c for c in cells if h in c)
        if h.startswith("SECTION 3"):
            src = src.replace('"/content/', f'"{tmp}/content/')        # keep the notebook's makedirs inside tmp
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, f"notebook:{h}", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def parity_registry(registry):
    d = os.environ.get("DCAI_PARITY_MODELS_DIR")
    if d:
        reg = ModelRegistry(d).load_all()
        assert not reg.missing_required, reg.unavailable
        return reg
    return registry


def _images():
    out = {k: make_synthetic_photo(k) for k in ("caries", "healthy", "uncertain", "caries_no_lesion", "spanning", "gum_lesion")}
    ex = os.path.join(config.ASSETS_DIR, "clinical_caries_example.jpg")
    if os.path.isfile(ex):
        from pipeline.preprocessing import load_rgb
        out["clinical_example"] = load_rgb(ex)
    rng = np.random.default_rng(7)
    out["noise_portrait"] = rng.integers(0, 255, (777, 431, 3)).astype(np.uint8)
    for p in sorted(glob.glob(os.path.join(os.environ.get("DCAI_PARITY_IMAGES", "/nonexistent"), "*")))[:20]:
        from pipeline.preprocessing import load_rgb
        out[os.path.basename(p)] = load_rgb(p)
    return out


def assert_same(a, b, path="res"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {set(a) ^ set(b)}"
        for k in a:
            assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, float):
        assert b == pytest.approx(a, rel=1e-9, abs=1e-12), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_preprocessing_functions_identical(nb):
    from pipeline import preprocessing as P
    rng = np.random.default_rng(1)
    for shape in ((420, 640, 3), (777, 431, 3), (100, 100, 3)):
        rgb = rng.integers(0, 255, shape).astype(np.uint8)
        H, W = shape[:2]
        assert np.array_equal(P.s1_prepare(rgb, 224), nb["s1_prepare"](rgb, 224))
        assert np.array_equal(P.loc_prepare(rgb, 520, 416), nb["loc_prepare"](rgb, 520, 416))
        assert np.array_equal(P.loc_prepare(rgb, 640, 512), nb["loc_prepare"](rgb, 640, 512))
        for _ in range(25):
            x1, y1 = rng.uniform(-0.05, 0.9, 2); box = np.array([x1, y1, x1 + rng.uniform(0.005, 0.4), y1 + rng.uniform(0.005, 0.4)], np.float32)
            for pad in (0.0, 0.5, 1.0):
                assert np.array_equal(P.gvb_prepare_crop(rgb, box, pad, 224, 48), nb["gvb_prepare_crop"](rgb, box, pad, 224, 48))
            w = {"mult": 1.5, "min_frac_w": 0.33, "min_frac_h": 0.55}
            assert np.array_equal(P.icdas_prepare_from_box(rgb, box, 288, w), nb["icdas_prepare_from_box"](rgb, box, 288, w))
            assert P.to_px(box, W, H) == nb["to_px"](box, W, H)
    raw = rng.normal(0, 3, (32, 32, 31)).astype(np.float32)
    for a, b in zip(P.decode_fdi_detections(raw, 0.5, 0.45, 40), nb["decode_fdi_detections"](raw, 0.5, 0.45, 40)):
        assert np.array_equal(a, b)
    for a, b in zip(P.decode_detections(raw[..., :5], 0.4, 0.4, 10), nb["decode_detections"](raw[..., :5], 0.4, 0.4, 10)):
        assert np.array_equal(a, b)


def test_run_pipeline_identical_to_notebook(nb, parity_registry):
    nb["PIPELINE"]["models"].clear(); nb["PIPELINE"]["meta"].clear()
    nb["PIPELINE"]["models"].update(parity_registry.models); nb["PIPELINE"]["meta"].update(parity_registry.meta)
    n_lesions = 0
    for name, rgb in _images().items():
        for kw in ({}, {"continue_if_uncertain": True}, {"score_threshold": 0.2}):
            ref = nb["run_pipeline"](rgb, **kw)
            mine = run_pipeline(parity_registry, rgb, **kw)
            mine.pop("steps")
            assert_same(ref, mine, f"{name}{kw}")
            n_lesions += len(ref["lesions"])
            assert np.array_equal(nb["draw_result"](rgb, ref), draw_result(rgb, mine)), f"annotation differs for {name}"
            # report pieces (notebook Section 82)
            ref["input"] = mine["input"] = name
            assert nb["lesion_table"](ref).equals(RG.lesion_table(mine))
            if len(ref["teeth"]):
                assert nb["teeth_table"](ref).equals(RG.teeth_table(mine))
            patient = {"patient_id": "X", "age": 30}
            risk = score_answers({q["key"]: list(q["options"])[-1] for q in RISK_QUESTIONS})
            assert nb["final_report_lines"](ref, patient, risk) == RG.final_report_lines(mine, patient, risk, parity_registry.meta.get("fdi"))
    assert n_lesions > 0


def test_questionnaire_identical_to_notebook(nb):
    import itertools
    keys = [q["key"] for q in RISK_QUESTIONS]
    assert nb["RISK_QUESTIONS"] == RISK_QUESTIONS and nb["RISK_LEVELS"] == [(3, "LOW"), (7, "MODERATE"), (13, "HIGH")]
    for combo in itertools.product(*[list(q["options"]) for q in RISK_QUESTIONS]):
        ans = dict(zip(keys, combo))
        with contextlib.redirect_stdout(io.StringIO()):
            ref = nb["ask_oral_habits"](answers=ans)
        assert ref == score_answers(ans)


def test_upload_validation_matches_notebook(nb, tmp_path):
    from PIL import Image
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 255, (120, 90, 3)).astype(np.uint8)
    cases = {}
    b = io.BytesIO(); Image.fromarray(arr).save(b, format="PNG"); cases["ok.png"] = b.getvalue()
    exif = Image.Exif(); exif[0x0112] = 8
    b = io.BytesIO(); Image.fromarray(arr).save(b, format="JPEG", exif=exif); cases["rot.jpg"] = b.getvalue()
    cases["bad.jpg"] = b"garbage"
    b = io.BytesIO(); Image.fromarray(arr[:30, :30]).save(b, format="PNG"); cases["tiny.png"] = b.getvalue()
    for name, data in cases.items():
        p = tmp_path / name; p.write_bytes(data)
        ref_rgb, ref_err = nb["validate_input_image"](str(p))
        rgb, _, err = validate_image_bytes(data, name)
        assert (ref_err is None) == (err is None), name
        if ref_rgb is not None:
            assert np.array_equal(ref_rgb, rgb)
