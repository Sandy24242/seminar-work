"""FastAPI layer: health, model status (loaded and missing), prediction, upload validation.

Model-dependent tests use the TEST-ONLY stand-in models from standin_models.py (not the trained models)."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from standin_models import make_synthetic_photo, write_standin_models

from api import main


def _png(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client_with(monkeypatch):
    def make(models_dir):
        monkeypatch.setenv("DCAI_SKIP_MODEL_LOAD", "1")
        main.load_registry(models_dir)
        return TestClient(main.app)
    return make


def test_health(client_with, tmp_path):
    c = client_with(str(tmp_path))
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_status_reports_missing_models(client_with, tmp_path):
    body = client_with(str(tmp_path)).get("/models/status").json()
    assert body["ready"] is False
    assert set(body["missing_required"]) == {"stage1", "localizer", "gvb", "icdas"}
    assert all(not m["loaded"] and m["reason"] for m in body["models"])
    assert str(tmp_path) not in str(body)                       # no absolute server paths


def test_predict_refuses_without_stage1(client_with, tmp_path):
    c = client_with(str(tmp_path))
    r = c.post("/predict", files={"file": ("x.png", _png(make_synthetic_photo("caries")), "image/png")})
    assert r.status_code == 503
    assert "Stage 1" in r.json()["detail"]


def test_status_and_predict_with_standins(client_with, standin_dir):
    c = client_with(standin_dir)
    st = c.get("/models/status").json()
    assert st["ready"] is True and all(m["loaded"] for m in st["models"])
    r = c.post("/predict", files={"file": ("mouth.png", _png(make_synthetic_photo("caries")), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stage1"]["prediction"] == "CARIES"
    assert body["lesions"] and all("box_xyxy_px" in l for l in body["lesions"])
    assert body["annotated_jpeg_base64"]
    assert body["stage_status"]["stage1"] == "Completed"


def test_predict_healthy_stops_cascade(client_with, standin_dir):
    c = client_with(standin_dir)
    body = c.post("/predict", files={"file": ("h.png", _png(make_synthetic_photo("healthy")), "image/png")}).json()
    assert body["stage1"]["prediction"] == "HEALTHY"
    assert body["lesions"] == [] and body["stage_status"]["stage3"] == "Skipped"


def test_partial_models_report_unavailable_stage(client_with, tmp_path):
    d = write_standin_models(str(tmp_path), stages=("stage1", "localizer"))
    c = client_with(d)
    body = c.post("/predict", files={"file": ("m.png", _png(make_synthetic_photo("caries")), "image/png")}).json()
    assert body["stage_status"]["stage3"] == "Unavailable"
    assert body["stage_status"]["stage2"] == "Unavailable"
    assert all(l["gvb_class"] is None for l in body["lesions"])


@pytest.mark.parametrize("name,data,ctype", [
    ("notes.txt", b"hello", "text/plain"),
    ("fake.png", b"not an image at all", "image/png"),
    ("empty.jpg", b"", "image/jpeg"),
    ("tiny.png", _png(np.zeros((10, 10, 3), np.uint8)), "image/png"),
])
def test_invalid_uploads(client_with, tmp_path, name, data, ctype):
    r = client_with(str(tmp_path)).post("/predict", files={"file": (name, data, ctype)})
    assert r.status_code == 422
    assert r.json()["detail"].startswith("Invalid image:")
