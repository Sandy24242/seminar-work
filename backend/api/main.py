"""
REST API for the React frontend. A thin layer over the existing `pipeline/` package (the same code the Streamlit UI and
the notebook-parity tests use); it adds no model logic of its own.

    GET  /health          process health (does not touch the models)
    GET  /models/status   which trained models actually loaded, and why the others did not
    POST /predict         multipart image upload -> real cascade result (+ annotated image as base64 JPEG)

Models are loaded once at startup from config.resolve_models_dir(). A stage whose model is missing is reported as
UNAVAILABLE by pipeline.runner; nothing is substituted. Uploaded images are processed in memory and never written to disk.
"""
import base64
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

import config  # noqa: E402
from pipeline.annotation import draw_result  # noqa: E402
from pipeline.image_validation import validate_image_bytes  # noqa: E402
from pipeline.model_loader import ModelRegistry  # noqa: E402
from pipeline.report_generator import encode_jpeg  # noqa: E402
from pipeline.runner import STEP_ORDER, STEP_TITLES, run_pipeline, strip_private, ui_status  # noqa: E402

log = logging.getLogger("dcai.api")
logging.basicConfig(level=os.environ.get("DCAI_LOG_LEVEL", "INFO"))

_state = {"registry": None, "models_dir": None}
_infer_lock = threading.Lock()          # Keras models are shared; run one cascade at a time


def load_registry(models_dir=None):
    models_dir = models_dir or config.resolve_models_dir()
    log.info("loading models from %s", models_dir)
    reg = ModelRegistry(models_dir).load_all()
    for k, why in reg.unavailable.items():
        log.warning("model %s unavailable: %s", k, why)
    _state.update(registry=reg, models_dir=models_dir)
    return reg


@asynccontextmanager
async def lifespan(_app):
    if os.environ.get("DCAI_SKIP_MODEL_LOAD") != "1":
        load_registry()
    yield


app = FastAPI(title=config.APP_TITLE, version=config.APP_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get(
        "DCAI_CORS_ORIGINS",       # Vite dev server (5173) and `npm run preview` of the production build (4173)
        "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:4173,http://localhost:4173").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _registry():
    reg = _state["registry"]
    if reg is None:
        reg = load_registry()
    return reg


@app.get("/health")
def health():
    return {"status": "ok", "app": config.APP_TITLE, "version": config.APP_VERSION}


@app.get("/models/status")
def models_status():
    reg = _registry()
    rows = []
    for k in ("stage1", "fdi", "localizer", "gvb", "icdas"):
        rows.append({
            "key": k, "title": config.MODEL_TITLES[k], "file": config.MODEL_FILES[k],
            "required": k in config.REQUIRED_MODELS, "loaded": reg.has(k),
            "reason": None if reg.has(k) else reg.unavailable.get(k, "not loaded"),
            "class_names": reg.meta.get(k, {}).get("class_names") if reg.has(k) else None,
            "load_seconds": reg.load_seconds.get(k),
            "test_standin": reg.has(k) and reg.meta.get(k, {}).get("model") == "TEST STAND-IN",   # tests/standin_models.py
        })
    return {
        "ready": not reg.missing_required,
        "test_standins_loaded": any(r["test_standin"] for r in rows),
        "missing_required": reg.missing_required,
        "models_dir_name": os.path.basename(os.path.normpath(_state["models_dir"] or "")),   # no absolute server path
        "versions": reg.versions,
        "models": rows,
        "limits": {"max_upload_mb": config.MAX_UPLOAD_MB, "allowed_extensions": list(config.ALLOWED_UPLOAD_EXT),
                   "min_side_px": config.MIN_UPLOAD_PX},
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), continue_if_uncertain: bool = Form(False)):
    data = await file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    rgb, info, err = validate_image_bytes(data, file.filename)
    if err:
        raise HTTPException(status_code=422 if len(data) <= config.MAX_UPLOAD_MB * 1024 * 1024 else 413,
                            detail=f"Invalid image: {err}")
    reg = _registry()
    if not reg.has("stage1"):
        return JSONResponse(status_code=503, content={
            "detail": "Stage 1 model is not loaded, so no analysis can be run.",
            "reason": reg.unavailable.get("stage1", "not loaded"), "missing_required": reg.missing_required})
    with _infer_lock:
        res = run_pipeline(reg, rgb, continue_if_uncertain=continue_if_uncertain)
    annotated = draw_result(rgb, res) if res.get("lesions") or res.get("teeth") else None
    out = strip_private(res)
    out["image"] = info
    out["stage_status"] = {k: ui_status(res, k) for k in ("stage1", "stage2", "localization", "stage3", "stage4")}
    out["step_order"] = [s for s in STEP_ORDER if s in out.get("steps", {})]
    out["step_titles"] = STEP_TITLES
    out["annotated_jpeg_base64"] = base64.b64encode(encode_jpeg(annotated)).decode() if annotated is not None else None
    out["disclaimer"] = config.ACADEMIC_DISCLAIMER
    out["test_standins_used"] = any(m.get("model") == "TEST STAND-IN" for m in reg.meta.values())   # never true with real weights
    log.info("predict: %s %sx%s -> %s", info["format"], info["width"], info["height"], out.get("overall"))
    return out
