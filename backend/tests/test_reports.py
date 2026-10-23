"""Report generation: PDF, TXT, JSON, CSV, annotated JPEG and ZIP bundle."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader

from pipeline.questionnaire import score_answers
from pipeline.report_generator import build_all, final_report_lines
from pipeline.runner import run_pipeline
from standin_models import make_synthetic_photo

PATIENT = {"patient_id": "P-001 Test", "age": 34, "assessment_datetime": "2026-10-21 10:00:00"}
RISK = score_answers({"brush_twice": "no", "junk_sugary": "once a day", "snack_night": "sometimes", "fluoride": "not sure",
                      "dental_visit": "6-12 months ago", "pain": "yes", "tobacco": "yes"})


def _bundle(registry, kind):
    rgb = make_synthetic_photo(kind)
    res = run_pipeline(registry, rgb, keep_crops=True)
    res["input"] = f"{kind}.png"
    crops = res.pop("_crops", [])
    return res, build_all(res, PATIENT, RISK, rgb, registry.summary(), registry.meta.get("fdi"), crops)


def test_all_files_for_caries(registry):
    res, (base, files) = _bundle(registry, "caries")
    assert base.startswith("dental_assessment_") and base.endswith("_P-001Test")
    exts = sorted(n[len(base):] for n in files)
    assert exts == ["_all_results.zip", "_annotated.jpg", "_lesions.csv", "_report.pdf", "_results.json", "_summary.txt"]
    pdf = PdfReader(io.BytesIO(files[f"{base}_report.pdf"]))
    text = "\n".join(p.extract_text() for p in pdf.pages)
    for s in ("Dental Caries AI Assessment Report", "HIGH", "9 / 13", "Stage 1", "G.V. Black", "ICDAS", "Limitations",
              "not a clinical diagnosis", "P-001 Test", "2026-10-21 10:00:00"):
        assert s in text, s
    assert len(pdf.pages) >= 2
    txt = files[f"{base}_summary.txt"].decode()
    assert "FINAL DENTAL REPORT" in txt and "Habit-based caries risk: HIGH  (score 9/13)" in txt and "CARIES DETECTED" in txt
    js = json.loads(files[f"{base}_results.json"])
    assert js["habit_risk"]["score"] == 9 and len(js["image_analysis"]["lesions"]) == len(res["lesions"])
    assert js["stage_status"]["stage1"] == "Completed"
    df = pd.read_csv(io.BytesIO(files[f"{base}_lesions.csv"]))
    assert len(df) == len(res["lesions"]) and {"FDI tooth number", "G.V. Black class", "ICDAS score (model-predicted)",
                                                "Tooth detection confidence", "Uncertainty flags"} <= set(df.columns)
    im = Image.open(io.BytesIO(files[f"{base}_annotated.jpg"]))
    assert im.size == (640, 420)
    z = zipfile.ZipFile(io.BytesIO(files[f"{base}_all_results.zip"]))
    assert len(z.namelist()) == 5 and z.testzip() is None


def test_healthy_report_does_not_claim_downstream_stages(registry):
    res, (base, files) = _bundle(registry, "healthy")
    txt = files[f"{base}_summary.txt"].decode()
    assert "Stage 1 = HEALTHY" in txt and "Stages 2–4 were not required" in txt
    pdf_text = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(files[f"{base}_report.pdf"])).pages)
    assert "Skipped" in pdf_text and "No lesion classified" in pdf_text
    assert pd.read_csv(io.BytesIO(files[f"{base}_lesions.csv"])).empty


def test_report_lines_without_questionnaire(registry):
    res = run_pipeline(registry, make_synthetic_photo("caries"))
    lines = final_report_lines(res, None, None)
    assert "   questionnaire skipped" in lines and "   not collected" in lines
