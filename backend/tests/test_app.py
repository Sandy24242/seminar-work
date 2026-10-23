"""Streamlit UI tests (headless AppTest): navigation, form validation, questionnaire, models page, results page, downloads."""
import os

import pytest
from streamlit.testing.v1 import AppTest

import config
from standin_models import make_synthetic_photo

APP = os.path.join(config.PROJECT_ROOT, "streamlit_app.py")


@pytest.fixture
def app(standin_dir, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR_CANDIDATES", [standin_dir])
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _texts(at):
    return " ".join(m.value for m in at.markdown) + " ".join(str(c.value) for c in at.caption)


def test_home_page(app):
    assert "Dental Caries AI Assessment System" in _texts(app)
    assert "AI-assisted dental image analysis and caries assessment." in _texts(app)
    assert any(b.label == "Start Assessment" for b in app.button)


def test_start_button_opens_assessment_and_validates(app):
    next(b for b in app.button if b.label == "Start Assessment").click().run()
    assert app.session_state.page == "Assessment"
    assert len(app.radio) >= 7 + 1          # 7 questions + sidebar navigation
    submit = next(b for b in app.button if "Calculate risk score" in b.label)
    submit.click().run()
    errs = " ".join(e.value for e in app.error)
    assert "Age is required" in errs and "Please answer every question" in errs
    assert "risk" not in app.session_state


def test_questionnaire_submission_scores_like_notebook(app):
    next(b for b in app.button if b.label == "Start Assessment").click().run()
    app.text_input[0].input("P-7")
    app.number_input[0].set_value(20)
    picks = ["no", "once a day", "sometimes", "not sure", "6-12 months ago", "yes", "yes"]
    q_radios = [r for r in app.radio if r.label.startswith("**Q")]
    assert len(q_radios) == 7
    for r, v in zip(q_radios, picks):
        r.set_value(v)
    next(b for b in app.button if "Calculate risk score" in b.label).click().run()
    assert not app.exception, app.exception
    risk = app.session_state.risk
    assert (risk["score"], risk["level"]) == (9, "HIGH")
    assert app.session_state.patient["age"] == 20 and app.session_state.patient["patient_id"] == "P-7"
    assert "9 / 13 points" in _texts(app)
    assert any("Dental photograph" in m.value for m in app.markdown)


def test_models_page_lists_models(app):
    app.sidebar.radio[0].set_value("Models & About").run()
    assert not app.exception
    assert any("All five trained models are loaded" in s.value for s in app.success)


def test_models_page_reports_missing_models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR_CANDIDATES", [str(tmp_path)])
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state.page = "Models & About"
    at.run()
    assert not at.exception
    assert any("Required model(s) unavailable" in e.value for e in at.error)


def test_results_page_renders_real_pipeline_output(app, registry):
    from pipeline.questionnaire import score_answers
    from pipeline.report_generator import build_all
    from pipeline.runner import run_pipeline
    rgb = make_synthetic_photo("caries")
    res = run_pipeline(registry, rgb, keep_crops=True); res["input"] = "caries.png"
    crops = res.pop("_crops")
    patient = {"patient_id": None, "age": 41, "assessment_datetime": "2026-10-21 09:00:00"}
    risk = score_answers({"brush_twice": "yes", "junk_sugary": "rarely", "snack_night": "no", "fluoride": "yes",
                          "dental_visit": "within 6 months", "pain": "no", "tobacco": "no"})
    base, files = build_all(res, patient, risk, rgb, registry.summary(), registry.meta.get("fdi"), crops)
    for k, v in dict(page="Results", result=res, crops=crops, rgb=rgb, files=files, base=base, patient=patient,
                     risk=risk, models_dir=config.MODELS_DIR_CANDIDATES[0]).items():
        app.session_state[k] = v
    app.run()
    assert not app.exception, app.exception
    txt = _texts(app)
    for s in ("Assessment results", "Stage 1 — Healthy vs Caries", "Stage 2 — FDI tooth identification", "Cavity localization",
              "Stage 3 — G.V. Black classification", "Stage 4 — ICDAS scoring", "CARIES DETECTED", "LOW RISK"):
        assert s in txt, s
    assert len(app.dataframe) >= 2                          # teeth + lesion tables
    labels = [e.proto.label for e in app.get("download_button")]
    assert sorted(labels) == sorted(["PDF report", "TXT summary", "JSON results", "CSV per lesion", "Annotated image", "Everything (ZIP)"])
