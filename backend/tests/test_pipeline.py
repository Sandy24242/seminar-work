"""Cascade behaviour with stand-in models: stopping rules, associations, uncertainty, missing / broken models."""
import json
import os
import shutil

import numpy as np
import pytest

from pipeline.model_loader import ModelRegistry
from pipeline.runner import run_pipeline, ui_status
from standin_models import FILES, make_synthetic_photo, write_standin_models


def run(reg, kind, **kw):
    return run_pipeline(reg, make_synthetic_photo(kind), **kw)


def test_models_load_with_keras_load_model(registry):
    assert set(registry.models) == {"stage1", "fdi", "localizer", "gvb", "icdas"}
    assert registry.models["fdi"].output_shape[1:] == (32, 32, 31)
    assert registry.models["localizer"].output_shape[1:] == (26, 26, 5)


def test_caries_full_cascade_and_association(registry):
    res = run(registry, "caries")
    assert res["stage1"]["prediction"] == "CARIES"
    assert res["overall"] == "CARIES DETECTED"
    assert [t["fdi"] for t in res["teeth"]] == ["11", "21", "12"]          # sorted left to right
    assert [l["fdi_tooth"] for l in res["lesions"]] == ["11", "12"]
    assert all(l["fdi_status"] == "ASSIGNED" for l in res["lesions"])
    for l in res["lesions"]:
        assert l["gvb_display"].startswith("Class ") and l["icdas"] in range(1, 7)
        assert abs(sum(l["gvb_probabilities"].values()) - 1) < 1e-5 and abs(sum(l["icdas_probabilities"].values()) - 1) < 1e-5
        assert "ICDAS 0" not in l["icdas_probabilities"]
    assert all(ui_status(res, k) in ("Completed", "Uncertain") for k in ("stage1", "stage2", "localization", "stage3", "stage4"))


def test_healthy_stops_cascade(registry):
    res = run(registry, "healthy")
    assert res["stage1"]["prediction"] == "HEALTHY" and res["overall"].startswith("HEALTHY")
    assert res["lesions"] == [] and res["teeth"] == []
    for k in ("stage2", "localization", "stage3", "stage4"):
        assert res[k]["status"] == "NOT RUN" and "Stage 1 = HEALTHY" in res[k]["reason"]
        assert ui_status(res, k) == "Skipped"


def test_uncertain_stops_unless_requested(registry):
    res = run(registry, "uncertain")
    assert res["stage1"]["prediction"] == "UNCERTAIN" and res["overall"].startswith("UNCERTAIN")
    assert res["stage2"]["status"] == "NOT RUN" and ui_status(res, "stage1") == "Uncertain"
    res2 = run(registry, "uncertain", continue_if_uncertain=True)
    assert res2["stage2"]["status"] == "RAN" and res2["overall"].startswith("UNCERTAIN (Stage 1) — cascade continued")


def test_stage1_decision_rule_uses_metadata(registry):
    from pipeline.stage1 import stage1_decide
    m = registry.meta["stage1"]
    assert stage1_decide(0.99, m)["prediction"] == "CARIES"
    assert stage1_decide(0.01, m)["prediction"] == "HEALTHY"
    near = stage1_decide(m["threshold"] + 0.03, m)
    assert near["prediction"] == "UNCERTAIN" and near["leaning"] == "CARIES" and "within" in near["reason"]
    low = stage1_decide(0.45, m)                       # HEALTHY with confidence 55 % < 60 %
    assert low["prediction"] == "UNCERTAIN" and "confidence" in low["reason"]


def test_no_lesion_no_fabricated_classification(registry):
    res = run(registry, "caries_no_lesion")
    assert res["overall"] == "CARIES DETECTED — but no lesion localized above the threshold"
    assert res["lesions"] == []
    assert res["stage3"]["status"] == "NOT RUN" and res["stage4"]["status"] == "NOT RUN"
    assert res["stage2"]["status"] == "RAN"


def test_ambiguous_and_no_tooth(registry):
    amb = run(registry, "spanning")
    assert any(l["fdi_status"] == "AMBIGUOUS" and l["fdi_tooth"] is None and l["fdi_display"].startswith("Uncertain:")
               for l in amb["lesions"])
    gum = run(registry, "gum_lesion")
    assert any(l["fdi_status"] == "NO_TOOTH" and l["fdi_display"] == "Unknown / uncertain" for l in gum["lesions"])


def test_lower_threshold_flags_low_certainty(registry):
    res = run(registry, "caries", score_threshold=0.05)
    assert res["localization"]["score_threshold_used"] == 0.05
    assert any(l["below_validated_threshold"] for l in res["lesions"])


def test_progress_callback_reports_real_steps(registry):
    seen = []
    run(registry, "caries", progress=lambda s, st, m: seen.append((s, st)))
    done = [s for s, st in seen if st != "running"]
    assert done == ["stage1", "stage2", "localization", "association", "stage3", "stage4"]
    assert seen.index(("stage1", "running")) < seen.index(("stage1", "done"))


def _partial(tmp_path, stages, **over):
    d = write_standin_models(str(tmp_path / "m"), stages=stages, meta_overrides=over or None)
    return ModelRegistry(d).load_all()


def test_missing_fdi_detector(tmp_path):
    reg = _partial(tmp_path, ("stage1", "localizer", "gvb", "icdas"))
    res = run(reg, "caries")
    assert res["stage2"]["status"] == "UNAVAILABLE" and "not found" in res["stage2"]["reason"]
    assert res["teeth"] == [] and res["lesions"]
    assert all(l["fdi_display"] == "Unavailable" and l["fdi_tooth"] is None for l in res["lesions"])
    assert ui_status(res, "stage2") == "Unavailable"


def test_missing_classifiers_are_unavailable_not_invented(tmp_path):
    reg = _partial(tmp_path, ("stage1", "fdi", "localizer"))
    res = run(reg, "caries")
    assert res["stage3"]["status"] == "UNAVAILABLE" and res["stage4"]["status"] == "UNAVAILABLE"
    for l in res["lesions"]:
        assert l["gvb_confidence"] is None and l["icdas"] is None and l["gvb_display"] == "Unavailable"


def test_missing_localizer(tmp_path):
    reg = _partial(tmp_path, ("stage1", "fdi", "gvb", "icdas"))
    res = run(reg, "caries")
    assert res["localization"]["status"] == "UNAVAILABLE" and res["lesions"] == []
    assert "localization unavailable" in res["overall"]


def test_missing_stage1_cannot_analyse(tmp_path):
    reg = _partial(tmp_path, ("fdi", "localizer", "gvb", "icdas"))
    res = run(reg, "caries")
    assert res["stage1"] is None and res["overall"].startswith("NOT ANALYSED")
    assert ui_status(res, "stage1") == "Unavailable"


def test_missing_metadata_and_corrupted_model(tmp_path):
    d = write_standin_models(str(tmp_path / "m"))
    os.remove(os.path.join(d, FILES["gvb"].replace(".keras", "_metadata.json")))
    with open(os.path.join(d, FILES["icdas"]), "wb") as f:
        f.write(b"not a keras zip")
    reg = ModelRegistry(d).load_all()
    assert "metadata" in reg.unavailable["gvb"]
    assert "could not be loaded" in reg.unavailable["icdas"]
    assert set(reg.missing_required) == {"gvb", "icdas"}


def test_empty_folder(tmp_path):
    reg = ModelRegistry(str(tmp_path)).load_all()
    assert set(reg.missing_required) == {"stage1", "localizer", "gvb", "icdas"}
    assert not reg.models


def test_gvb_uncertainty_threshold_from_metadata(tmp_path):
    reg = _partial(tmp_path, ("stage1", "fdi", "localizer", "gvb", "icdas"),
                   gvb={"confidence_threshold": 1.01}, icdas={"confidence_threshold": 1.01})
    res = run(reg, "caries")
    assert all(l["gvb_uncertain"] and l["icdas_uncertain"] for l in res["lesions"])
    assert ui_status(res, "stage3") == "Uncertain" and ui_status(res, "stage4") == "Uncertain"


def test_class_mapping_comes_from_metadata(tmp_path, registry):
    base = run(registry, "caries")
    names = ["GVB_Class_VI", "GVB_Class_V", "GVB_Class_IV", "GVB_Class_III", "GVB_Class_II", "GVB_Class_I"]
    reg = _partial(tmp_path, ("stage1", "fdi", "localizer", "gvb", "icdas"), gvb={"class_names": names},
                   icdas={"scores": [6, 5, 4, 3, 2, 1]})
    res = run(reg, "caries")
    for a, b in zip(base["lesions"], res["lesions"]):
        assert a["gvb_confidence"] == pytest.approx(b["gvb_confidence"])
        ia = ["I", "II", "III", "IV", "V", "VI"].index(a["gvb_display"].split()[-1])
        assert b["gvb_display"] == "Class " + ["VI", "V", "IV", "III", "II", "I"][ia]
        assert b["icdas"] == 7 - a["icdas"]


def test_result_is_json_serialisable(registry):
    from pipeline.runner import strip_private
    res = run(registry, "caries", keep_crops=True)
    assert "_crops" in res and len(res["_crops"]) == len(res["lesions"])
    json.dumps(strip_private(res))
