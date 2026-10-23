"""
Dental Caries AI Assessment System — Streamlit website.

Run:  streamlit run streamlit_app.py
The trained models are found automatically (see config.resolve_models_dir); in Google Colab they are read directly from
MyDrive/dentalcaries/pipeline_stage1_to_4_corrected, the notebook's own output folder.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import streamlit as st

import config
from components import image_viewer, questionnaire, result_cards
from components.styles import (card, disclaimer, esc, img_data_uri, inject_css, risk_badge, show_image, step_bar)
from pipeline.annotation import draw_result
from pipeline.model_loader import ModelRegistry
from pipeline.questionnaire import general_advice
from pipeline.report_generator import build_all, lesion_table, teeth_table
from pipeline.runner import STEP_ORDER, STEP_TITLES, run_pipeline, ui_status

st.set_page_config(page_title=config.APP_TITLE, layout="wide", initial_sidebar_state="auto",
                   page_icon=os.path.join(config.ASSETS_DIR, "favicon.png") if os.path.isfile(
                       os.path.join(config.ASSETS_DIR, "favicon.png")) else "🦷")
inject_css()

PAGES = ["Home", "Assessment", "Results", "Models & About"]


# ------------------------------------------------------------------------------------------------ model cache
@st.cache_resource(show_spinner=False)
def get_registry(models_dir):
    """Loaded once per server process and shared read-only by all sessions (patient data is never stored here)."""
    return ModelRegistry(models_dir).load_all()


def models_source_label(path):
    """Describe where the models come from without exposing the full server path."""
    if path == os.path.join(config.PROJECT_ROOT, "models"):
        return "project folder models/"
    if path.startswith(config.DRIVE_BASE):
        return f"Google Drive · dentalcaries/{os.path.basename(path)}"
    return f"custom folder …/{os.path.basename(path.rstrip('/'))}"


def go(page):
    st.session_state.page = page
    st.rerun()


if "page" not in st.session_state:
    st.session_state.page = "Home"

# ------------------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    show_image("logo.png", width=72)
    st.markdown(f"### {config.APP_TITLE}")
    st.caption(config.APP_SUBTITLE)
    choice = st.radio("Navigate", PAGES, index=PAGES.index(st.session_state.page), label_visibility="collapsed")
    if choice != st.session_state.page:
        go(choice)
    st.divider()
    if st.button("Start a new assessment", icon=":material/restart_alt:", width="stretch"):
        for k in ("patient", "risk", "result", "upload", "files", "crops", "rgb", "base"):
            st.session_state.pop(k, None)
        st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
        go("Assessment")
    st.caption("Academic proof of concept — not a clinical diagnosis.")


# ------------------------------------------------------------------------------------------------ pages
def page_home():
    hero_uri = img_data_uri("hero_dental_ai.jpg")
    c1, c2 = st.columns([1.05, 1], vertical_alignment="center")
    with c1:
        st.markdown(f'<div class="dc-hero"><h1>{esc(config.APP_TITLE)}</h1><div class="dc-sub">{esc(config.APP_SUBTITLE)}</div>'
                    '<div class="dc-muted" style="font-size:1rem">Answer a short oral-habits questionnaire, upload an intraoral '
                    'photograph and run the four-stage deep-learning pipeline: caries screening, FDI tooth numbering, lesion '
                    'localization, G.V. Black classification and ICDAS scoring — with a downloadable report.</div>'
                    '<div style="margin-top:.8rem"><span class="dc-pill">TensorFlow / Keras CNNs</span><span class="dc-pill">FDI numbering</span>'
                    '<span class="dc-pill">G.V. Black I–VI</span><span class="dc-pill">ICDAS 1–6</span><span class="dc-pill">PDF report</span></div></div>',
                    unsafe_allow_html=True)
        st.write("")
        if st.button("Start Assessment", type="primary", icon=":material/arrow_forward:"):
            go("Assessment")
    with c2:
        if hero_uri:
            st.markdown(f'<img src="{hero_uri}" alt="Illustration of AI-assisted dental image analysis" '
                        f'style="width:100%;border-radius:22px;box-shadow:0 10px 36px rgba(11,37,69,.12)"/>', unsafe_allow_html=True)

    st.markdown("## The four-stage AI pipeline")
    show_image("workflow_stages.png", width="stretch")
    stages = [("icon_stage1.png", "Stage 1 — Healthy vs Caries", "CNN 1 screens the whole photograph. Healthy → the caries-specific "
               "stages stop. Uncertain (close to the validated threshold or low confidence) → stop and retake the photo."),
              ("icon_stage2.png", "Stage 2 — FDI tooth identification", "A grid detector finds each tooth and predicts its two-digit FDI "
               "number. Low-confidence or duplicated numbers are shown as “Unknown / uncertain”."),
              ("icon_localization.png", "Cavity localization", "A separate, class-agnostic localizer marks where lesions are. Each lesion "
               "is linked to the tooth box that covers most of it."),
              ("icon_stage3.png", "Stage 3 — G.V. Black", "CNN 2 classifies a context crop around every lesion into G.V. Black "
               "Classes I–VI, with its own confidence."),
              ("icon_stage4.png", "Stage 4 — ICDAS", "CNN 3 scores a tooth-scale window around every lesion on the ICDAS 1–6 scale "
               "(ICDAS 0 is not modelled — that is Stage 1’s job).")]
    cols = st.columns(2)
    for i, (ic, t, d) in enumerate(stages):
        with cols[i % 2]:
            card(f"<h4>{esc(t)}</h4><div class='dc-muted'>{esc(d)}</div>", icon=ic)

    st.markdown("## How an assessment works")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        for n, (t, d) in enumerate([("Patient details", "Optional name or ID and the patient’s age. Date and time are recorded automatically."),
                                    ("Oral-habits questionnaire", "Seven questions → a habit-based risk score out of 13 with personalised advice."),
                                    ("Upload a dental photo", "JPG, JPEG or PNG. The file is validated, EXIF-rotated and kept in memory only."),
                                    ("AI analysis", "The real Stage 1–4 models run with live progress for every step."),
                                    ("Results & report", "Stage cards, annotated image, per-lesion table and PDF / TXT / JSON / CSV downloads.")], 1):
            card(f"<h4>{n}. {esc(t)}</h4><div class='dc-muted'>{esc(d)}</div>")
    with c2:
        show_image("tooth_anatomy.png", caption="Tooth anatomy (illustration). Caries starts in the enamel and can progress into dentin and pulp.",
                   width="stretch")

    st.markdown("## Questionnaire-based risk assessment")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        card("<h4>Habit-based risk — kept separate from the AI</h4><div class='dc-muted'>Brushing, sugar intake, night-time snacking, "
             "fluoride toothpaste, dental visits, pain / sensitivity and tobacco use give 0–13 points: "
             f"{risk_badge('LOW')} 0–3 &nbsp; {risk_badge('MODERATE')} 4–7 &nbsp; {risk_badge('HIGH')} 8–13.<br><br>"
             "The score and advice come only from the answers. They are reported next to the image analysis but never change "
             "any CNN prediction and are not a diagnosis.</div>", icon="icon_questionnaire.png")
    with c2:
        show_image("clinical_caries_example.jpg", caption="Example intraoral photograph with visible caries "
                   "(from this project’s dataset, as shown in the notebook run).", width="stretch")
    disclaimer()


def page_assessment():
    done = set()
    if st.session_state.get("risk"):
        done |= {1, 2}
    if st.session_state.get("result"):
        done |= {3, 4, 5}
    step_bar(3 if 2 in done else 1, done)
    st.markdown("# Patient assessment")

    if not st.session_state.get("risk") or st.session_state.get("edit_form"):
        out = questionnaire.patient_and_questionnaire_form()
        if out:
            st.session_state.patient, st.session_state.risk = out
            st.session_state.edit_form = False
            st.session_state.pop("result", None)
            st.rerun()
        st.info("Complete the patient details and all 7 questions to unlock the dental photo upload.")
        return

    p, risk = st.session_state.patient, st.session_state.risk
    c1, c2 = st.columns([4, 1])
    c1.markdown(f"**Patient:** {esc(p.get('patient_id') or '— (no ID)')} · **Age:** {p['age']} · **Assessment:** {esc(p['assessment_datetime'])}")
    if c2.button("Edit answers", icon=":material/edit:"):
        st.session_state.edit_form = True
        st.rerun()
    st.markdown("### Questionnaire result")
    questionnaire.show_risk_result(risk)

    st.markdown("### Dental photograph")
    rgb, info = image_viewer.upload_panel()
    with st.expander("Advanced pipeline options (defaults = the notebook's validated settings)"):
        cont = st.checkbox("Continue the caries stages even when Stage 1 is UNCERTAIN (results are flagged)", value=False,
                           help="Notebook option continue_if_uncertain. Default off: an uncertain Stage 1 stops the cascade.")
        lower = st.checkbox("Use a lower localizer score threshold (adds low-certainty regions, flagged as such)", value=False)
        thr = st.slider("Localizer score threshold", 0.05, 0.95, 0.20, 0.05, disabled=not lower)
    if st.button("Analyze Dental Image", type="primary", icon=":material/biotech:", disabled=rgb is None, width="stretch"):
        analyse(rgb, info, continue_if_uncertain=cont, score_threshold=thr if lower else None)


def analyse(rgb, info, continue_if_uncertain=False, score_threshold=None):
    progress = st.progress(0.0, text="Starting…")
    box = st.status("Running the AI pipeline…", expanded=True)
    n_steps = len(STEP_ORDER)
    done_steps = []
    icons = {"done": "✅", "uncertain": "⚠️", "skipped": "⏭️", "unavailable": "🚫", "failed": "❌"}

    def cb(step, state, msg):
        if state == "running":
            progress.progress(len(done_steps) / n_steps, text=f"{STEP_TITLES[step]}: {msg}…")
            return
        done_steps.append(step)
        box.write(f"{icons.get(state, '•')} **{STEP_TITLES[step]}** — {msg}")
        progress.progress(min(len(done_steps) / n_steps, 1.0), text=f"{STEP_TITLES[step]}: {state}")

    cb("validation", "done", f"{info['filename']} · {info['width']}×{info['height']} px · {info['format']} · EXIF-corrected RGB")
    cb("models", "running", "loading the trained models (first run only)")
    models_dir = config.resolve_models_dir()
    registry = get_registry(models_dir)
    loaded = [k for k in ("stage1", "fdi", "localizer", "gvb", "icdas") if registry.has(k)]
    missing = [k for k in ("stage1", "fdi", "localizer", "gvb", "icdas") if not registry.has(k)]
    cb("models", "done" if not missing else ("failed" if "stage1" in missing else "unavailable"),
       f"{len(loaded)}/5 loaded from {models_source_label(models_dir)}" + (f"; unavailable: {', '.join(missing)}" if missing else ""))
    if "stage1" in missing:
        box.update(label="Models not available", state="error")
        st.error("The trained Stage 1 model could not be loaded, so the photo cannot be analysed. "
                 f"Reason: {registry.unavailable.get('stage1')}. See **Models & About** and the Colab guide (step: model files).")
        return
    try:
        res = run_pipeline(registry, rgb, score_threshold=score_threshold, continue_if_uncertain=continue_if_uncertain,
                           keep_crops=True, progress=cb)
    except Exception as e:  # defensive: run_pipeline already turns stage errors into FAILED statuses
        box.update(label="Analysis failed", state="error")
        st.exception(e)
        return
    res["input"] = info["filename"]
    crops = res.pop("_crops", [])
    cb("report", "running", "building PDF, TXT, JSON, CSV and annotated image")
    try:
        base, files = build_all(res, st.session_state.patient, st.session_state.risk, rgb, registry.summary(),
                                registry.meta.get("fdi"), crops)
        cb("report", "done", f"{len(files)} files ready")
    except Exception as e:
        base, files = None, {}
        cb("report", "failed", f"report generation failed ({type(e).__name__}: {e})")
    st.session_state.update(result=res, crops=crops, rgb=rgb, files=files, base=base, models_dir=models_dir)
    box.update(label="Analysis complete — opening results", state="complete", expanded=False)
    go("Results")


def page_results():
    res = st.session_state.get("result")
    if not res:
        step_bar(1, set())
        st.markdown("# Results")
        st.info("No analysis yet. Complete the assessment and analyse a dental photograph first.")
        if st.button("Go to the assessment", type="primary"):
            go("Assessment")
        return
    step_bar(5, {1, 2, 3, 4, 5})
    p, risk, rgb, crops = st.session_state.patient, st.session_state.risk, st.session_state.rgb, st.session_state.get("crops", [])
    registry = get_registry(st.session_state.get("models_dir") or config.resolve_models_dir())
    st.markdown("# Assessment results")

    # A. summary
    s1 = res.get("stage1")
    overall_status = ("Failed" if s1 is None else "Uncertain" if s1["prediction"] == "UNCERTAIN" else "Completed")
    c1, c2, c3 = st.columns([1.1, 1, 1.3])
    with c1:
        card(f'<h4>Patient</h4><div class="dc-kv"><b>Name / ID</b><span>{esc(p.get("patient_id") or "—")}</span>'
             f'<b>Age</b><span>{p["age"]} years</span><b>Assessed</b><span>{esc(p["assessment_datetime"])}</span>'
             f'<b>Image</b><span>{esc(res.get("input", "—"))}</span></div>')
    with c2:
        card(f'<h4>Questionnaire</h4>{risk_badge(risk["level"])}<div class="dc-big" style="margin-top:.4rem">{risk["score"]} / {risk["max"]}</div>'
             f'<div class="dc-muted">habit-based — separate from the AI</div>', icon="icon_questionnaire.png")
    with c3:
        card(f'<h4>Overall pipeline result &nbsp;{result_cards.badge(overall_status)}</h4><div class="dc-big">{esc(res.get("overall", "—"))}</div>'
             + (f'<div class="dc-muted">Stage 1: {esc(s1["prediction"])} · {s1["confidence"] * 100:.1f}% confidence</div>' if s1 else ""),
             icon="icon_stage1.png")

    # B. stage cards
    st.markdown("### Stage results")
    result_cards.stage_cards(res)

    # C. images
    if s1 is not None:
        st.markdown("### Original and annotated image")
        image_viewer.comparison(rgb, draw_result(rgb, res))

    # D. tables
    tt = teeth_table(res)
    if len(tt):
        st.markdown("### Detected teeth (Stage 2)")
        st.dataframe(tt, hide_index=True, width="stretch")
    st.markdown("### Per-lesion results")
    lt = lesion_table(res, extended=True)
    if len(lt):
        st.dataframe(lt, hide_index=True, width="stretch")
        result_cards.lesion_details(res, crops)
    else:
        reason = (res.get("stage3") or {}).get("reason") or res.get("overall")
        st.info(f"No lesion classified — G.V. Black and ICDAS are not assigned. Reason: {reason}")

    # E. recommendations
    st.markdown("### Recommendations")
    card("".join(f"<div>• {esc(a)}</div>" for a in general_advice(res, risk)), icon="icon_questionnaire.png")

    # downloads
    st.markdown("### Download the report")
    files, base = st.session_state.get("files") or {}, st.session_state.get("base")
    if not files:
        st.error("Report files could not be generated for this run (see Technical information).")
    else:
        mimes = {".pdf": "application/pdf", ".txt": "text/plain", ".json": "application/json", ".csv": "text/csv",
                 ".jpg": "image/jpeg", ".zip": "application/zip"}
        labels = {"_report.pdf": ("PDF report", ":material/picture_as_pdf:"), "_summary.txt": ("TXT summary", ":material/description:"),
                  "_results.json": ("JSON results", ":material/data_object:"), "_lesions.csv": ("CSV per lesion", ":material/table:"),
                  "_annotated.jpg": ("Annotated image", ":material/image:"), "_all_results.zip": ("Everything (ZIP)", ":material/folder_zip:")}
        cols = st.columns(3)
        for i, (suffix, (lab, ic)) in enumerate(labels.items()):
            name = f"{base}{suffix}"
            if name in files:
                cols[i % 3].download_button(lab, files[name], file_name=name, mime=mimes[os.path.splitext(name)[1]], icon=ic,
                                            width="stretch", type="primary" if suffix == "_report.pdf" else "secondary",
                                            key=f"dl_{suffix}")
        st.caption("Files are generated in memory for this session only; nothing about the patient is stored on the server.")

    # F. technical
    result_cards.technical_info(res, registry, models_source_label(st.session_state.get("models_dir") or config.resolve_models_dir()))
    disclaimer()


def page_about():
    st.markdown("# Models & About")
    models_dir = config.resolve_models_dir()
    with st.spinner("Loading models (first time only)…"):
        registry = get_registry(models_dir)
    st.markdown(f"**Model folder:** {esc(models_source_label(models_dir))} · **TensorFlow** {registry.versions.get('tensorflow', '—')}"
                f" · **Keras** {registry.versions.get('keras', '—')}")
    df = pd.DataFrame(registry.summary())
    st.dataframe(df, hide_index=True, width="stretch")
    if registry.missing_required:
        st.error("Required model(s) unavailable: " + ", ".join(registry.missing_required) +
                 ". Put the trained .keras files and their *_metadata.json files in the model folder (see README / Colab guide). "
                 "The website never substitutes a random or placeholder model.")
    elif not registry.has("fdi"):
        st.warning("Stage 2 FDI detector unavailable — the cascade still runs and reports tooth numbers as Unavailable.")
    else:
        st.success("All five trained models are loaded.")
    if st.button("Reload models", icon=":material/refresh:"):
        get_registry.clear()
        st.rerun()
    st.markdown("### About this project")
    card("<div class='dc-muted'>Academic project: a multi-CNN cascade trained in Google Colab "
         "(Dental_Caries_Pipeline_Stage1_to_Stage4_with_FDI_Stage2.ipynb). The website reuses the notebook's preprocessing, "
         "thresholds, class mappings, lesion→tooth association and report logic unchanged; it does not retrain any model.</div>")
    disclaimer()


{"Home": page_home, "Assessment": page_assessment, "Results": page_results, "Models & About": page_about}[st.session_state.page]()
