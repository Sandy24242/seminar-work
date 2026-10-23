"""
Final assessment report: TXT (notebook Section 82 `final_report_lines`, ported), JSON (structured results), CSV
(per-lesion table), annotated JPEG and a professional PDF (ReportLab). Everything is built in memory — nothing about the
patient is written to the server's disk.
"""
import csv
import io
import json
import os
import time
import uuid
import zipfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

import config
from pipeline.annotation import draw_result
from pipeline.questionnaire import RISK_QUESTIONS, general_advice
from pipeline.runner import STEP_ORDER, STEP_TITLES, strip_private, ui_status

LINE = "=" * 70
STAGE_TITLES = {"stage2": "Stage 2 FDI tooth numbering", "localization": "Lesion localization",
                "stage3": "Stage 3 G.V. Black", "stage4": "Stage 4 ICDAS"}


def _center(t):
    return t.center(70).rstrip()


def _pct(v):
    return "Unavailable" if v is None else f"{v * 100:.1f}%"


# ---------------------------------------------------------------------------------------------------------------
# tables (notebook lesion_table / teeth_table; None-safe for stages that did not run)
# ---------------------------------------------------------------------------------------------------------------
def _gvb_cell(l):
    if l.get("gvb_display") in (None, "Unavailable") or l.get("gvb_confidence") is None:
        return "Unavailable"
    return l["gvb_display"] if not l["gvb_uncertain"] else f"Uncertain (best guess {l['gvb_display']})"


def _icdas_cell(l):
    if l.get("icdas") is None:
        return "Unavailable"
    return ((f"ICDAS {l['icdas']}" if not l["icdas_uncertain"] else f"Uncertain (best guess ICDAS {l['icdas']})")
            + (" [rare class]" if l.get("icdas_unreliable_class") else ""))


LESION_COLUMNS = ["Lesion", "FDI tooth number", "Tooth-number confidence", "Localization confidence",
                  "G.V. Black class", "G.V. Black confidence", "ICDAS score (model-predicted)", "ICDAS confidence"]


def uncertainty_flags(l):
    f = []
    if l.get("fdi_status") not in (None, "ASSIGNED"):
        f.append({"NO_TOOTH": "no covering tooth", "AMBIGUOUS": "lesion spans 2+ teeth",
                  "TOOTH_NUMBER_UNCERTAIN": "tooth number uncertain",
                  "STAGE2_UNAVAILABLE": "Stage 2 unavailable"}.get(l.get("fdi_status"), str(l.get("fdi_status"))))
    if l.get("below_validated_threshold"):
        f.append("low-certainty region")
    if l.get("gvb_uncertain"):
        f.append("G.V. Black uncertain")
    if l.get("gvb_confidence") is None:
        f.append("G.V. Black unavailable")
    if l.get("icdas_uncertain"):
        f.append("ICDAS uncertain")
    if l.get("icdas") is None:
        f.append("ICDAS unavailable")
    if l.get("icdas_unreliable_class"):
        f.append("ICDAS class rarely seen in training")
    return f


def lesion_table(res, extended=False):
    """One row per localized lesion; every value is a model output (or Unavailable / Unknown / Uncertain)."""
    teeth = {t["tooth_id"]: t for t in res.get("teeth", [])}
    rows = []
    for l in res.get("lesions", []):
        r = {"Lesion": l["lesion"],
             "FDI tooth number": l["fdi_display"],
             "Tooth-number confidence": _pct(l.get("fdi_confidence")),
             "Localization confidence": _pct(l["localizer_score"]) + (" (low-certainty region)" if l["below_validated_threshold"] else ""),
             "G.V. Black class": _gvb_cell(l),
             "G.V. Black confidence": _pct(l.get("gvb_confidence")),
             "ICDAS score (model-predicted)": _icdas_cell(l),
             "ICDAS confidence": _pct(l.get("icdas_confidence"))}
        if extended:
            t = teeth.get(l.get("tooth_id"))
            r = {**r, "Tooth detection confidence": _pct(t["detection_score"]) if t else "Unavailable",
                 "Uncertainty flags": "; ".join(uncertainty_flags(l)) or "none"}
        rows.append(r)
    cols = LESION_COLUMNS + (["Tooth detection confidence", "Uncertainty flags"] if extended else [])
    if extended:
        cols = cols[:3] + ["Tooth detection confidence"] + cols[3:8] + ["Uncertainty flags"]
    return pd.DataFrame(rows, columns=cols)


def teeth_table(res):
    return pd.DataFrame([{"Tooth": t["tooth_id"], "FDI number": t["fdi_display"], "Number confidence": _pct(t["fdi_confidence"]),
                          "Detection score": _pct(t["detection_score"]), "Note": t.get("fdi_note") or ""} for t in res.get("teeth", [])],
                        columns=["Tooth", "FDI number", "Number confidence", "Detection score", "Note"])


def lesion_label(les):
    g = "Unavailable" if les.get("gvb_confidence") is None else (
        les["gvb_display"] if not les["gvb_uncertain"] else f"uncertain ({les['gvb_display']}?)")
    i = "ICDAS unavailable" if les.get("icdas") is None else (
        f"ICDAS {les['icdas']}" if not les["icdas_uncertain"] else f"ICDAS uncertain ({les['icdas']}?)")
    gc = f" ({les['gvb_confidence'] * 100:.0f}%)" if les.get("gvb_confidence") is not None else ""
    ic = f" ({les['icdas_confidence'] * 100:.0f}%)" if les.get("icdas_confidence") is not None else ""
    return (f"Lesion {les['lesion']}: tooth {les['fdi_display']} | G.V. Black {g}{gc} | {i}{ic}"
            + ("  [low-certainty region]" if les["below_validated_threshold"] else ""))


def report_limitations(res, fdi_meta=None):
    L = ["RESEARCH PROTOTYPE — not a substitute for professional dental diagnosis."]
    off = [f"{STAGE_TITLES[k]}: {res[k]['status']} ({res[k].get('reason', '')})" for k in STAGE_TITLES
           if k in res and res[k].get("status") != "RAN"]
    if off:
        L += ["Stages not run / unavailable for this image: " + "; ".join(off)]
    fm = fdi_meta or {}
    t = fm.get("test_metrics") or {}
    if "mAP50" in t and "at_threshold_number_accuracy_on_matched" in t:
        L += [f"Stage 2 detector (measured on its held-out test images): mAP@0.5 {t['mAP50']:.3f}, FDI-number accuracy on matched "
              f"teeth {t['at_threshold_number_accuracy_on_matched']:.3f}." + (f" Lesion->tooth association accuracy "
              f"{fm['association_test']['accuracy']:.3f} (n={fm['association_test']['n_lesions']})." if fm.get("association_test") else
              " Lesion->tooth association accuracy could not be measured (no caries boxes in the FDI test data).")]
    L += ["Stage 1 healthy and caries images come from different archives (source confound); no external validation yet.",
          "FDI numbers come from one photograph; mirror (occlusal) views, missing or partially visible teeth reduce reliability. "
          "'Unknown / uncertain' is shown when the number is not confident or the lesion lies across two teeth.",
          "ICDAS scores are MODEL PREDICTIONS from a photograph, not clinically confirmed scores: ICDAS requires a cleaned, air-dried "
          "tooth, visual-tactile examination and often radiographs. ICDAS 0 (sound) is not modelled — Stage 4 outputs 1–6 only.",
          "G.V. Black classes are predicted from the lesion's appearance in a photograph; proximal lesions are often not visible.",
          "Confidences are softmax / sigmoid outputs, not calibrated clinical probabilities. Missed lesions receive no classification.",
          "Please consult a dentist for diagnosis and treatment."]
    return L


# ---------------------------------------------------------------------------------------------------------------
# TXT report (notebook final_report_lines + assessment metadata)
# ---------------------------------------------------------------------------------------------------------------
def final_report_lines(res, patient, risk, fdi_meta=None):
    L = [LINE, _center("FINAL DENTAL REPORT"), _center("research prototype — not a clinical diagnosis"), LINE, ""]
    L += ["1. PATIENT INFORMATION"]
    if patient:
        L += [f"   Name / ID: {patient.get('patient_id') or '-'}", f"   Age: {patient['age']} years" if patient.get("age") else "   Age: -"]
        if patient.get("assessment_datetime"):
            L += [f"   Assessment date / time: {patient['assessment_datetime']}"]
    else:
        L += ["   not collected"]
    L += ["", "2. ORAL-HABITS QUESTIONNAIRE (habit-based risk — separate from the image analysis, not a diagnosis)"]
    if risk:
        for n, q in enumerate(RISK_QUESTIONS, 1):
            a = risk["answers"][q["key"]]
            L += [f"   Q{n}. {q['text']}", f"       -> {a}  ({q['options'][a]} point{'s' if q['options'][a] != 1 else ''})"]
        L += [f"   Habit-based caries risk: {risk['level']}  (score {risk['score']}/{risk['max']})"]
    else:
        L += ["   questionnaire skipped"]
    L += ["", "IMAGE-BASED CNN ANALYSIS" + (f"  —  {res.get('input')}" if res.get("input") else ""), "-" * 70]
    s1 = res.get("stage1")
    L += ["3. Stage 1 — Healthy vs Caries:"]
    if s1:
        L += ["   " + {"CARIES": "CARIES", "HEALTHY": "HEALTHY — the model did not detect caries",
                       "UNCERTAIN": f"UNCERTAIN (leaning {s1['leaning']}) — {s1['reason']}"}[s1["prediction"]],
              f"   Confidence: {s1['confidence'] * 100:.1f}%  (P(caries) {s1['p_caries']:.3f}, threshold {s1['threshold']:.3f})"]
    else:
        L += ["   not run — " + str(res.get("error") or res.get("overall"))]
    st2 = res.get("stage2", {"status": "NOT RUN"})
    L += ["", "4. Stage 2 — FDI tooth detection:"]
    if st2["status"] == "RAN":
        L += [f"   {st2['n_teeth']} teeth detected ({st2['n_number_uncertain']} with an uncertain number)"]
        L += [f"   Tooth {t['tooth_id']}: FDI {t['fdi_display']}   number confidence {t['fdi_confidence'] * 100:.1f}%   "
              f"detection {t['detection_score'] * 100:.1f}%" for t in res.get("teeth", [])]
    else:
        L += [f"   {st2['status']} — {st2.get('reason', '')}"]
    loc = res.get("localization", {"status": "NOT RUN"})
    L += ["", "5. Lesion localization:"]
    L += ([f"   {loc['n_regions']} region(s) at score >= {loc['score_threshold_used']:.2f} (validated threshold "
           f"{loc['validated_threshold']:.2f}); {loc['n_valid_crops']} valid lesion crop(s)"] if loc["status"] == "RAN" else
          [f"   {loc['status']} — {loc.get('reason', '')}"])
    L += ["", "6–8. Per-lesion results (tooth, G.V. Black, ICDAS — model predictions, each with its own confidence):"]
    tab = lesion_table(res)
    if len(tab):
        L += ["   " + l for l in tab.to_string(index=False).splitlines()]
        for les in res["lesions"]:
            if les.get("fdi_note"):
                L += [f"   Lesion {les['lesion']} tooth: {les['fdi_note']}"]
    else:
        reason = res.get("stage3", {}).get("reason") or res.get("overall")
        L += [f"   No lesion classified — G.V. Black: Unavailable, ICDAS: Unavailable ({reason})"]
    L += ["", LINE, "FINAL RESULT", LINE]
    if s1 is None:
        L += ["Image not analysed: " + str(res.get("error") or res.get("overall"))]
    elif s1["prediction"] == "HEALTHY":
        L += ["The image model did not detect caries (Stage 1 = HEALTHY). Stages 2–4 were not required."]
    elif s1["prediction"] == "UNCERTAIN" and not res.get("lesions"):
        L += ["Image result uncertain — please retake a clear, well-lit photo or have a dentist examine the teeth."]
    else:
        L += [res["overall"]]
        for les in res.get("lesions", []):
            L += ["  " + lesion_label(les)]
    L += ["", "9. PREVENTIVE ADVICE (from the questionnaire + general advice)"] + [f"   • {a}" for a in general_advice(res, risk)]
    L += ["", "10. LIMITATIONS"] + ["   " + t for t in report_limitations(res, fdi_meta)]
    L += ["", config.DISCLAIMER, LINE]
    return L


# ---------------------------------------------------------------------------------------------------------------
# JSON / CSV / image
# ---------------------------------------------------------------------------------------------------------------
def build_json(res, patient, risk, model_info=None):
    payload = {
        "report": "Dental Caries AI Assessment — structured results",
        "app_version": config.APP_VERSION,
        "disclaimer": config.ACADEMIC_DISCLAIMER,
        "patient": patient,
        "habit_risk": risk,
        "image_analysis": strip_private({k: v for k, v in res.items() if k != "steps"}),
        "stage_status": {k: ui_status(res, k) for k in ("stage1", "stage2", "localization", "stage3", "stage4")},
        "pipeline_steps": strip_private(res.get("steps", {})),
        "models": model_info or [],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


def build_csv(res):
    buf = io.StringIO()
    lesion_table(res, extended=True).to_csv(buf, index=False, quoting=csv.QUOTE_MINIMAL)
    return buf.getvalue()


def encode_jpeg(rgb, quality=92):
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return enc.tobytes()


def unique_base_name(patient=None):
    pid = (patient or {}).get("patient_id") or ""
    pid = "".join(ch for ch in pid if ch.isalnum() or ch in "-_")[:24]
    return f"dental_assessment_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}" + (f"_{pid}" if pid else "")


# ---------------------------------------------------------------------------------------------------------------
# PDF (ReportLab)
# ---------------------------------------------------------------------------------------------------------------
_FONTS_READY = {}


def _register_fonts():
    if _FONTS_READY:
        return _FONTS_READY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    fdir = os.path.join(config.ASSETS_DIR, "fonts")
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(fdir, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(fdir, "DejaVuSans-Bold.ttf")))
        _FONTS_READY.update(regular="DejaVu", bold="DejaVu-Bold")
    except Exception:  # fonts missing -> built-in Helvetica (a few symbols may not render)
        _FONTS_READY.update(regular="Helvetica", bold="Helvetica-Bold")
    return _FONTS_READY


NAVY, TEAL, LIGHT = "#0B2545", "#0E7C86", "#EAF6F8"
STATUS_COLORS = {"Completed": "#1B7F3B", "Uncertain": "#B26A00", "Unavailable": "#6B7280", "Skipped": "#475569", "Failed": "#B42318"}


def build_pdf(res, patient, risk, rgb, annotated, model_rows=None, fdi_meta=None, crops=None):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image as RLImage, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)
    from xml.sax.saxutils import escape

    f = _register_fonts()
    W, H = A4
    S = {
        "h1": ParagraphStyle("h1", fontName=f["bold"], fontSize=17, leading=21, textColor=colors.HexColor(NAVY)),
        "h2": ParagraphStyle("h2", fontName=f["bold"], fontSize=12, leading=15, textColor=colors.HexColor(NAVY), spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", fontName=f["regular"], fontSize=9, leading=12.5),
        "small": ParagraphStyle("small", fontName=f["regular"], fontSize=7.6, leading=9.8),
        "smallb": ParagraphStyle("smallb", fontName=f["bold"], fontSize=7.6, leading=9.8, textColor=colors.white),
        "center": ParagraphStyle("center", fontName=f["regular"], fontSize=8, leading=10, alignment=TA_CENTER,
                                 textColor=colors.HexColor("#475569")),
        "disc": ParagraphStyle("disc", fontName=f["regular"], fontSize=8.4, leading=11.5, textColor=colors.HexColor("#7A2E0E")),
    }
    P = lambda t, s="body": Paragraph(escape(str(t)), S[s])

    def table(data, widths, header=True, zebra=True, font_size=None):
        rows = [[c if not isinstance(c, str) else P(c, "smallb" if (header and i == 0) else "small") for c in r]
                for i, r in enumerate(data)]
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
              ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TEAL)))
        if zebra:
            for i in range(1 if header else 0, len(rows)):
                if i % 2 == 0:
                    st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F4FAFB")))
        t.setStyle(TableStyle(st))
        return t

    def rl_image(arr, max_w, max_h):
        bio = io.BytesIO(encode_jpeg(arr, 88))
        h, w = arr.shape[:2]; s = min(max_w / w, max_h / h)
        return RLImage(bio, width=w * s, height=h * s)

    stamp = (patient or {}).get("assessment_datetime") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def on_page(canv, doc):
        canv.saveState()
        canv.setFillColor(colors.HexColor(TEAL)); canv.rect(0, H - 9 * mm, W, 9 * mm, stroke=0, fill=1)
        canv.setFillColor(colors.white); canv.setFont(f["bold"], 9)
        canv.drawString(15 * mm, H - 6 * mm, config.APP_TITLE)
        canv.setFont(f["regular"], 8); canv.drawRightString(W - 15 * mm, H - 6 * mm, f"Assessment {stamp}")
        canv.setFillColor(colors.HexColor("#64748B")); canv.setFont(f["regular"], 7.5)
        canv.drawString(15 * mm, 9 * mm, "Academic AI proof of concept — not a clinical diagnosis.")
        canv.drawRightString(W - 15 * mm, 9 * mm, f"Page {doc.page}")
        canv.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Dental Caries AI Assessment Report", author="Dental Caries AI Assessment System")
    CW = W - 30 * mm
    story = []
    logo = os.path.join(config.ASSETS_DIR, "logo.png")
    head = [P("Dental Caries AI Assessment Report", "h1"), P(config.APP_SUBTITLE, "body")]
    if os.path.isfile(logo):
        story.append(Table([[RLImage(logo, width=16 * mm, height=16 * mm), head]], colWidths=[20 * mm, CW - 20 * mm],
                           style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    else:
        story += head
    story += [Spacer(1, 4), P(config.ACADEMIC_DISCLAIMER, "disc"), Spacer(1, 6)]

    # 1. summary
    s1 = res.get("stage1")
    story.append(P("1. Assessment summary", "h2"))
    summ = [["Item", "Value"],
            ["Patient name / ID", (patient or {}).get("patient_id") or "—"],
            ["Age", f"{patient['age']} years" if (patient or {}).get("age") else "—"],
            ["Assessment date / time", stamp],
            ["Image", f"{res.get('input', '—')}" + (f"  ({rgb.shape[1]}×{rgb.shape[0]} px)" if rgb is not None else "")],
            ["Questionnaire risk", f"{risk['level']} — score {risk['score']}/{risk['max']}" if risk else "not completed"],
            ["Stage 1 result", (f"{s1['prediction']} — confidence {s1['confidence'] * 100:.1f}%" if s1 else "not run")],
            ["Overall pipeline result", res.get("overall", "—")]]
    story.append(table(summ, [45 * mm, CW - 45 * mm]))

    # 2. questionnaire
    story.append(P("2. Oral-habits questionnaire (habit-based risk — separate from the image analysis)", "h2"))
    if risk:
        q = [["#", "Question", "Answer", "Points"]]
        for n, qq in enumerate(RISK_QUESTIONS, 1):
            a = risk["answers"][qq["key"]]
            q.append([str(n), qq["text"], a, str(qq["options"][a])])
        q.append(["", "Total score / risk category", risk["level"], f"{risk['score']} / {risk['max']}"])
        story.append(table(q, [8 * mm, CW - 8 * mm - 45 * mm - 16 * mm, 45 * mm, 16 * mm]))
        story.append(Spacer(1, 3))
        story.append(P("Risk categories (notebook rules): LOW 0–3 · MODERATE 4–7 · HIGH 8–13. The questionnaire never changes a CNN "
                       "prediction and is not a diagnosis.", "small"))
    else:
        story.append(P("Questionnaire not completed.", "body"))

    # 3. stage overview
    story.append(P("3. AI pipeline — stage results", "h2"))
    rows = [["Stage", "Status", "Result"]]
    st2, loc = res.get("stage2", {}), res.get("localization", {})
    detail = {
        "stage1": (f"{s1['prediction']} (leaning {s1['leaning']}) · confidence {s1['confidence'] * 100:.1f}% · P(caries) {s1['p_caries']:.3f}"
                   f" · threshold {s1['threshold']:.3f}" + (f" · {s1['reason']}" if s1 and s1.get('reason') else "")) if s1 else
                  str(res.get("error") or "not run"),
        "stage2": (f"{st2.get('n_teeth')} teeth detected, {st2.get('n_number_uncertain')} with an uncertain number"
                   if st2.get("status") == "RAN" else f"{st2.get('status')} — {st2.get('reason', '')}"),
        "localization": (f"{loc.get('n_regions')} region(s) at score ≥ {loc.get('score_threshold_used', 0):.2f} (validated "
                         f"{loc.get('validated_threshold', 0):.2f}); {loc.get('n_valid_crops')} valid crop(s)"
                         if loc.get("status") == "RAN" else f"{loc.get('status')} — {loc.get('reason', '')}"),
        "stage3": (f"{len(res.get('lesions', []))} lesion(s) classified" if res.get("stage3", {}).get("status") == "RAN"
                   else f"{res.get('stage3', {}).get('status')} — {res.get('stage3', {}).get('reason', '')}"),
        "stage4": (f"{len(res.get('lesions', []))} lesion(s) scored (ICDAS 1–6 model)" if res.get("stage4", {}).get("status") == "RAN"
                   else f"{res.get('stage4', {}).get('status')} — {res.get('stage4', {}).get('reason', '')}"),
    }
    names = {"stage1": "Stage 1 — Healthy vs Caries", "stage2": "Stage 2 — FDI tooth identification",
             "localization": "Cavity localization", "stage3": "Stage 3 — G.V. Black", "stage4": "Stage 4 — ICDAS"}
    status_rows = []
    for k in ("stage1", "stage2", "localization", "stage3", "stage4"):
        stt = ui_status(res, k)
        rows.append([names[k], stt, detail[k]]); status_rows.append(stt)
    t = table(rows, [48 * mm, 24 * mm, CW - 72 * mm])
    for i, stt in enumerate(status_rows, 1):
        t.setStyle(TableStyle([("TEXTCOLOR", (1, i), (1, i), colors.HexColor(STATUS_COLORS.get(stt, NAVY)))]))
    story.append(t)

    # 4. images
    if rgb is not None:
        iw = (CW - 6 * mm) / 2
        imgs = [rl_image(rgb, iw, 80 * mm), rl_image(annotated if annotated is not None else rgb, iw, 80 * mm)]
        story.append(KeepTogether([
            P("4. Original and annotated images", "h2"),
            Table([imgs, [P("Original upload (EXIF-corrected)", "center"), P("AI annotation", "center")]], colWidths=[iw + 3 * mm] * 2,
                  style=[("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]),
            Spacer(1, 2),
            P("Legend: orange boxes = Stage 2 teeth with FDI number ('?' = number uncertain); green boxes = localized lesions, "
              "labelled 'L<lesion> T<tooth>: <G.V. Black class> | ICDAS <score>'. All boxes come from the models' outputs for this photo.",
              "small")]))

    # 5. teeth
    story.append(P("5. Stage 2 — detected teeth (FDI)", "h2"))
    tt = teeth_table(res)
    if len(tt):
        story.append(table([list(tt.columns)] + tt.astype(str).values.tolist(), [14 * mm, 52 * mm, 26 * mm, 24 * mm, CW - 116 * mm]))
    else:
        story.append(P(detail["stage2"] if st2.get("status") != "RAN" else "No tooth detected above the validated threshold.", "body"))

    # 6. lesions
    story.append(P("6. Per-lesion results (model predictions)", "h2"))
    lt = lesion_table(res, extended=True)
    if len(lt):
        heads = ["#", "FDI tooth", "Tooth-no. conf.", "Tooth det. conf.", "Localization conf.", "G.V. Black", "G.V.B. conf.",
                 "ICDAS", "ICDAS conf.", "Flags"]
        widths = [7, 27, 15, 15, 19, 22, 14, 23, 14, 24]
        sc = CW / (sum(widths) * mm)
        story.append(table([heads] + lt.astype(str).values.tolist(), [w * mm * sc for w in widths]))
        story.append(Spacer(1, 4))
        for l in res["lesions"]:
            parts = [f"Lesion {l['lesion']}"]
            if l.get("gvb_probabilities"):
                parts.append("G.V. Black probabilities: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in l["gvb_probabilities"].items()))
            if l.get("icdas_probabilities"):
                parts.append("ICDAS probabilities: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in l["icdas_probabilities"].items())
                             + " (ICDAS 0 not modelled)")
            if l.get("fdi_note"):
                parts.append("Tooth note: " + l["fdi_note"])
            story.append(P(" · ".join(parts), "small"))
        if crops:
            cells, caps = [], []
            for c, l in zip(crops, res["lesions"]):
                cells += [rl_image(c["gvb_crop"], 26 * mm, 26 * mm), rl_image(c["icdas_crop"], 26 * mm, 26 * mm)]
                caps += [P(f"L{l['lesion']} G.V. Black input", "center"), P(f"L{l['lesion']} ICDAS input", "center")]
            per = 6
            for s in range(0, len(cells), per):
                story.append(Spacer(1, 3))
                story.append(Table([cells[s:s + per], caps[s:s + per]], colWidths=[CW / per] * len(cells[s:s + per])))
    else:
        reason = res.get("stage3", {}).get("reason") or res.get("overall")
        story.append(P(f"No lesion classified — G.V. Black: Unavailable, ICDAS: Unavailable ({reason}).", "body"))

    # 7. advice, 8. limitations, 9. technical
    story.append(P("7. Preventive advice (questionnaire + general advice)", "h2"))
    for a in general_advice(res, risk):
        story.append(P("• " + a, "body"))
    story.append(P("8. Limitations and warnings", "h2"))
    for a in report_limitations(res, fdi_meta):
        story.append(P("• " + a, "small"))
    if model_rows:
        story.append(P("9. Technical information — models used", "h2"))
        story.append(table([["Model", "Status", "Details"]] + [[r["Model"], r["Status"], str(r["Details"])] for r in model_rows],
                           [52 * mm, 22 * mm, CW - 74 * mm]))
        steps = res.get("steps", {})
        if steps:
            story.append(Spacer(1, 4))
            story.append(table([["Pipeline step", "State", "Message"]] +
                               [[STEP_TITLES[k], steps[k]["state"], steps[k]["message"]] for k in STEP_ORDER if k in steps],
                               [52 * mm, 22 * mm, CW - 74 * mm]))
    story += [Spacer(1, 8), P(config.ACADEMIC_DISCLAIMER + " " + config.DISCLAIMER, "disc")]
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------------------------------------------
def build_all(res, patient, risk, rgb, model_rows=None, fdi_meta=None, crops=None):
    """-> (base_name, {filename: bytes}) for PDF, TXT, JSON, CSV, annotated JPG, plus a ZIP of all of them."""
    base = unique_base_name(patient)
    annotated = draw_result(rgb, res) if (rgb is not None and res.get("stage1")) else rgb
    files = {
        f"{base}_report.pdf": build_pdf(res, patient, risk, rgb, annotated, model_rows, fdi_meta, crops),
        f"{base}_summary.txt": "\n".join(final_report_lines(res, patient, risk, fdi_meta)).encode("utf-8"),
        f"{base}_results.json": build_json(res, patient, risk, model_rows).encode("utf-8"),
        f"{base}_lesions.csv": build_csv(res).encode("utf-8"),
    }
    if annotated is not None:
        files[f"{base}_annotated.jpg"] = encode_jpeg(annotated)
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    files[f"{base}_all_results.zip"] = zbuf.getvalue()
    return base, files
