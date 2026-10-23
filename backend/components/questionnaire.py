"""Patient form + oral-habits questionnaire UI (wording, options and points from pipeline.questionnaire = notebook Section 82)."""
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

import config
from components.styles import card, esc, risk_badge
from pipeline.questionnaire import QUESTION_TOPICS, RISK_QUESTIONS, level_ranges, score_answers


def _label(opt):
    return opt[:1].upper() + opt[1:]


def patient_and_questionnaire_form():
    """Returns (patient, risk) when the form is submitted and valid, else None. Errors are shown inline."""
    prev_p = st.session_state.get("patient") or {}
    prev_r = (st.session_state.get("risk") or {}).get("answers", {})
    with st.form("assessment_form", clear_on_submit=False):
        st.markdown("#### Patient information")
        c1, c2, c3 = st.columns([2, 1, 1.4])
        pid = c1.text_input("Patient name or ID (optional)", value=prev_p.get("patient_id") or "", max_chars=60,
                            help="Optional. Avoid entering more personal information than you need.")
        age = c2.number_input("Age (years) *", min_value=config.AGE_MIN, max_value=config.AGE_MAX, step=1,
                              value=prev_p.get("age"), placeholder="e.g. 34")
        c3.text_input("Assessment date / time", value="generated automatically on submit", disabled=True)

        st.markdown("#### Oral-habits questionnaire")
        st.caption("Answer all 7 questions. The habit-based score is separate from the image analysis and never changes an AI prediction.")
        answers = {}
        for n, q in enumerate(RISK_QUESTIONS, 1):
            opts = list(q["options"])
            prev = prev_r.get(q["key"])
            answers[q["key"]] = st.radio(f"**Q{n}. {q['text']}**", opts, index=opts.index(prev) if prev in opts else None,
                                         format_func=_label, horizontal=True, key=f"q_{q['key']}")
        submitted = st.form_submit_button("Calculate risk score and continue", type="primary", width="stretch")
    if not submitted:
        return None
    errors = []
    if age is None or not float(age).is_integer() or not (config.AGE_MIN <= int(age) <= config.AGE_MAX):
        errors.append(f"Age is required: enter a whole number between {config.AGE_MIN} and {config.AGE_MAX}.")
    missing = [f"Q{n}" for n, q in enumerate(RISK_QUESTIONS, 1) if answers.get(q["key"]) is None]
    if missing:
        errors.append("Please answer every question (missing: " + ", ".join(missing) + ").")
    if errors:
        for e in errors:
            st.error(e)
        return None
    patient = {"patient_id": pid.strip() or None, "age": int(age),
               "assessment_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    return patient, score_answers(answers)


def risk_gauge(risk):
    colors = {"LOW": "#2E9E5B", "MODERATE": "#E0A100", "HIGH": "#D64545"}
    steps = [{"range": [lo, hi + (1 if name == "HIGH" else 0.999)], "color": c}
             for (name, lo, hi), c in zip(level_ranges(), ("#DDF2E4", "#FFF0CC", "#FBDADA"))]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk["score"], number={"suffix": f" / {risk['max']}", "font": {"color": "#0B2545", "size": 34}},
        gauge={"axis": {"range": [0, risk["max"]], "tickvals": [0, 3, 7, 13]}, "bar": {"color": colors[risk["level"]], "thickness": 0.32},
               "steps": steps, "borderwidth": 0}))
    fig.update_layout(height=210, margin=dict(l=20, r=20, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def show_risk_result(risk, compact=False):
    c1, c2 = st.columns([1, 1.35])
    with c1:
        st.markdown(f'<div class="dc-card"><h4>Habit-based caries risk</h4>{risk_badge(risk["level"])}'
                    f'<div class="dc-big" style="margin-top:.5rem">{risk["score"]} / {risk["max"]} points</div>'
                    f'<div class="dc-muted">LOW 0–3 · MODERATE 4–7 · HIGH 8–13 (questionnaire only — not a diagnosis)</div></div>',
                    unsafe_allow_html=True)
        st.plotly_chart(risk_gauge(risk), width="stretch", config={"displayModeBar": False})
    with c2:
        rows = "".join(f"<tr><td>Q{n}</td><td>{esc(QUESTION_TOPICS[q['key']])}</td><td>{esc(_label(risk['answers'][q['key']]))}</td>"
                       f"<td style='text-align:right'><b>{risk['points'][q['key']]}</b></td></tr>"
                       for n, q in enumerate(RISK_QUESTIONS, 1))
        st.markdown('<div class="dc-card"><h4>Points per answer</h4><table style="width:100%;font-size:.92rem">'
                    '<tr><th align="left">#</th><th align="left">Topic</th><th align="left">Answer</th><th align="right">Points</th></tr>'
                    f'{rows}</table></div>', unsafe_allow_html=True)
        if not compact:
            adv = risk["advice"] or ["Your answers did not trigger any specific habit advice — keep up the good habits."]
            card("<h4>Personalised advice</h4>" + "".join(f"<div>• {esc(a)}</div>" for a in adv), icon="icon_questionnaire.png")
