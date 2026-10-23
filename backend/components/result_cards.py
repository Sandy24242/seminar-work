"""Results dashboard pieces: stage status cards, per-lesion details with probability charts, technical information."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.styles import badge, card, esc
from pipeline.runner import STEP_ORDER, STEP_TITLES, ui_status


def _pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def stage_cards(res):
    s1, st2, loc, s3, s4 = (res.get(k) or {} for k in ("stage1", "stage2", "localization", "stage3", "stage4"))
    lesions = res.get("lesions", [])
    reason = lambda d: f'<div class="dc-muted">{esc(d.get("reason") or "")}</div>'

    if res.get("stage1"):
        body = (f'<div class="dc-big">{esc(s1["prediction"])}</div>'
                f'<div>Confidence <b>{_pct(s1["confidence"])}</b> · P(caries) {s1["p_caries"]:.3f} · validated threshold {s1["threshold"]:.3f}</div>'
                + (f'<div class="dc-muted">Leaning {esc(s1["leaning"])} — {esc(s1["reason"])}</div>' if s1.get("reason") else ""))
    else:
        body = f'<div class="dc-muted">{esc(res.get("error") or "not run")}</div>'
    s2b = (f'<div class="dc-big">{st2["n_teeth"]} teeth</div><div>{st2["n_number_uncertain"]} with an uncertain FDI number · '
           f'detection ≥ {st2["score_threshold"]:.2f}, number confidence ≥ {st2["class_confidence_threshold"]:.2f}</div>'
           + (f'<div style="margin-top:.35rem">' + "".join(f'<span class="dc-pill">{esc(t["fdi"] if not t["fdi_uncertain"] else "?" )}'
                                                         f' · {_pct(t["fdi_confidence"])}</span>' for t in res.get("teeth", [])) + "</div>")
           if st2.get("status") == "RAN" else reason(st2))
    locb = (f'<div class="dc-big">{loc["n_valid_crops"]} lesion(s)</div><div>{loc["n_regions"]} region(s) at score ≥ '
            f'{loc["score_threshold_used"]:.2f} (validated {loc["validated_threshold"]:.2f})</div>'
            + "".join(f'<span class="dc-pill">L{l["lesion"]} · {_pct(l["localizer_score"])}{" · low-certainty" if l["below_validated_threshold"] else ""}</span>'
                      for l in lesions) if loc.get("status") == "RAN" else reason(loc))
    s3b = ("".join(f'<div>L{l["lesion"]}: <b>{esc(l["gvb_display"])}</b> · {_pct(l["gvb_confidence"])}'
                   f'{" · <i>uncertain</i>" if l.get("gvb_uncertain") else ""}</div>' for l in lesions)
           if s3.get("status") == "RAN" else reason(s3))
    s4b = ("".join(f'<div>L{l["lesion"]}: <b>ICDAS {l["icdas"]}</b> · {_pct(l["icdas_confidence"])}'
                   f'{" · <i>uncertain</i>" if l.get("icdas_uncertain") else ""}{" · rare class" if l.get("icdas_unreliable_class") else ""}</div>'
                   for l in lesions) + '<div class="dc-muted">Model covers ICDAS 1–6 only (no ICDAS 0).</div>'
           if s4.get("status") == "RAN" else reason(s4))
    items = [("stage1", "Stage 1 — Healthy vs Caries", body, "icon_stage1.png"),
             ("stage2", "Stage 2 — FDI tooth identification", s2b, "icon_stage2.png"),
             ("localization", "Cavity localization", locb, "icon_localization.png"),
             ("stage3", "Stage 3 — G.V. Black classification", s3b, "icon_stage3.png"),
             ("stage4", "Stage 4 — ICDAS scoring", s4b, "icon_stage4.png")]
    cols = st.columns(2)
    for i, (k, title, b, icon) in enumerate(items):
        with cols[i % 2]:
            card(f'<h4>{esc(title)} &nbsp;{badge(ui_status(res, k))}</h4>{b}', icon=icon)


def prob_chart(probs, highlight, title):
    names, vals = list(probs), [v * 100 for v in probs.values()]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", text=[f"{v:.1f}%" for v in vals], textposition="outside",
                           marker_color=["#0E7C86" if n == highlight else "#B9DDE3" for n in names]))
    fig.update_layout(title=dict(text=title, font=dict(size=14, color="#0B2545")), height=60 + 28 * len(names),
                      margin=dict(l=10, r=40, t=36, b=10), xaxis=dict(range=[0, 115], showgrid=False, visible=False),
                      yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def lesion_details(res, crops):
    for i, l in enumerate(res.get("lesions", [])):
        with st.expander(f"Lesion {l['lesion']} — tooth {l['fdi_display']} · {l['gvb_display']} · "
                         f"{'ICDAS ' + str(l['icdas']) if l.get('icdas') is not None else 'ICDAS unavailable'}", expanded=i == 0):
            c1, c2, c3 = st.columns([1, 1.3, 1.3])
            with c1:
                if crops and i < len(crops):
                    a, b = st.columns(2)
                    a.image(crops[i]["gvb_crop"], caption="G.V. Black input", width="stretch")
                    b.image(crops[i]["icdas_crop"], caption="ICDAS input", width="stretch")
                st.markdown(f'<div class="dc-kv" style="font-size:.88rem"><b>Box (px)</b><span>{l["box_xyxy_px"]}</span>'
                            f'<b>Localization</b><span>{_pct(l["localizer_score"])}</span>'
                            f'<b>Tooth link</b><span>{esc(l["fdi_status"])} · overlap {_pct(l.get("fdi_overlap"))}</span>'
                            + (f'<b>Note</b><span>{esc(l["fdi_note"])}</span>' if l.get("fdi_note") else "") + "</div>",
                            unsafe_allow_html=True)
            with c2:
                if l.get("gvb_probabilities"):
                    st.plotly_chart(prob_chart(l["gvb_probabilities"], l["gvb_display"], "G.V. Black probabilities"),
                                    width="stretch", config={"displayModeBar": False}, key=f"gvb_{i}")
                else:
                    st.info(l.get("gvb_note") or "G.V. Black prediction unavailable.")
            with c3:
                if l.get("icdas_probabilities"):
                    st.plotly_chart(prob_chart(l["icdas_probabilities"], f"ICDAS {l['icdas']}", "ICDAS probabilities (1–6)"),
                                    width="stretch", config={"displayModeBar": False}, key=f"icdas_{i}")
                else:
                    st.info(l.get("icdas_note") or "ICDAS prediction unavailable.")
            st.caption("Model predictions from a photograph — not confirmed clinical findings. Confidences are softmax outputs, not calibrated probabilities.")


def technical_info(res, registry, models_source):
    with st.expander("Technical information — models, stage statuses, warnings", expanded=False):
        st.markdown(f"**Model folder:** {esc(models_source)}  ·  **TensorFlow** {registry.versions.get('tensorflow', '—')} · "
                    f"**Keras** {registry.versions.get('keras', '—')}")
        st.dataframe(pd.DataFrame(registry.summary()), hide_index=True, width="stretch")
        steps = res.get("steps", {})
        if steps:
            st.markdown("**Pipeline steps (this run)**")
            st.dataframe(pd.DataFrame([{"Step": STEP_TITLES[k], "State": steps[k]["state"], "Seconds": steps[k]["seconds"],
                                        "Message": steps[k]["message"]} for k in STEP_ORDER if k in steps]),
                         hide_index=True, width="stretch")
        warn = [f"{k}: {v.get('status')} — {v.get('reason')}" for k, v in res.items()
                if k in ("stage2", "localization", "stage3", "stage4") and isinstance(v, dict) and v.get("status") != "RAN"]
        for w in warn:
            st.warning(w)
        m = registry.meta
        if m.get("stage1"):
            st.caption(f"Stage 1 decision rule: P(caries) ≥ {m['stage1']['threshold']:.4f} → CARIES; UNCERTAIN if within "
                       f"±{m['stage1']['uncertain_margin']} of the threshold or confidence < {m['stage1']['confidence_threshold']}.")
        if m.get("gvb"):
            st.caption(f"G.V. Black class mapping (from metadata): " +
                       ", ".join(f"{i} → {n}" for i, n in enumerate(m['gvb']['class_names'])))
        if m.get("icdas"):
            st.caption(f"ICDAS output mapping (from metadata): " + ", ".join(f"{i} → ICDAS {s}" for i, s in enumerate(m['icdas']['scores'])))
