"""Global CSS (white / light-blue background, teal accents, navy headings, rounded cards) and small HTML helpers."""
import base64
import html
import os

import streamlit as st

import config

STATUS_STYLE = {
    "Completed": ("#E7F6EC", "#1B7F3B", "✔"),
    "Uncertain": ("#FFF4E0", "#9A5B00", "?"),
    "Unavailable": ("#EEF1F4", "#4B5563", "–"),
    "Skipped": ("#EEF2F7", "#334155", "↷"),
    "Failed": ("#FDECEA", "#B42318", "✖"),
}
RISK_STYLE = {"LOW": ("#E7F6EC", "#1B7F3B"), "MODERATE": ("#FFF4E0", "#9A5B00"), "HIGH": ("#FDECEA", "#B42318")}

CSS = """
<style>
:root { --navy:#0B2545; --teal:#0E7C86; --teal2:#13A3B4; --sky:#CDEBF0; --light:#EAF6F8; --muted:#5B6B7B; }
.stApp { background: linear-gradient(180deg, #F6FBFD 0%, #EEF7FA 100%); }
h1, h2, h3, h4 { color: var(--navy) !important; letter-spacing: -0.01em; }
.block-container { padding-top: 2.2rem; max-width: 1240px; }
.dc-card { background:#fff; border-radius:18px; padding:1.2rem 1.3rem; box-shadow:0 6px 24px rgba(11,37,69,.08);
           border:1px solid #E3EEF2; margin-bottom:1rem; }
.dc-card h4 { margin:.1rem 0 .4rem 0; font-size:1.05rem; }
.dc-hero { background:linear-gradient(135deg,#FFFFFF 0%,#E3F4F7 100%); border-radius:24px; padding:2rem 2rem 1.6rem 2rem;
           box-shadow:0 10px 36px rgba(11,37,69,.10); border:1px solid #DCEDF2; }
.dc-hero h1 { font-size:2.35rem; line-height:1.15; margin-bottom:.4rem; }
.dc-sub { color:var(--teal); font-size:1.12rem; font-weight:600; margin-bottom:.9rem; }
.dc-muted { color:var(--muted); font-size:.93rem; }
.dc-badge { display:inline-block; padding:.18rem .62rem; border-radius:999px; font-weight:700; font-size:.80rem; }
.dc-pill { display:inline-block; padding:.2rem .7rem; border-radius:999px; background:var(--light); color:var(--teal);
           font-weight:600; font-size:.82rem; margin:.1rem .25rem .1rem 0; border:1px solid #CFE7EC; }
.dc-stage { display:flex; gap:.9rem; align-items:flex-start; }
.dc-stage img { width:54px; height:54px; border-radius:14px; flex:0 0 54px; }
.dc-kv { display:grid; grid-template-columns: max-content 1fr; gap:.25rem .9rem; font-size:.95rem; }
.dc-kv b { color:var(--navy); }
.dc-big { font-size:1.55rem; font-weight:800; color:var(--navy); line-height:1.2; }
.dc-disclaimer { background:#FFF8EC; border:1px solid #F5D9A8; color:#7A2E0E; border-radius:14px; padding:.8rem 1rem; font-size:.92rem; }
.dc-steps { display:flex; flex-wrap:wrap; gap:.4rem; margin:.3rem 0 1rem 0; }
.dc-step { padding:.35rem .8rem; border-radius:999px; background:#fff; border:1px solid #D5E6EC; color:var(--muted); font-weight:600; font-size:.85rem; }
.dc-step.on { background:var(--teal); color:#fff; border-color:var(--teal); }
.dc-step.done { background:#E7F6EC; color:#1B7F3B; border-color:#BFE3CB; }
.dc-legend span { display:inline-block; width:14px; height:14px; border-radius:3px; margin:0 .35rem -2px .9rem; }
div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
    background:linear-gradient(135deg,#13A3B4,#0E7C86); border:0; border-radius:12px; font-weight:700; padding:.55rem 1.2rem; }
div.stButton > button, div.stDownloadButton > button { border-radius:12px; }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #E3EEF2; }
[data-testid="stFileUploaderDropzone"] { border:2px dashed #9FD3DB; background:#F7FCFD; border-radius:16px; }
@media (max-width: 760px) {
  .dc-hero { padding:1.2rem; } .dc-hero h1 { font-size:1.65rem; } .block-container { padding-left:.8rem; padding-right:.8rem; }
  .dc-kv { grid-template-columns: 1fr; }
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def asset_path(*parts):
    return os.path.join(config.ASSETS_DIR, *parts)


def asset_exists(*parts):
    return os.path.isfile(asset_path(*parts))


def img_data_uri(*parts):
    """Base64 data URI of an asset, or None if the (optional, decorative) file is missing."""
    p = asset_path(*parts)
    if not os.path.isfile(p):
        return None
    mime = "image/png" if p.lower().endswith(".png") else "image/jpeg"
    with open(p, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def show_image(*parts, caption=None, **kw):
    """st.image for an optional decorative asset; silently skipped when the file is missing."""
    if asset_exists(*parts):
        st.image(asset_path(*parts), caption=caption, **kw)


def esc(t):
    return html.escape(str(t))


def badge(status):
    bg, fg, sym = STATUS_STYLE.get(status, ("#EEF1F4", "#334155", "•"))
    return f'<span class="dc-badge" style="background:{bg};color:{fg}">{sym} {esc(status)}</span>'


def risk_badge(level):
    bg, fg = RISK_STYLE.get(level, ("#EEF1F4", "#334155"))
    return f'<span class="dc-badge" style="background:{bg};color:{fg};font-size:.95rem">{esc(level)} RISK</span>'


def card(inner_html, icon=None):
    uri = img_data_uri("icons", icon) if icon else None
    body = f'<div class="dc-stage"><img src="{uri}" alt=""/><div style="flex:1">{inner_html}</div></div>' if uri else inner_html
    st.markdown(f'<div class="dc-card">{body}</div>', unsafe_allow_html=True)


def disclaimer(text=None):
    st.markdown(f'<div class="dc-disclaimer">⚠️ {esc(text or config.ACADEMIC_DISCLAIMER)}</div>', unsafe_allow_html=True)


def step_bar(current, done):
    names = ["1 · Patient", "2 · Questionnaire", "3 · Dental photo", "4 · AI analysis", "5 · Results"]
    out = []
    for i, n in enumerate(names, 1):
        cls = "on" if i == current else ("done" if i in done else "")
        out.append(f'<span class="dc-step {cls}">{"✓ " if i in done and i != current else ""}{esc(n)}</span>')
    st.markdown('<div class="dc-steps">' + "".join(out) + "</div>", unsafe_allow_html=True)
