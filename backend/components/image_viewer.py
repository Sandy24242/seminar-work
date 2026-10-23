"""Upload widget with preview / replace / remove, and the original-vs-annotated comparison view."""
import streamlit as st

from components.styles import esc
from pipeline.image_validation import validate_image_bytes


def upload_panel():
    """Returns (rgb, info) for a valid upload, else (None, None). The photo stays in memory only."""
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    up = st.file_uploader("Drag and drop an intraoral photograph here, or click to browse (JPG, JPEG or PNG)",
                          type=["jpg", "jpeg", "png"], key=f"uploader_{st.session_state.uploader_key}")
    if up is None:
        st.session_state.pop("upload", None)
        return None, None
    data = up.getvalue()
    cached = st.session_state.get("upload")
    if not cached or cached["name"] != up.name or cached["size"] != len(data):
        rgb, info, err = validate_image_bytes(data, up.name)
        cached = {"name": up.name, "size": len(data), "rgb": rgb, "info": info, "error": err}
        st.session_state.upload = cached
    if cached["error"]:
        st.error(f"This file cannot be analysed: {cached['error']}. Please upload a different photo.")
        if st.button("Remove file", key="rm_bad"):
            remove_upload()
        return None, None
    info, rgb = cached["info"], cached["rgb"]
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.image(rgb, caption="Preview (EXIF orientation corrected, RGB)", width="stretch")
    with c2:
        st.markdown(f'<div class="dc-card"><h4>Uploaded image</h4><div class="dc-kv">'
                    f'<b>File</b><span>{esc(info["filename"])}</span>'
                    f'<b>Format</b><span>{esc(info["format"])}</span>'
                    f'<b>Dimensions</b><span>{info["width"]} × {info["height"]} px</span>'
                    f'<b>File size</b><span>{info["size_kb"]} KB</span>'
                    f'<b>EXIF rotation</b><span>{"applied" if info["exif_rotated"] else "none needed"}</span>'
                    f'</div><div class="dc-muted" style="margin-top:.6rem">To replace the photo, drop a new file on the upload box.'
                    f'</div></div>', unsafe_allow_html=True)
        if st.button("Remove image", icon=":material/delete:", width="stretch"):
            remove_upload()
    return rgb, info


def remove_upload():
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
    st.session_state.pop("upload", None)
    st.rerun()


LEGEND = ('<div class="dc-legend dc-muted" style="margin:.2rem 0 .6rem 0">Legend:'
          '<span style="background:#FF8C00"></span>Stage 2 tooth + FDI number (“?” = number uncertain)'
          '<span style="background:#00FF00"></span>Lesion: L&lt;id&gt; T&lt;tooth&gt;: G.V. Black | ICDAS</div>')


def comparison(rgb, annotated):
    st.markdown(LEGEND, unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Side by side", "Annotated (large)", "Original (large)"])
    with t1:
        c1, c2 = st.columns(2)
        c1.image(rgb, caption="Original upload", width="stretch")
        c2.image(annotated, caption="AI annotation (real model outputs)", width="stretch")
    with t2:
        st.image(annotated, width="stretch")
    with t3:
        st.image(rgb, width="stretch")
