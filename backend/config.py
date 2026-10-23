"""
Central configuration for the Dental Caries AI Assessment website.

Every model-specific number (thresholds, input sizes, class lists, crop padding, TTA variant, ICDAS window) is read
at runtime from the *_metadata.json files written by the training notebook
(Dental_Caries_Pipeline_Stage1_to_Stage4_with_FDI_Stage2.ipynb). The values below are only the notebook's own
Section 3 fall-backs, used exactly where the notebook uses them (Section 6.6 `_gvb_input_cfg`, `_icdas_input_cfg`,
`generate_cavity_crops`).
"""
import os

APP_TITLE = "Dental Caries AI Assessment System"
APP_SUBTITLE = "AI-assisted dental image analysis and caries assessment."
APP_VERSION = "1.0.0"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

# ---------------------------------------------------------------------------------------------------------------
# Where the trained models live. First existing folder that contains stage1_caries_model.keras wins:
#   1. environment variable DCAI_MODELS_DIR (set by the Colab launcher / guide)
#   2. <project>/models                     (copy the files here for an offline / local run)
#      or <repo>/models                     (where the Git LFS weights are committed)
#   3. the notebook's own output folder on Google Drive (Colab with Drive mounted)
# ---------------------------------------------------------------------------------------------------------------
DRIVE_BASE = "/content/drive/MyDrive/dentalcaries"
NOTEBOOK_OUT_SUBDIR = "pipeline_stage1_to_4_corrected"
MODELS_DIR_CANDIDATES = [
    os.environ.get("DCAI_MODELS_DIR", ""),
    os.path.join(PROJECT_ROOT, "models"),
    os.path.join(os.path.dirname(PROJECT_ROOT), "models"),   # repo-root models/ (Git LFS weights)
    f"{DRIVE_BASE}/{NOTEBOOK_OUT_SUBDIR}",
]

# File names exactly as written by notebook Section 4 (MODEL_PATHS) — do not rename the files.
MODEL_FILES = {
    "stage1": "stage1_caries_model.keras",        # CNN 1  Stage 1 healthy vs caries
    "fdi": "stage2_fdi_detector.keras",            # Stage 2 FDI tooth detector (optional in the notebook)
    "localizer": "cavity_localizer_model.keras",   # class-agnostic cavity / lesion localizer
    "gvb": "stage3_gvblack_model.keras",           # CNN 2  Stage 3 G.V. Black
    "icdas": "stage4_icdas_model.keras",           # CNN 3  Stage 4 ICDAS
}
META_FILES = {k: v.replace(".keras", "_metadata.json") for k, v in MODEL_FILES.items()}
MODEL_TITLES = {
    "stage1": "Stage 1 — Healthy vs Caries (CNN 1)",
    "fdi": "Stage 2 — FDI tooth detector",
    "localizer": "Cavity / lesion localizer",
    "gvb": "Stage 3 — G.V. Black classifier (CNN 2)",
    "icdas": "Stage 4 — ICDAS classifier (CNN 3)",
}
REQUIRED_MODELS = ("stage1", "localizer", "gvb", "icdas")   # notebook MODEL_KEYS
OPTIONAL_MODELS = ("fdi",)                                    # notebook OPTIONAL_MODEL_KEYS


def resolve_models_dir():
    """First candidate folder that contains the Stage 1 model; otherwise the first non-empty candidate (for messages)."""
    for d in MODELS_DIR_CANDIDATES:
        if d and os.path.isfile(os.path.join(d, MODEL_FILES["stage1"])):
            return d
    for d in MODELS_DIR_CANDIDATES:
        if d and os.path.isdir(d):
            return d
    return os.path.join(PROJECT_ROOT, "models")


# ---------------------------------------------------------------------------------------------------------------
# Notebook Section 3 fall-backs (used only when a metadata key is missing — same as the notebook)
# ---------------------------------------------------------------------------------------------------------------
GVB_FALLBACK = {"IMG_SIZE": 224, "DEFAULT_PAD": 0.5, "MIN_CROP_PX": 48}
ICDAS_FALLBACK = {"IMG_SIZE": 224, "DEFAULT_WINDOW": (2.5, 0.5)}
TOOTH_FRAC_FALLBACK = (0.33, 0.55)          # notebook: globals().get("TOOTH_FRAC", (0.33, 0.55))
LOC_FALLBACK = {"MIN_BOX_PX": 4}

# ---------------------------------------------------------------------------------------------------------------
# Upload validation (notebook Section 82)
# ---------------------------------------------------------------------------------------------------------------
ALLOWED_UPLOAD_EXT = (".jpg", ".jpeg", ".png")
ALLOWED_PIL_FORMATS = {"JPEG", "MPO", "PNG"}     # MPO = multi-picture JPEG written by many phone cameras
MIN_UPLOAD_PX = 64
MAX_UPLOAD_MB = 20                               # also set in .streamlit/config.toml (server.maxUploadSize)
MAX_IMAGE_PIXELS = 60_000_000                    # decompression-bomb guard

AGE_MIN, AGE_MAX = 1, 120                        # notebook ask_age(): whole number 1–120

DISCLAIMER = "Research proof of concept — not a clinical diagnosis. Please consult a dentist."
ACADEMIC_DISCLAIMER = (
    "This website is an academic AI proof of concept. Its outputs are model predictions from a photograph, "
    "not a clinical diagnosis, and it is not a substitute for examination by a qualified dentist."
)
