"""
Upload validation — notebook Section 82 `validate_input_image`, adapted to work on the uploaded bytes in memory
(the website never writes the patient's photograph to disk).

Checks: non-empty, allowed extension, real file format (JPEG / MPO / PNG), full decode, EXIF orientation, RGB,
minimum size. Returns (rgb, info, None) or (None, None, reason).
"""
import io
import os
import re

from PIL import Image, UnidentifiedImageError

import config
from pipeline.preprocessing import load_rgb

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS


def safe_filename(name, fallback="image"):
    """Strip any path and unsafe characters (same rule as the notebook's _safe_upload_name, without the timestamp)."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(str(name or ""))).strip("._")
    return base[:80] or fallback


def validate_image_bytes(data, filename):
    if data is None or len(data) == 0:
        return None, None, "the file is empty"
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        return None, None, f"the file is larger than {config.MAX_UPLOAD_MB} MB"
    ext = os.path.splitext(str(filename or ""))[1].lower()
    if ext not in config.ALLOWED_UPLOAD_EXT:
        return None, None, f"unsupported file type '{ext or '(none)'}' — please upload a JPG, JPEG or PNG image"
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            stored_size = im.size
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        return None, None, f"not a readable image ({type(e).__name__}: {e})"
    if fmt not in config.ALLOWED_PIL_FORMATS:
        return None, None, f"the file content is {fmt}, not JPEG / PNG"
    try:
        rgb = load_rgb(bytes(data))                      # EXIF orientation applied, converted to RGB (same loader as training)
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        return None, None, f"the image is corrupted and could not be decoded ({type(e).__name__}: {e})"
    if min(rgb.shape[:2]) < config.MIN_UPLOAD_PX:
        return None, None, (f"the image is too small ({rgb.shape[1]}x{rgb.shape[0]} px; "
                            f"at least {config.MIN_UPLOAD_PX} px per side)")
    info = {"filename": safe_filename(filename), "format": fmt, "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
            "stored_width": int(stored_size[0]), "stored_height": int(stored_size[1]),
            "exif_rotated": tuple(stored_size) != (rgb.shape[1], rgb.shape[0]), "size_kb": round(len(data) / 1024, 1)}
    return rgb, info, None
