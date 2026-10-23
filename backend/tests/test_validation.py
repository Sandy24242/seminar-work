"""Upload validation: formats, corrupted / empty files, sizes, EXIF orientation."""
import io

import numpy as np
from PIL import Image

from pipeline.image_validation import safe_filename, validate_image_bytes


def _encode(arr, fmt, **kw):
    b = io.BytesIO(); Image.fromarray(arr).save(b, format=fmt, **kw); return b.getvalue()


IMG = (np.random.default_rng(0).integers(0, 255, (200, 300, 3))).astype(np.uint8)


def test_valid_png_and_jpeg():
    for fmt, name in (("PNG", "a.png"), ("JPEG", "b.jpg"), ("JPEG", "c.JPEG")):
        rgb, info, err = validate_image_bytes(_encode(IMG, fmt), name)
        assert err is None and rgb.shape == (200, 300, 3) and rgb.dtype == np.uint8
        assert info["width"] == 300 and info["height"] == 200


def test_empty_and_unsupported():
    assert validate_image_bytes(b"", "x.jpg")[2] == "the file is empty"
    assert "unsupported file type" in validate_image_bytes(_encode(IMG, "PNG"), "x.gif")[2]
    assert "unsupported file type" in validate_image_bytes(_encode(IMG, "PNG"), "noext")[2]


def test_wrong_content_and_corruption():
    assert "not a readable image" in validate_image_bytes(b"this is not an image at all", "x.jpg")[2]
    gif = _encode(IMG, "GIF")
    assert "not JPEG / PNG" in validate_image_bytes(gif, "renamed.png")[2]
    jpg = _encode(IMG, "JPEG")
    err = validate_image_bytes(jpg[: len(jpg) // 3], "truncated.jpg")[2]
    assert err is not None and ("corrupted" in err or "not a readable" in err)


def test_too_small():
    err = validate_image_bytes(_encode(IMG[:40, :40], "PNG"), "small.png")[2]
    assert "too small" in err


def test_exif_orientation_applied():
    exif = Image.Exif(); exif[0x0112] = 6        # rotate 90° CW on display
    b = io.BytesIO(); Image.fromarray(IMG).save(b, format="JPEG", exif=exif)
    rgb, info, err = validate_image_bytes(b.getvalue(), "phone.jpg")
    assert err is None and rgb.shape[:2] == (300, 200) and info["exif_rotated"]


def test_rgba_and_grayscale_converted_to_rgb():
    rgba = np.dstack([IMG, np.full(IMG.shape[:2], 255, np.uint8)])
    assert validate_image_bytes(_encode(rgba, "PNG"), "a.png")[0].shape[2] == 3
    assert validate_image_bytes(_encode(IMG[..., 0], "PNG"), "g.png")[0].shape[2] == 3


def test_safe_filename():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("my photo (1).jpg") == "my_photo_1_.jpg"
    assert safe_filename("") == "image"
