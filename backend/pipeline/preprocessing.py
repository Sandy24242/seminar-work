"""
Preprocessing contract — ported unchanged from notebook Section 6.1 (and the TTA helpers of Section 6.7).

One function per stage turns an image into model input; these are the same functions the notebook used to build the
training, evaluation and inference data. Every classifier contains its own normalisation layer, so classifier inputs
are plain RGB uint8 [0, 255]. Only the two grid detectors (localizer, FDI detector) use MobileNetV2 preprocess_input.
"""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb(source):
    """Path or bytes -> RGB uint8 with EXIF orientation applied. Used by EVERY stage (training + inference)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with Image.open(source) as im:
        return np.array(ImageOps.exif_transpose(im).convert("RGB"))


def mobilenet_v2_preprocess(x):
    """Same arithmetic as keras.applications.mobilenet_v2.preprocess_input (mode 'tf'): x / 127.5 - 1."""
    x = np.asarray(x, np.float32)
    x = x / 127.5
    x -= 1.0
    return x


# ---------- Stage 1 contract ----------
def s1_prepare(rgb, size):
    """EXIF-corrected RGB -> cv2.resize(size x size, INTER_AREA) -> uint8 [0,255]. Normalisation is inside the model."""
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


# ---------- Stage 3 contract (square context crop around a lesion box) ----------
def to_px(box, W, H):
    x1 = int(np.floor(box[0] * W)); y1 = int(np.floor(box[1] * H))
    x2 = int(np.ceil(box[2] * W));  y2 = int(np.ceil(box[3] * H))
    return max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)


def square_context_box(box, W, H, pad, min_px):
    """Normalised x1y1x2y2 box -> square pixel window centred on the box, side = longer box side * (1 + 2*pad)."""
    x1, y1, x2, y2 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(max(x2 - x1, y2 - y1) * (1 + 2 * pad), min_px) / 2
    return int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))


def crop_with_border(rgb, x1, y1, x2, y2):
    """Crop that may extend past the image edge; outside pixels are black, so the lesion stays centred."""
    H, W = rgb.shape[:2]
    out = np.zeros((max(y2 - y1, 1), max(x2 - x1, 1), 3), np.uint8)
    sx1, sy1, sx2, sy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = rgb[sy1:sy2, sx1:sx2]
    return out


def gvb_prepare_crop(rgb, box, pad, size, min_px):
    """Stage 3 contract: full-resolution RGB + normalised box -> square context crop -> resize -> uint8 [0,255]."""
    H, W = rgb.shape[:2]
    c = crop_with_border(rgb, *square_context_box(box, W, H, pad, min_px))
    interp = cv2.INTER_AREA if c.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(c, (size, size), interpolation=interp)


def letterbox(rgb, out_h, out_w):
    """Resize keeping the aspect ratio, centre on a black canvas (no distortion)."""
    H, W = rgb.shape[:2]
    s = min(out_h / H, out_w / W)
    nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
    r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.zeros((out_h, out_w, 3), np.uint8)
    t, l = (out_h - nh) // 2, (out_w - nw) // 2
    out[t:t + nh, l:l + nw] = r
    return out


# ---------- Stage 4 contract ----------
def icdas_prepare(rgb, size):
    """Stage 4 contract: tooth / lesion crop -> square letterbox (aspect kept, black padding) -> uint8 [0,255]."""
    return letterbox(rgb, size, size)


def icdas_window_box(box, W, H, mult, min_frac_w, min_frac_h):
    """Lesion box (normalised x1y1x2y2) -> tooth-scale pixel window centred on the lesion and kept inside the image.
    The ICDAS CNN is trained on TOOTH crops; a localizer box marks a LESION, so a window of tooth scale is cut around it."""
    x1, y1, x2, y2 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = float(min(max((x2 - x1) * mult, min_frac_w * W, 8.0), W))
    h = float(min(max((y2 - y1) * mult, min_frac_h * H, 8.0), H))
    left = int(round(np.clip(cx - w / 2, 0, W - w))); top = int(round(np.clip(cy - h / 2, 0, H - h)))
    return left, top, min(int(round(left + w)), W), min(int(round(top + h)), H)


def icdas_prepare_from_box(rgb, box, size, window):
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = icdas_window_box(box, W, H, window["mult"], window["min_frac_w"], window["min_frac_h"])
    return icdas_prepare(rgb[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)], size)


# ---------- localizer / FDI detector contract ----------
def loc_prepare(rgb, cache_side, input_size):
    x = cv2.resize(rgb, (cache_side, cache_side), interpolation=cv2.INTER_AREA).astype(np.float32)
    return mobilenet_v2_preprocess(cv2.resize(x, (input_size, input_size), interpolation=cv2.INTER_AREA))


def grid_centres(grid):
    c = (np.arange(grid, dtype=np.float32) + 0.5) / grid
    return np.meshgrid(c, c)


def _nms(boxes_yxyx, scores, max_det, iou):
    import tensorflow as tf
    return tf.image.non_max_suppression(boxes_yxyx, scores, max_det, iou_threshold=iou).numpy()


def decode_detections(raw, score_thr, nms_iou, max_det):
    """raw (G,G,5) -> boxes (N,4) normalised x1y1x2y2, scores (N,). NMS = tf.image.non_max_suppression."""
    g = raw.shape[0]; gx, gy = grid_centres(g)
    scores = (1 / (1 + np.exp(-raw[..., 0]))).reshape(-1)
    d = np.logaddexp(0.0, raw[..., 1:5]) / g
    boxes = np.clip(np.stack([gx - d[..., 0], gy - d[..., 1], gx + d[..., 2], gy + d[..., 3]], -1), 0, 1).reshape(-1, 4)
    idx = np.where(scores >= score_thr)[0]
    if idx.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    keep = _nms(boxes[idx][:, [1, 0, 3, 2]], scores[idx], max_det, nms_iou)
    return boxes[idx[keep]].astype(np.float32), scores[idx[keep]].astype(np.float32)


def decode_fdi_detections(raw, score_thr, nms_iou, max_det):
    """raw (G,G,5+C) -> boxes (N,4) normalised x1y1x2y2, objectness (N,), FDI class probabilities (N,C). Class-agnostic NMS."""
    g = raw.shape[0]; gx, gy = grid_centres(g); n_cls = raw.shape[-1] - 5
    scores = (1 / (1 + np.exp(-raw[..., 0]))).reshape(-1)
    d = np.logaddexp(0.0, raw[..., 1:5]) / g
    boxes = np.clip(np.stack([gx - d[..., 0], gy - d[..., 1], gx + d[..., 2], gy + d[..., 3]], -1), 0, 1).reshape(-1, 4)
    idx = np.where(scores >= score_thr)[0]
    if idx.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0, n_cls), np.float32)
    logits = raw[..., 5:].reshape(-1, n_cls)[idx].astype(np.float64)
    probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    keep = _nms(boxes[idx][:, [1, 0, 3, 2]], scores[idx], max_det, nms_iou)
    return boxes[idx[keep]].astype(np.float32), scores[idx[keep]].astype(np.float32), probs[keep].astype(np.float32)


def overlap_fraction(box, boxes):
    """Fraction of `box`'s area covered by each of `boxes` (all normalised x1y1x2y2)."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4); b = np.asarray(box, np.float32)
    iw = np.clip(np.minimum(b[2], boxes[:, 2]) - np.maximum(b[0], boxes[:, 0]), 0, None)
    ih = np.clip(np.minimum(b[3], boxes[:, 3]) - np.maximum(b[1], boxes[:, 1]), 0, None)
    return iw * ih / max(float((b[2] - b[0]) * (b[3] - b[1])), 1e-9)


# ---------- forward passes (Section 6.6 _forward, Section 6.7 tta_views / tta_forward) ----------
def forward(model, x_u8):
    import tensorflow as tf
    x = tf.convert_to_tensor(np.asarray(x_u8, np.float32)[None])
    return np.asarray(model(x, training=False), np.float64)[0]


def tta_views(xb, variant):
    """xb: float32 tensor (B,S,S,3) in [0,255]. Deterministic views; no randomness."""
    import tensorflow as tf
    views = [xb]
    if "hflip" in variant:
        views.append(tf.image.flip_left_right(xb))
    if "zoom" in variant:
        s = int(xb.shape[1]); views.append(tf.image.resize(tf.image.central_crop(xb, 0.9), (s, s), method="bilinear"))
    return views


def tta_forward(model, X_u8, variant="none"):
    import tensorflow as tf
    xb = tf.convert_to_tensor(np.asarray(X_u8, np.float32))
    return np.mean([np.asarray(model(v, training=False), np.float64) for v in tta_views(xb, variant)], axis=0)
