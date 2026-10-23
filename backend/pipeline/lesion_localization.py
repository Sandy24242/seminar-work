"""
Cavity / lesion localization + per-lesion crop generation — notebook Section 6.6 `localize`, `_gvb_input_cfg`,
`_icdas_input_cfg`, `generate_cavity_crops`. The localizer is class-agnostic: it only answers WHERE a lesion is.
"""
import numpy as np

import config
from pipeline.preprocessing import (decode_detections, forward, gvb_prepare_crop, icdas_prepare_from_box,
                                    icdas_window_box, loc_prepare, to_px)


def localize(model, meta, rgb, score_threshold=None):
    thr = meta["score_threshold"] if score_threshold is None else float(score_threshold)
    raw = forward(model, loc_prepare(rgb, meta["cache_side"], meta["input_size"]))
    return decode_detections(raw, thr, meta["nms_iou"], meta["max_detections"])


def gvb_input_cfg(gvb_meta):
    m = gvb_meta or {}
    return (m.get("image_size", config.GVB_FALLBACK["IMG_SIZE"]), m.get("context_pad", config.GVB_FALLBACK["DEFAULT_PAD"]),
            m.get("min_crop_px", config.GVB_FALLBACK["MIN_CROP_PX"]))


def icdas_input_cfg(icdas_meta):
    m = icdas_meta or {}
    w = m.get("inference_window")
    if w is None:
        mult, scale = config.ICDAS_FALLBACK["DEFAULT_WINDOW"]
        fw, fh = config.TOOTH_FRAC_FALLBACK
        w = {"mult": mult, "min_scale": scale, "min_frac_w": scale * fw, "min_frac_h": scale * fh, "source": "config default"}
    return m.get("image_size", config.ICDAS_FALLBACK["IMG_SIZE"]), w


def generate_cavity_crops(rgb, boxes, scores, gvb_meta, icdas_meta, localizer_meta):
    """For EACH localized lesion: G.V. Black context crop + ICDAS tooth-scale crop (boxes below min_box_px are dropped)."""
    scores = np.ones(len(boxes), np.float32) if scores is None else scores
    H, W = rgb.shape[:2]
    gsize, pad, min_px = gvb_input_cfg(gvb_meta); isize, window = icdas_input_cfg(icdas_meta)
    min_box = (localizer_meta or {}).get("min_box_px", config.LOC_FALLBACK["MIN_BOX_PX"])
    out = []
    for b, sc in zip(boxes, scores):
        x1, y1, x2, y2 = to_px(b, W, H)
        if x2 - x1 < min_box or y2 - y1 < min_box:
            continue
        out.append({"box_norm": [float(v) for v in b], "box_xyxy_px": [x1, y1, x2, y2], "localizer_score": float(sc),
                    "gvb_crop": gvb_prepare_crop(rgb, np.asarray(b, np.float32), pad, gsize, min_px),
                    "icdas_window_px": list(icdas_window_box(b, W, H, window["mult"], window["min_frac_w"], window["min_frac_h"])),
                    "icdas_crop": icdas_prepare_from_box(rgb, np.asarray(b, np.float32), isize, window)})
    return out
