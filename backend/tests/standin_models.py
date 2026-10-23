"""
TEST FIXTURES ONLY — hand-built stand-in models with the same file names, input shapes, output shapes and metadata keys
as the real trained models. They are NOT trained and their predictions mean nothing clinically. They exist so the
loading, preprocessing, decoding, association, reporting and UI code paths can be tested where the real weights are not
available. The website never creates or loads these files on its own; tests write them to a temporary folder.

Colour conventions of the synthetic test photo (make_synthetic_photo):
  gum background   dark red        -> no tooth, no lesion
  teeth            light tints     -> FDI detector objectness; tint selects the FDI class (pink -> '11', green -> '21', blue -> '12')
  lesion           yellow-brown    -> localizer objectness
  Stage 1 stand-in: P(caries) = sigmoid(k * (mean R - mean B)) -> red-dominant photo = caries, blue = healthy, grey = uncertain.
"""
import json
import os

import numpy as np

FDI_NAMES = ["11", "21", "12", "22", "13", "23", "14", "24", "15", "25", "16", "26", "17",
             "31", "41", "32", "42", "33", "43", "34", "44", "35", "45", "36", "46", "37"]   # 26 classes like the real detector

META = {
    "stage1": {"model": "TEST STAND-IN", "backbone": "stand-in", "image_size": 224, "class_names": ["Healthy", "Caries"],
               "class_mapping": {"0": "Healthy", "1": "Caries"}, "threshold": 0.6029214859008789, "uncertain_margin": 0.07,
               "confidence_threshold": 0.6},
    "localizer": {"model": "TEST STAND-IN", "input_size": 416, "cache_side": 520, "stride": 16, "grid": 26,
                  "score_threshold": 0.40, "nms_iou": 0.40, "max_detections": 10, "min_box_px": 4},
    "fdi": {"model": "TEST STAND-IN", "class_names": FDI_NAMES, "num_classes": 26, "input_size": 512, "cache_side": 640, "stride": 16,
            "grid": 32, "score_threshold": 0.55, "class_confidence_threshold": 0.50, "nms_iou": 0.45, "max_detections": 40,
            "assoc_min_overlap": 0.30, "assoc_ambiguity_ratio": 0.80, "unreliable_classes": ["37"],
            "test_metrics": {"mAP50": 0.5, "at_threshold_number_accuracy_on_matched": 0.5}},
    "gvb": {"model": "TEST STAND-IN", "backbone": "stand-in", "class_names": ["GVB_Class_I", "GVB_Class_II", "GVB_Class_III",
            "GVB_Class_IV", "GVB_Class_V", "GVB_Class_VI"], "image_size": 224, "context_pad": 0.5, "min_crop_px": 48, "tta": "zoom",
            "confidence_threshold": 0.50},
    "icdas": {"model": "TEST STAND-IN", "backbone": "stand-in", "class_names": ["ICDAS_1", "ICDAS_2", "ICDAS_3", "ICDAS_4", "ICDAS_5", "ICDAS_6"],
              "scores": [1, 2, 3, 4, 5, 6], "image_size": 224, "tta": "hflip", "confidence_threshold": 0.50, "unreliable_scores": [4],
              "inference_window": {"mult": 1.5, "min_scale": 1.0, "min_frac_w": 0.12, "min_frac_h": 0.3}},
}
FILES = {"stage1": "stage1_caries_model.keras", "fdi": "stage2_fdi_detector.keras", "localizer": "cavity_localizer_model.keras",
         "gvb": "stage3_gvblack_model.keras", "icdas": "stage4_icdas_model.keras"}


def _stage1():
    import keras
    from keras import layers
    inp = keras.Input((224, 224, 3), name="image_rgb_0_255")
    x = layers.GlobalAveragePooling2D()(inp)
    out = layers.Dense(1, activation="sigmoid", name="probabilities")(x)
    m = keras.Model(inp, out)
    m.layers[-1].set_weights([np.array([[0.05], [0.0], [-0.05]], np.float32), np.zeros(1, np.float32)])
    return m


def _grid_detector(size, n_out, kernel, bias):
    import keras
    from keras import layers
    inp = keras.Input((size, size, 3), name="image")
    x = layers.AveragePooling2D(16)(inp)
    out = layers.Conv2D(n_out, 1, name="head")(x)
    m = keras.Model(inp, out)
    m.get_layer("head").set_weights([kernel.reshape(1, 1, 3, n_out).astype(np.float32), bias.astype(np.float32)])
    return m


def _localizer():
    k = np.zeros((3, 5)); k[:, 0] = (-2.0, 8.0, -12.0)            # objectness = 10*(G-B) - 6*mean(RGB) - 4 (dark yellow-brown)
    b = np.array([-4.0, 1.5, 2.0, 1.5, 2.0])                      # box half-size ~ 1.7 x 2.1 cells
    return _grid_detector(416, 5, k, b)


def _fdi():
    n = 5 + len(FDI_NAMES)
    k = np.zeros((3, n)); k[:, 0] = 20.0 / 3                       # objectness = 20*mean(RGB) - 10 (bright tooth centre)
    k[0, 5], k[1, 6], k[2, 7] = 8.0, 8.0, 8.0                       # class '11' <- R, '21' <- G, '12' <- B
    b = np.full(n, -10.0); b[0] = -10.0; b[5:8] = 0.0
    b[1], b[2], b[3], b[4] = 3.98, 8.0, 3.98, 8.0                   # tooth half-size ~ 4 x 8 grid cells
    return _grid_detector(512, n, k, b)


def _classifier(seed):
    import keras
    from keras import layers
    inp = keras.Input((224, 224, 3), name="image_rgb_0_255")
    x = layers.GlobalAveragePooling2D()(inp)
    out = layers.Dense(6, activation="softmax", name="probabilities")(x)
    m = keras.Model(inp, out)
    rng = np.random.default_rng(seed)
    m.layers[-1].set_weights([rng.normal(0, 0.03, (3, 6)).astype(np.float32), rng.normal(0, 0.5, 6).astype(np.float32)])
    return m


BUILDERS = {"stage1": _stage1, "localizer": _localizer, "fdi": _fdi, "gvb": lambda: _classifier(3), "icdas": lambda: _classifier(4)}


def write_standin_models(folder, stages=("stage1", "fdi", "localizer", "gvb", "icdas"), meta_overrides=None):
    os.makedirs(folder, exist_ok=True)
    for s in stages:
        BUILDERS[s]().save(os.path.join(folder, FILES[s]))
        meta = {**META[s], **((meta_overrides or {}).get(s, {}))}
        with open(os.path.join(folder, FILES[s].replace(".keras", "_metadata.json")), "w") as f:
            json.dump(meta, f, indent=2)
    return folder


def _blob(img, cx, cy, rx, ry, color):
    """Soft elliptical blob (weight 1 at the centre -> 0 at the rim), so the stand-in detectors peak at the centre."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.clip(1 - ((xx - cx) / rx) ** 2 - ((yy - cy) / ry) ** 2, 0, 1)[..., None]
    img[:] = (img * (1 - g) + np.array(color, np.float32) * g).astype(np.uint8)


TEETH = [(140, 225, "11", (255, 190, 190)), (310, 225, "21", (190, 255, 190)), (480, 225, "12", (190, 190, 255))]


def make_synthetic_photo(kind="caries", W=640, H=420):
    """Synthetic 'mouth' for tests.
    kind: caries (lesions on '11' and '12') | healthy | uncertain | caries_no_lesion | spanning (lesion between '11' and '21')
          | gum_lesion (lesion outside every tooth)."""
    img = np.zeros((H, W, 3), np.uint8)
    if kind == "healthy":
        img[:] = (40, 60, 200)                     # blue-dominant -> stand-in Stage 1 says HEALTHY
        return img
    if kind == "uncertain":
        img[:] = (128, 128, 128)                   # P(caries)=0.5 -> UNCERTAIN
        return img
    img[:] = (180, 40, 40)                          # gum, red-dominant -> CARIES
    for cx, cy, _, c in TEETH:
        _blob(img, cx, cy, 82, 108, c)
    lesion = (150, 100, 10)
    if kind == "caries":
        _blob(img, 140, 240, 32, 32, lesion); _blob(img, 480, 210, 32, 32, lesion)
    elif kind == "spanning":
        _blob(img, 225, 225, 32, 32, lesion)
    elif kind == "gum_lesion":
        _blob(img, 600, 40, 32, 32, lesion)
    return img


def to_png_bytes(rgb):
    import io
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(rgb).save(b, format="PNG"); return b.getvalue()
