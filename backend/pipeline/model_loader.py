"""
Model registry — notebook Section 6.6 `load_stage` / `load_all_models`, without retraining and without globals.

Each stage = one .keras file + its *_metadata.json, loaded with keras.models.load_model(path, compile=False) exactly as
in the notebook (`load_keras`). A missing or unloadable file is recorded as UNAVAILABLE with the reason; nothing is ever
substituted. Stage 1, the localizer, Stage 3 and Stage 4 are required for the caries cascade; Stage 2 (FDI) is optional,
as in the notebook.
"""
import json
import os
import time

import config


class ModelRegistry:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.models, self.meta, self.unavailable, self.load_seconds = {}, {}, {}, {}
        self.versions = {}

    # ---------- loading ----------
    def _read_meta(self, stage):
        p = os.path.join(self.models_dir, config.META_FILES[stage])
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
        pm = os.path.join(self.models_dir, "pipeline_metadata.json")   # notebook fallback: read_metadata().get(stage)
        if os.path.isfile(pm):
            with open(pm) as f:
                return json.load(f).get(stage, {})
        return {}

    def load_stage(self, stage):
        path = os.path.join(self.models_dir, config.MODEL_FILES[stage])
        try:
            meta = self._read_meta(stage)
        except (OSError, ValueError) as e:
            self.unavailable[stage] = f"metadata file could not be read ({type(e).__name__}: {e})"
            return False
        if not meta:
            self.unavailable[stage] = f"metadata file {config.META_FILES[stage]} not found"
            return False
        if not os.path.isfile(path):
            self.unavailable[stage] = f"model file {config.MODEL_FILES[stage]} not found"
            return False
        t0 = time.time()
        try:
            import keras
            model = keras.models.load_model(path, compile=False)
        except Exception as e:  # incompatible Keras/TensorFlow version, truncated download, ...
            self.unavailable[stage] = f"model file could not be loaded ({type(e).__name__}: {str(e)[:300]})"
            return False
        self.models[stage], self.meta[stage] = model, meta
        self.unavailable.pop(stage, None)
        self.load_seconds[stage] = round(time.time() - t0, 1)
        return True

    def load_all(self, progress=None):
        try:
            import keras
            import tensorflow as tf
            self.versions = {"tensorflow": tf.__version__, "keras": keras.__version__}
        except Exception as e:
            for k in config.REQUIRED_MODELS + config.OPTIONAL_MODELS:
                self.unavailable[k] = f"TensorFlow / Keras could not be imported ({type(e).__name__}: {e})"
            return self
        for k in config.REQUIRED_MODELS + config.OPTIONAL_MODELS:
            if progress:
                progress(k)
            if k not in self.models:
                self.load_stage(k)
        return self

    # ---------- status ----------
    def has(self, stage):
        return stage in self.models

    @property
    def missing_required(self):
        return [k for k in config.REQUIRED_MODELS if k not in self.models]

    def summary(self):
        """Model availability table (no absolute server paths are exposed)."""
        rows = []
        for k in ("stage1", "fdi", "localizer", "gvb", "icdas"):
            m = self.meta.get(k, {})
            detail = {
                "stage1": lambda: (f"{m.get('backbone')} · {m.get('image_size')} px · threshold {m.get('threshold', float('nan')):.3f}"
                                   f" ± {m.get('uncertain_margin', float('nan')):.2f} · confidence ≥ {m.get('confidence_threshold', float('nan')):.2f}"),
                "fdi": lambda: (f"{len(m.get('class_names', []))} FDI classes · score ≥ {m.get('score_threshold', float('nan')):.2f}"
                                f" · number confidence ≥ {m.get('class_confidence_threshold', float('nan')):.2f}"),
                "localizer": lambda: f"input {m.get('input_size')} px · score ≥ {m.get('score_threshold', float('nan')):.2f}",
                "gvb": lambda: (f"{m.get('backbone')} · {m.get('image_size')} px · pad {m.get('context_pad')} · TTA {m.get('tta', 'none')}"
                                f" · classes {len(m.get('class_names', []))}"),
                "icdas": lambda: (f"{m.get('backbone')} · {m.get('image_size')} px · scores {m.get('scores')} · TTA {m.get('tta', 'none')}"),
            }[k]
            rows.append({"Model": config.MODEL_TITLES[k], "File": config.MODEL_FILES[k],
                         "Status": "Loaded" if k in self.models else "Unavailable",
                         "Details": detail() if k in self.models else self.unavailable.get(k, "not loaded"),
                         "Required": "yes" if k in config.REQUIRED_MODELS else "optional",
                         "Load time (s)": self.load_seconds.get(k)})
        return rows
