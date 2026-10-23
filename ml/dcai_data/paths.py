"""
Dataset and model locations. Everything lives on Google Drive; nothing here is committed to Git.

Defaults match the training notebook (MyDrive/dentalcaries/...). Override any path with an environment variable, e.g.
    DCAI_DATA_ROOT=/content/drive/MyDrive/dentalcaries   (folder that contains the dataset folders / ZIPs)
    DCAI_MODELS_DIR=/content/drive/MyDrive/dentalcaries/pipeline_stage1_to_4_corrected
"""
import os
from dataclasses import dataclass, field

DEFAULT_DATA_ROOT = "/content/drive/MyDrive/dentalcaries"

# Folder (or ZIP) names as they exist on Drive (verified via the Drive connector, 23 Oct 2026).
DATASETS = {
    # task -> (extracted folder name, archive name)
    "fdi": ("caries_FDI_dataset_final", "caries_FDI_dataset_final.zip"),
    "caries_views": ("caries", "caries.zip"),
    "healthy": ("healthy", "healthy.zip"),
    "icdas": ("icdascl", "icdascl.zip"),
    "lower_occlusal_yolo": ("lowoccyolo", "lowoccyolo (4).zip"),
}
MODELS_SUBDIR = "pipeline_stage1_to_4_corrected"


@dataclass
class DataPaths:
    root: str = field(default_factory=lambda: os.environ.get("DCAI_DATA_ROOT", DEFAULT_DATA_ROOT))

    def dataset_dir(self, key):
        env = os.environ.get(f"DCAI_{key.upper()}_DIR")
        return env or os.path.join(self.root, DATASETS[key][0])

    def dataset_zip(self, key):
        return os.path.join(self.root, DATASETS[key][1])

    @property
    def models_dir(self):
        return os.environ.get("DCAI_MODELS_DIR") or os.path.join(self.root, MODELS_SUBDIR)

    def describe(self):
        out = {"root": self.root, "models_dir": self.models_dir}
        for k in DATASETS:
            d, z = self.dataset_dir(k), self.dataset_zip(k)
            out[k] = {"dir": d, "dir_exists": os.path.isdir(d), "zip": z, "zip_exists": os.path.isfile(z)}
        return out
