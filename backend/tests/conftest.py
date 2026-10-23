import os
import sys

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from standin_models import write_standin_models  # noqa: E402


@pytest.fixture(scope="session")
def standin_dir(tmp_path_factory):
    """Folder with ALL five stand-in models + metadata (test fixtures — not the trained models)."""
    return write_standin_models(str(tmp_path_factory.mktemp("standin_models")))


@pytest.fixture(scope="session")
def registry(standin_dir):
    from pipeline.model_loader import ModelRegistry
    reg = ModelRegistry(standin_dir).load_all()
    assert not reg.unavailable, reg.unavailable
    return reg
