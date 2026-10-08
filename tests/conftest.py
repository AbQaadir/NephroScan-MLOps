"""Pytest Fixtures and test utilities."""

import base64
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure root and src are on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import app
from KDC.pipeline.prediction import ModelManager


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Creates a synthetic 224x224 RGB test image."""
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    return img


@pytest.fixture
def sample_image_bytes(sample_pil_image: Image.Image) -> bytes:
    """Returns JPEG encoded bytes of sample image."""
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_base64_image(sample_image_bytes: bytes) -> str:
    """Returns base64 encoded string of sample image."""
    return base64.b64encode(sample_image_bytes).decode("utf-8")


@pytest.fixture
def sample_data_uri_base64(sample_base64_image: str) -> str:
    """Returns base64 string with data URI prefix."""
    return f"data:image/jpeg;base64,{sample_base64_image}"


@pytest.fixture
def mock_keras_model():
    """Mocks a Keras model predicting tumor probability."""
    mock_model = MagicMock()
    # Return mock prediction shape (1, 2) -> [Normal=0.05, Tumor=0.95]
    mock_model.predict.return_value = np.array([[0.05, 0.95]], dtype=np.float32)
    return mock_model


@pytest.fixture
def test_client(mock_keras_model):
    """Provides FastAPI test client with mocked model."""
    with patch.object(ModelManager, "get_model", return_value=mock_keras_model):
        with patch.object(ModelManager, "is_loaded", return_value=True):
            with patch.object(ModelManager, "get_loaded_path", return_value="model/model.h5"):
                client = TestClient(app)
                yield client
