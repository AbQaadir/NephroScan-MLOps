"""Unit tests for Prediction Pipeline and ModelManager."""

from unittest.mock import patch

import numpy as np
from PIL import Image

from KDC.pipeline.prediction import ModelManager, PredictionPipeline


def test_preprocess_image(sample_pil_image: Image.Image, sample_image_bytes: bytes):
    # From PIL Image
    arr_from_pil = PredictionPipeline.preprocess_image(sample_pil_image)
    assert arr_from_pil.shape == (1, 224, 224, 3)
    assert arr_from_pil.dtype == np.float32

    # From bytes
    arr_from_bytes = PredictionPipeline.preprocess_image(sample_image_bytes)
    assert arr_from_bytes.shape == (1, 224, 224, 3)
    assert arr_from_bytes.dtype == np.float32


def test_prediction_pipeline_predict_image(sample_image_bytes: bytes, mock_keras_model):
    pipeline = PredictionPipeline()
    with patch.object(ModelManager, "get_model", return_value=mock_keras_model):
        res = pipeline.predict_image(sample_image_bytes)
        assert res["prediction"] == "Tumor"
        assert res["confidence"] == 0.95
        assert res["class_index"] == 1
        assert "Tumor" in res["probabilities"]
        assert "Normal" in res["probabilities"]


def test_prediction_pipeline_legacy(sample_image_bytes: bytes, mock_keras_model, tmp_path):
    temp_img = tmp_path / "test.jpg"
    temp_img.write_bytes(sample_image_bytes)

    pipeline = PredictionPipeline(filename=str(temp_img))
    with patch.object(ModelManager, "get_model", return_value=mock_keras_model):
        res = pipeline.predict()
        assert isinstance(res, list)
        assert res[0] == {"image": "Tumor"}
