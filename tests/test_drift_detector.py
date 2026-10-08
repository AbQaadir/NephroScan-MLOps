"""Unit tests for ImageDriftDetector."""

import numpy as np
from PIL import Image

from KDC.monitoring.drift_detector import ImageDriftDetector


def test_extract_metrics():
    arr = np.ones((100, 100), dtype=np.float32) * 50
    metrics = ImageDriftDetector.extract_metrics(arr)
    assert metrics["brightness"] == 50.0
    assert metrics["contrast"] == 0.0


def test_record_inference_and_drift_check(sample_pil_image: Image.Image):
    detector = ImageDriftDetector(
        reference_stats={
            "brightness": [100.0, 102.0, 98.0, 101.0, 99.0],
            "contrast": [10.0, 12.0, 11.0, 10.5, 11.5],
        }
    )

    # Record 6 synthetic images
    for _ in range(6):
        detector.record_inference_image(sample_pil_image)

    result = detector.check_drift()
    assert "drift_detected" in result
    assert "metrics" in result
    assert "brightness" in result["metrics"]
    assert "contrast" in result["metrics"]
