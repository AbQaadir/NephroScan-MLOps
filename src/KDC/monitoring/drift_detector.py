"""Image data drift and quality monitoring for CT scan inference streams."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from KDC import logger


def compute_ks_2samp(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """Computes two-sample Kolmogorov-Smirnov statistic and asymptotic p-value using NumPy."""
    data1 = np.sort(sample1)
    data2 = np.sort(sample2)
    n1 = len(data1)
    n2 = len(data2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side="right") / n1
    cdf2 = np.searchsorted(data2, data_all, side="right") / n2
    d_stat = float(np.max(np.abs(cdf1 - cdf2)))

    # Asymptotic approximation of two-sample KS test p-value
    en = np.sqrt((n1 * n2) / (n1 + n2))
    p_val = float(2.0 * np.exp(-2.0 * ((en * d_stat) ** 2)))
    p_val = float(np.clip(p_val, 0.0, 1.0))
    return round(d_stat, 4), round(p_val, 4)


class ImageDriftDetector:
    """Monitors statistical properties of incoming CT scans against a reference baseline."""

    def __init__(self, reference_stats: Optional[Dict[str, List[float]]] = None):
        # Reference distribution metrics: brightness, contrast
        self.reference_stats = reference_stats or {
            "brightness": [120.0, 135.0, 110.0, 128.0, 125.0],
            "contrast": [55.0, 60.0, 48.0, 52.0, 50.0],
        }
        self.inference_window: Dict[str, List[float]] = {
            "brightness": [],
            "contrast": [],
        }

    @staticmethod
    def extract_metrics(img_array: np.ndarray) -> Dict[str, float]:
        """Extracts statistical summary metrics from an image array."""
        brightness = float(np.mean(img_array))
        contrast = float(np.std(img_array))

        return {
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
        }

    def record_inference_image(self, img: Image.Image) -> Dict[str, float]:
        """Records metrics from a single incoming inference image into current window."""
        arr = np.array(img.convert("L"), dtype=np.float32)
        metrics = self.extract_metrics(arr)

        for key, val in metrics.items():
            self.inference_window[key].append(val)

        return metrics

    def check_drift(self, p_value_threshold: float = 0.05) -> Dict[str, Any]:
        """Runs two-sample KS test between baseline distribution and current inference window."""
        drift_results = {}
        drift_detected = False

        for metric_name, ref_values in self.reference_stats.items():
            current_values = self.inference_window.get(metric_name, [])
            if len(current_values) < 5:
                drift_results[metric_name] = {
                    "status": "insufficient_data",
                    "sample_count": len(current_values),
                }
                continue

            stat, p_val = compute_ks_2samp(ref_values, current_values)
            is_drifted = bool(p_val < p_value_threshold)
            if is_drifted:
                drift_detected = True

            drift_results[metric_name] = {
                "ks_statistic": stat,
                "p_value": p_val,
                "drift_detected": is_drifted,
            }

        logger.info(f"Drift check completed. Overall drift detected: {drift_detected}")
        return {
            "drift_detected": drift_detected,
            "metrics": drift_results,
        }
