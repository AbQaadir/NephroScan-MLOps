"""Production Prediction Pipeline for Kidney Disease Classification."""

import io
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from KDC import logger


class ModelManager:
    """Singleton model loader and cache manager."""

    _model: Optional[Any] = None
    _model_path: Optional[str] = None

    @classmethod
    def get_model(cls, model_path: Optional[str] = None) -> Any:
        # Default fallback candidate paths
        candidate_paths = [
            model_path,
            os.path.join("model", "model.h5"),
            os.path.join("artifacts", "training", "model.h5"),
        ]
        target_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                target_path = p
                break

        if cls._model is not None and (target_path == cls._model_path or target_path is None):
            return cls._model

        if target_path is None:
            raise FileNotFoundError(
                f"Model file not found. Looked in: {[p for p in candidate_paths if p]}"
            )

        logger.info(f"Loading Keras classification model from: {target_path}")
        from tensorflow.keras.models import load_model

        cls._model = load_model(target_path)
        cls._model_path = target_path
        return cls._model

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None

    @classmethod
    def get_loaded_path(cls) -> Optional[str]:
        return cls._model_path


class PredictionPipeline:
    """Robust prediction pipeline supporting both files and in-memory images."""

    CLASS_NAMES = ["Normal", "Tumor"]

    def __init__(
        self, filename: Optional[Union[str, Path]] = None, model_path: Optional[str] = None
    ):
        self.filename = str(filename) if filename else None
        self.model_path = model_path

    @staticmethod
    def preprocess_image(
        image_input: Union[str, Path, bytes, Image.Image],
        target_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Preprocesses an image from path, bytes, or PIL Image into model input format."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Ensure RGB format
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to target size (224, 224)
        img = img.resize(target_size, Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32)

        # Expand batch dimension: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
    ) -> Dict[str, Any]:
        """Runs inference on an image and returns a structured dictionary."""
        model = ModelManager.get_model(self.model_path)
        processed_img = self.preprocess_image(image_input)

        raw_preds = model.predict(processed_img, verbose=0)
        # raw_preds shape can be (1, 2) for binary softmax or (1, 1) for sigmoid
        if raw_preds.shape[-1] == 1:
            # Binary sigmoid
            prob_tumor = float(raw_preds[0][0])
            prob_normal = 1.0 - prob_tumor
            probs = [prob_normal, prob_tumor]
            class_idx = 1 if prob_tumor >= 0.5 else 0
        else:
            # Softmax / multi-class
            probs = [float(p) for p in raw_preds[0]]
            class_idx = int(np.argmax(probs))

        confidence = float(probs[class_idx])
        predicted_label = self.CLASS_NAMES[class_idx]

        probabilities_dict = {
            self.CLASS_NAMES[i]: round(probs[i], 4) for i in range(len(self.CLASS_NAMES))
        }

        logger.info(
            f"Prediction completed: label={predicted_label}, confidence={confidence:.4f}, class_idx={class_idx}"
        )

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "class_index": class_idx,
            "probabilities": probabilities_dict,
            "status": "success",
        }

    def predict(self) -> list:
        """Legacy-compatible prediction method returning [{'image': prediction}]."""
        if not self.filename:
            raise ValueError("Filename was not supplied to PredictionPipeline instance.")

        res = self.predict_image(self.filename)
        return [{"image": res["prediction"]}]
