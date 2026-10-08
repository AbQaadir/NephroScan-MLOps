"""Schema package."""

from KDC.schema.api_schema import (
    HealthResponse,
    PredictBase64Request,
    PredictionResponse,
    TrainTriggerResponse,
)

__all__ = [
    "PredictBase64Request",
    "PredictionResponse",
    "HealthResponse",
    "TrainTriggerResponse",
]
