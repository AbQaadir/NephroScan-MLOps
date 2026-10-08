"""API Request and Response Pydantic Schemas for KDC Service."""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictBase64Request(BaseModel):
    """Payload for Base64 encoded image inference."""

    image: str = Field(
        ..., description="Base64 encoded image string (with or without data URI prefix)"
    )


class PredictionResponse(BaseModel):
    """Standardized response schema for predictions."""

    prediction: str = Field(..., description="Predicted class name, e.g. Normal or Tumor")
    confidence: float = Field(..., description="Probability confidence between 0.0 and 1.0")
    class_index: int = Field(..., description="Predicted class index (0 for Normal, 1 for Tumor)")
    probabilities: Dict[str, float] = Field(
        default_factory=dict, description="Per-class predicted probabilities"
    )
    status: str = Field(default="success", description="Prediction status flag")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="healthy", description="Service health state")
    model_loaded: bool = Field(..., description="Whether the classification model is loaded")
    version: str = Field(default="1.0.0", description="API version")
    model_path: Optional[str] = Field(default=None, description="Path to loaded model")


class TrainTriggerResponse(BaseModel):
    """Response returned upon training pipeline trigger."""

    status: str = Field(default="accepted", description="Status of training task initiation")
    message: str = Field(..., description="Execution status or task details")
