"""PyTorch deep learning architectures and research pipelines for Kidney CT scan classification."""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

from KDC import logger


class KidneyPyTorchClassifier(nn.Module):
    """Deep learning transfer-learning model for Kidney Disease Classification using PyTorch.

    Supports ResNet and VGG backbones with customized classification heads,
    dropout for regularization, and export to TorchScript / ONNX.
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        elif self.backbone_name == "vgg16":
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            backbone = models.vgg16(weights=weights)
            in_features = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. Choose 'resnet18' or 'vgg16'."
            )

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def export_onnx(
        self, output_path: str, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)
    ):
        """Exports the trained model to ONNX format for ultra-fast production inference."""
        self.eval()
        dummy_input = torch.randn(*input_size)
        torch.onnx.export(
            self,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info(f"PyTorch model exported successfully to ONNX at {output_path}")

    def export_torchscript(
        self, output_path: str, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)
    ):
        """Exports the trained model to TorchScript (.pt) format."""
        self.eval()
        dummy_input = torch.randn(*input_size)
        traced_model = torch.jit.trace(self, dummy_input)
        traced_model.save(output_path)
        logger.info(f"TorchScript model saved at {output_path}")
