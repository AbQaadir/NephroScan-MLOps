"""Unit tests for PyTorch models and export."""

import pytest


def test_pytorch_model_architecture():
    torch = pytest.importorskip("torch")
    from KDC.models.pytorch_model import KidneyPyTorchClassifier

    model = KidneyPyTorchClassifier(num_classes=2, backbone_name="resnet18", pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)

    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()


def test_pytorch_model_export(tmp_path):
    torch = pytest.importorskip("torch")
    from KDC.models.pytorch_model import KidneyPyTorchClassifier

    model = KidneyPyTorchClassifier(num_classes=2, backbone_name="resnet18", pretrained=False)
    script_path = str(tmp_path / "model.pt")
    model.export_torchscript(script_path)

    loaded = torch.jit.load(script_path)
    dummy_input = torch.randn(1, 3, 224, 224)
    out = loaded(dummy_input)
    assert out.shape == (1, 2)
