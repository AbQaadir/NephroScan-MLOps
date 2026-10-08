"""Unit tests for utility functions in src/KDC/utils/common.py."""

from pathlib import Path

from KDC.utils.common import (
    clean_base64_string,
    create_directories,
    decode_image_bytes,
    load_json,
    read_yaml,
    save_json,
)


def test_clean_base64_string():
    raw_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    with_prefix = f"data:image/png;base64,{raw_b64}"
    assert clean_base64_string(with_prefix) == raw_b64
    assert clean_base64_string(raw_b64) == raw_b64


def test_decode_image_bytes(sample_base64_image, sample_data_uri_base64):
    decoded_raw = decode_image_bytes(sample_base64_image)
    assert isinstance(decoded_raw, bytes)
    assert len(decoded_raw) > 0

    decoded_prefixed = decode_image_bytes(sample_data_uri_base64)
    assert decoded_raw == decoded_prefixed


def test_json_operations(tmp_path: Path):
    test_file = tmp_path / "test.json"
    data = {"metric": "accuracy", "value": 0.98}
    save_json(test_file, data)

    loaded = load_json(test_file)
    assert loaded.metric == "accuracy"
    assert loaded.value == 0.98


def test_create_directories(tmp_path: Path):
    target_dir = tmp_path / "nested" / "subfolder"
    create_directories([target_dir])
    assert target_dir.exists()
    assert target_dir.is_dir()


def test_read_yaml():
    config_path = Path("config/config.yaml")
    if config_path.exists():
        content = read_yaml(config_path)
        assert hasattr(content, "artifacts_root")
        assert content.artifacts_root == "artifacts"
