"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient


def test_health_endpoint(test_client: TestClient):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "ready"]
    assert "model_loaded" in data
    assert data["version"] == "1.0.0"


def test_ready_endpoint(test_client: TestClient):
    response = test_client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_home_page(test_client: TestClient):
    response = test_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_base64_endpoint(test_client: TestClient, sample_base64_image: str):
    response = test_client.post(
        "/predict",
        json={"image": sample_base64_image},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["prediction"] == "Tumor"
    assert data[0]["confidence"] == 0.95
    assert data[0]["image"] == "Tumor"


def test_predict_api_v1(test_client: TestClient, sample_base64_image: str):
    response = test_client.post(
        "/api/v1/predict",
        json={"image": sample_base64_image},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Tumor"
    assert data["confidence"] == 0.95
    assert data["class_index"] == 1
    assert "probabilities" in data
    assert data["status"] == "success"


def test_predict_upload_file(test_client: TestClient, sample_image_bytes: bytes):
    response = test_client.post(
        "/predict/upload",
        files={"file": ("test_ct.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Tumor"
    assert data["confidence"] == 0.95


def test_predict_invalid_image(test_client: TestClient):
    response = test_client.post(
        "/predict",
        json={"image": "not-a-valid-base64-image-payload"},
    )
    assert response.status_code == 422


def test_trigger_training(test_client: TestClient):
    response = test_client.post("/train")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
