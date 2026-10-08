# NephroScan MLOps 🔬

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.0%2B-945DD6.svg?logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **NephroScan MLOps** is an enterprise-grade, end-to-end Machine Learning Operations platform designed for clinical CT scan classification (Normal vs. Kidney Tumor). It bridges cutting-edge deep learning research (**PyTorch** and **TensorFlow**) with high-performance production serving (**FastAPI**), reproducible pipelines (**DVC**), experiment tracking (**MLflow**), containerization (**Docker & Compose**), automated **CI/CD**, and production observability (**Prometheus & Drift Detection**).

---

## 📐 System Architecture

```mermaid
flowchart TD
    subgraph Data_Pipeline ["1. Data Versioning & Experimentation (DVC + MLflow)"]
        RawData["Raw CT Scans (DVC Remote)"] --> Ingest["Data Ingestion"]
        Ingest --> Prep["Base Model Preparation"]
        Prep --> TrainTF["Keras / VGG16 Training"]
        Prep --> TrainTorch["PyTorch / ResNet Research Training"]
        TrainTF --> Eval["Evaluation & Metrics"]
        TrainTorch --> Eval
        Eval --> MLflowStore["MLflow Tracking / DagsHub Registry"]
        TrainTorch -.-> ONNXExport["TorchScript & ONNX Export"]
    end

    subgraph CICD ["2. Automated CI/CD (GitHub Actions)"]
        CodePush["Commit / Pull Request"] --> Linter["Ruff Linter & Formatter"]
        Linter --> Tests["Pytest (Unit, API, Models, Utils)"]
        Tests --> DockerBuild["Multi-Stage Docker Image Build"]
        DockerBuild --> GHCR["GitHub Container Registry (ghcr.io)"]
    end

    subgraph Production_Serving ["3. Serving & Observability (FastAPI + Prometheus)"]
        GHCR --> Container["Production Container (Appuser)"]
        Container --> API["FastAPI Serving Microservice"]
        API --> UI["Web UI (CT Scan Visualizer)"]
        API --> Endpoints["/predict | /api/v1/predict | /predict/upload"]
        API --> MemoryInfer["In-Memory Zero-Disk Inference"]
        API --> Prom["Prometheus Metrics (/metrics)"]
        API --> Drift["Image Drift Detection (KS Stat)"]
    end
```

---

## 🛠️ Complete Tech Stack

| Category | Tools & Libraries | Purpose & Implementation Details |
| :--- | :--- | :--- |
| **API & Serving** | **FastAPI**, **Uvicorn**, **Pydantic v2**, **Jinja2** | High-throughput asynchronous serving, OpenAPI/Swagger autodocs (`/docs`), strict payload validation, and server-side web UI rendering. |
| **Deep Learning (Production)** | **TensorFlow 2.x**, **Keras** | Transfer learning on VGG16 backbone for kidney CT scan tumor classification. |
| **Deep Learning (Research)** | **PyTorch 2.x**, **Torchvision** | Research architecture (`KidneyPyTorchClassifier`) with ResNet18/VGG16 backbones, custom heads, TorchScript and ONNX export capability. |
| **Model Optimization** | **TorchScript**, **ONNX Runtime** | Serialized model artifacts for low-latency, hardware-accelerated production inference. |
| **Image Processing** | **Pillow (PIL)**, **NumPy**, **SciPy** | Zero-disk in-memory image decoding from Base64/Multipart streams, bilinear normalization, and statistical feature extraction. |
| **Pipeline & Versioning** | **DVC (Data Version Control)** | Multi-stage DAG pipeline (`dvc.yaml`) versioning raw datasets, base models, training runs, and evaluation metrics. |
| **Experiment Tracking** | **MLflow**, **DagsHub** | Tracking parameters, loss curves, confusion matrices, and model artifact registry with remote and local failover. |
| **Packaging & Environment** | **`uv`**, **pyproject.toml (PEP 621)** | Sub-second dependency resolution, Hatchling build backend, and modular optional dependency groups. |
| **Containerization** | **Docker**, **Docker Compose** | Multi-stage container builds, non-privileged runtime user (`appuser` UID 10001), health probes, and local 3-tier service orchestration (API, MLflow, Prometheus). |
| **Testing & Quality** | **Pytest**, **Pytest-Cov**, **HTTPX** | Comprehensive test suite covering config loading, memory decoders, API routes, Keras and PyTorch models, and image drift detection. |
| **Linting & Formatting** | **Ruff** | Lightning-fast static analysis, PEP 8 compliance, auto-formatting, and import sorting. |
| **CI/CD Automation** | **GitHub Actions** | Automated CI workflow (`ci.yml`) for lint, tests, and Docker build smoke test; CD workflow (`cd.yml`) for GHCR container publishing. |
| **Monitoring & Observability** | **Prometheus**, **ImageDriftDetector** | `/metrics` endpoint with latency histograms, and two-sample Kolmogorov-Smirnov statistical tests for input image drift detection. |

---

## ⚡ Quickstart

### 1. Prerequisites & Environment Setup

Using **`uv`** (recommended):
```bash
git clone https://github.com/AbQaadir/MLOPS-Kidney-Disease-Classification.git
cd MLOPS-Kidney-Disease-Classification

# Create virtual environment and install all packages
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[all]"
```

Or using standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

---

### 2. Launch Local Application

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

- **Interactive Web UI**: [http://localhost:8080](http://localhost:8080)
- **Interactive Swagger Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **ReDoc Documentation**: [http://localhost:8080/redoc](http://localhost:8080/redoc)
- **Health Probe**: [http://localhost:8080/health](http://localhost:8080/health)
- **Prometheus Metrics**: [http://localhost:8080/metrics](http://localhost:8080/metrics)

---

### 3. Launch Full MLOps Stack via Docker Compose

Run the API service, local MLflow tracking server, and Prometheus with a single command:

```bash
docker compose up --build
```

| Service | URL | Description |
| :--- | :--- | :--- |
| **NephroScan API** | `http://localhost:8080` | Production model inference service & UI |
| **MLflow Server** | `http://localhost:5000` | Experiment runs, metric charts, and model registry |
| **Prometheus** | `http://localhost:9090` | Real-time API latency and throughput monitoring |

---

## 📡 REST API Reference

### 1. JSON Base64 Prediction (`POST /api/v1/predict`)
```bash
curl -X POST "http://localhost:8080/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"image": "<BASE64_STRING>"}'
```

**Response**:
```json
{
  "prediction": "Tumor",
  "confidence": 0.9852,
  "class_index": 1,
  "probabilities": {
    "Normal": 0.0148,
    "Tumor": 0.9852
  },
  "status": "success"
}
```

### 2. Multipart Image Upload (`POST /predict/upload`)
```bash
curl -X POST "http://localhost:8080/predict/upload" \
     -F "file=@inputImage.jpg"
```

### 3. Trigger Asynchronous Training (`POST /train`)
```bash
curl -X POST "http://localhost:8080/train"
```

---

## 🔬 PyTorch Research & Experimentation

For research and experimentation, NephroScan includes a modular PyTorch transfer learning module with TorchScript and ONNX export capabilities:

```python
from KDC.models.pytorch_model import KidneyPyTorchClassifier

# Initialize model with ResNet18 backbone
model = KidneyPyTorchClassifier(num_classes=2, backbone_name="resnet18", pretrained=True)

# Export to TorchScript for production deployment
model.export_torchscript("artifacts/training/model.pt")

# Export to ONNX for cross-platform inference acceleration
model.export_onnx("artifacts/training/model.onnx")
```

Run the complete PyTorch research pipeline:
```bash
python research/pytorch_research_pipeline.py
```

---

## 🔁 DVC Pipeline & Experiment Reproduction

The end-to-end data processing and model training workflow is orchestrated via DVC:

```bash
# Reproduce the entire DVC pipeline
dvc repro

# Visualize DVC pipeline DAG
dvc dag
```

Pipeline stages defined in `dvc.yaml`:
1. `data_ingestion`: Downloads and extracts dataset archives.
2. `prepare_base_model`: Initializes pre-trained weights and attaches dense classification head.
3. `training`: Fits the model on data generators with real-time augmentation.
4. `evaluation`: Computes test evaluation scores and logs parameters/metrics to MLflow.

---

## 🧪 Testing & Code Quality

```bash
# Code formatting check
ruff format --check .

# Static code analysis and linting
ruff check .

# Execute unit and integration tests with coverage
pytest -v --cov=src/KDC tests/
```

All 20 unit and integration tests validate:
- Image decoders & in-memory stream processing
- Pydantic schema validation
- FastAPI endpoints (`/health`, `/ready`, `/predict`, `/train`, UI)
- Keras & PyTorch forward passes and model exports
- Image drift detection algorithms

---

## 🚀 CI/CD Automation

- **Continuous Integration (`.github/workflows/ci.yml`)**:
  - Triggers on all pull requests and commits to `main`.
  - Runs Ruff linter and formatter.
  - Executes Pytest test suite with JUnit artifact uploads.
  - Executes a Docker build smoke test.
- **Continuous Deployment (`.github/workflows/cd.yml`)**:
  - Triggers on version tags (`v*.*.*`) or manual workflow dispatch.
  - Builds and pushes multi-platform Docker container images to **GitHub Container Registry (`ghcr.io/AbQaadir/mlops-kidney-disease-classification`)**.

---

## 👤 Author

**Abdelqaadir**  
- Email: [qaadireng@gmail.com](mailto:qaadireng@gmail.com)  
- GitHub: [@AbQaadir](https://github.com/AbQaadir)

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).