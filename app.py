"""FastAPI Production Application for Kidney Disease Classification."""

import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure src directory is on sys.path
src_dir = str(Path(__file__).resolve().parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from KDC import logger
from KDC.pipeline.prediction import ModelManager, PredictionPipeline
from KDC.schema.api_schema import (
    HealthResponse,
    PredictBase64Request,
    PredictionResponse,
    TrainTriggerResponse,
)
from KDC.utils.common import decode_image_bytes

# Optional Prometheus metrics instrumentator
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to manage model warmup and resource cleanup."""
    logger.info("Initializing FastAPI Kidney Disease Classification service...")
    try:
        ModelManager.get_model()
        logger.info("Model pre-loaded successfully on startup.")
    except Exception as e:
        logger.warning(
            f"Model not immediately preloaded on startup (will load lazily on first inference): {e}"
        )
    yield
    logger.info("Shutting down FastAPI service...")


app = FastAPI(
    title="NephroScan MLOps API",
    description="Production Deep Learning & MLOps Platform for Kidney CT Scan Tumor Classification.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
templates_dir = Path("templates")
static_dir = Path("static")

if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    templates = None

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Initialize Prometheus Instrumentator if installed
if PROMETHEUS_AVAILABLE:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# Singleton prediction pipeline
prediction_pipeline = PredictionPipeline()


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def home(request: Request):
    """Render the web UI for CT scan image prediction."""
    if templates and (templates_dir / "index.html").exists():
        return templates.TemplateResponse(request=request, name="index.html")
    return HTMLResponse(
        "<h2>Kidney Disease Classification API is running. UI templates not found.</h2>"
    )


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Liveness & readiness probe."""
    is_loaded = ModelManager.is_loaded()
    loaded_path = ModelManager.get_loaded_path()
    return HealthResponse(
        status="healthy" if is_loaded else "ready",
        model_loaded=is_loaded,
        version="1.0.0",
        model_path=loaded_path,
    )


@app.get("/ready", tags=["Monitoring"])
async def readiness_probe():
    """Kubernetes / container readiness probe."""
    if not ModelManager.is_loaded():
        try:
            ModelManager.get_model()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not ready: {str(e)}",
            )
    return {"status": "ready"}


@app.post(
    "/predict",
    summary="Predict Kidney CT Scan class from Base64 image",
    tags=["Inference"],
)
async def predict_base64(payload: PredictBase64Request):
    """Runs inference on a base64 encoded image string.

    Returns a response format compatible with both the web UI and modern REST clients.
    """
    try:
        image_bytes = decode_image_bytes(payload.image)
        result = prediction_pipeline.predict_image(image_bytes)

        # Dual-compatible structure:
        # [ { "prediction": "Tumor", "confidence": 0.99, "image": "Tumor", ... } ]
        # Matches frontend expectance: res[0].image or res[0].prediction
        ui_compatible_response = [
            {
                "image": result["prediction"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "class_index": result["class_index"],
                "probabilities": result["probabilities"],
                "status": "success",
            }
        ]
        return JSONResponse(content=ui_compatible_response)
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference failed: {str(e)}",
        )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    summary="Standard REST prediction from Base64",
    tags=["Inference"],
)
async def predict_api(payload: PredictBase64Request):
    """Standard REST API endpoint returning structured Pydantic PredictionResponse."""
    try:
        image_bytes = decode_image_bytes(payload.image)
        result = prediction_pipeline.predict_image(image_bytes)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"API inference error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference failed: {str(e)}",
        )


@app.post(
    "/predict/upload",
    response_model=PredictionResponse,
    summary="Predict directly from uploaded image file",
    tags=["Inference"],
)
async def predict_upload(file: UploadFile = File(...)):
    """Accepts multipart image uploads (JPEG/PNG) and returns structured predictions."""
    try:
        contents = await file.read()
        result = prediction_pipeline.predict_image(contents)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"File upload inference error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference failed on uploaded file: {str(e)}",
        )


def _run_training_pipeline_task():
    """Background worker task to execute the model training pipeline."""
    try:
        logger.info("Executing training pipeline via main.py in background...")
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Training pipeline finished successfully:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training pipeline failed with return code {e.returncode}:\n{e.stderr}")
    except Exception as e:
        logger.exception(f"Unexpected error in background training: {e}")


@app.post("/train", response_model=TrainTriggerResponse, tags=["Training"])
async def trigger_training(background_tasks: BackgroundTasks):
    """Asynchronously triggers the DVC/MLflow training pipeline without blocking the API."""
    background_tasks.add_task(_run_training_pipeline_task)
    return TrainTriggerResponse(
        status="accepted",
        message="Model training pipeline triggered in background.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
