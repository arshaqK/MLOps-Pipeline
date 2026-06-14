"""
app.py — ML inference service exposing /predict, /health, and /metrics.

Design decisions:
  • FastAPI chosen for automatic request validation via Pydantic and async support.
  • Model is loaded once at startup into memory — no per-request disk I/O.
  • /metrics endpoint delegates to prometheus_client so Prometheus can scrape it.
  • Model is reloaded automatically if a newer version file appears on disk,
    enabling zero-downtime model updates without restarting the container.
"""

import os
import re
import glob
import logging
import joblib
import numpy as np
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from exporter.metrics import (
    PREDICT_REQUESTS, PREDICT_ERRORS, RESPONSE_DELAY_SECONDS, MODEL_VERSION as LOADED_MODEL_VER,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR  = os.getenv("MODEL_DIR", "model/artifacts")
LABEL_COL  = os.getenv("LABEL_COL", "label")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
# All metrics imported from exporter.metrics

# ---------------------------------------------------------------------------
# Model state — module-level so it survives across requests
# ---------------------------------------------------------------------------
_model         = None
_model_version = 0


def _get_latest_model_path() -> tuple[str, int] | tuple[None, None]:
    """Find the highest-versioned model .pkl in MODEL_DIR."""
    files = glob.glob(os.path.join(MODEL_DIR, "model_v*.pkl"))
    if not files:
        return None, None

    def _ver(p):
        m = re.search(r"model_v(\d+)\.pkl", os.path.basename(p))
        return int(m.group(1)) if m else 0

    latest = max(files, key=_ver)
    return latest, _ver(latest)


def _load_model() -> None:
    """Load (or reload) the latest model from disk into module-level state."""
    global _model, _model_version
    path, version = _get_latest_model_path()
    if path is None:
        log.warning("No model found in %s — /predict will return 503 until a model is trained.", MODEL_DIR)
        return
    if version != _model_version:
        _model = joblib.load(path)
        _model_version = version
        LOADED_MODEL_VER.set(version)
        log.info("Loaded model v%d from %s", version, path)


# ---------------------------------------------------------------------------
# Lifespan — load model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="MLOps Inference Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    features: list[float]  # ordered list of feature values, e.g. [0.5, -1.2]


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe — always returns 200 if the server is up."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Accept a feature vector, return predicted class and confidence.
    Reloads model from disk if a newer version is available.
    """
    # Reload model if a newer version has been saved since last request
    _load_model()

    if _model is None:
        PREDICT_ERRORS.inc()
        raise HTTPException(status_code=503, detail="No model available. Train a model first.")

    PREDICT_REQUESTS.inc()

    try:
        with RESPONSE_DELAY_SECONDS.time():
            X = np.array(request.features).reshape(1, -1)
            prediction  = int(_model.predict(X)[0])
            # predict_proba gives confidence for each class; take the winning class prob
            proba       = _model.predict_proba(X)[0]
            confidence  = float(np.max(proba))

        log.info("Prediction: %d (confidence=%.4f, model_v%d)", prediction, confidence, _model_version)
        return PredictResponse(
            prediction=prediction,
            confidence=confidence,
            model_version=_model_version,
        )

    except Exception as exc:
        PREDICT_ERRORS.inc()
        log.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics")
def metrics():
    """Prometheus-compatible metrics endpoint."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)