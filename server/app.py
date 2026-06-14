"""
app.py — ML inference service exposing /predict, /health, and /metrics.

Design decisions:
  • FastAPI chosen for automatic request validation via Pydantic and async support.
  • Model is loaded once at startup into memory — no per-request disk I/O.
  • /metrics endpoint delegates to prometheus_client so Prometheus can scrape it.
  • Model is reloaded automatically if a newer version file appears on disk,
    enabling zero-downtime model updates without restarting the container.
  • ingestion and retrain loops run as background threads so all metrics share
    the same Prometheus registry and are exposed at /metrics.
"""

import os
import re
import sys
import glob
import logging
import joblib
import threading
import numpy as np
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Ensure project root is on path so ingestion/model modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exporter.metrics import (
    PREDICT_REQUESTS, PREDICT_ERRORS, RESPONSE_DELAY_SECONDS, MODEL_VERSION,
    MODEL_ACCURACY, RETRAIN_COUNT, RECORDS_INGESTED, DATALAKE_UNAVAILABLE,
    FEATURE_ADDED, FEATURE_REMOVED, DISTRIBUTION_DRIFT_DETECTED,
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
MODEL_DIR = os.getenv("MODEL_DIR", "model/artifacts")
LABEL_COL = os.getenv("LABEL_COL", "label")

# ---------------------------------------------------------------------------
# Model state — module-level so it survives across requests
# ---------------------------------------------------------------------------
_model         = None
_model_version = 0


def _get_latest_model_path():
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
        MODEL_VERSION.set(version)
        log.info("Loaded model v%d from %s", version, path)


# ---------------------------------------------------------------------------
# Background thread runners
# ---------------------------------------------------------------------------

def _run_ingestion():
    """Run ingestion loop in a background thread."""
    try:
        from ingestion.ingestion import run_ingestion_loop
        log.info("Starting ingestion loop in background thread...")
        run_ingestion_loop()
    except Exception as exc:
        log.error("Ingestion thread crashed: %s", exc)


def _run_retrain_watcher():
    """Run retrain watcher in a background thread."""
    try:
        from model.retrain_trigger import run_watch_loop
        log.info("Starting retrain watcher in background thread...")
        run_watch_loop()
    except Exception as exc:
        log.error("Retrain watcher thread crashed: %s", exc)


# ---------------------------------------------------------------------------
# Lifespan — load model and start background threads on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the latest model
    _load_model()

    # Start ingestion as a daemon thread — dies when main process dies
    ingestion_thread = threading.Thread(target=_run_ingestion, daemon=True)
    ingestion_thread.start()

    # Start retrain watcher as a daemon thread
    retrain_thread = threading.Thread(target=_run_retrain_watcher, daemon=True)
    retrain_thread.start()

    log.info("Background threads started: ingestion + retrain watcher")
    yield


app = FastAPI(title="MLOps Inference Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    features: list[float]


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
    _load_model()

    if _model is None:
        PREDICT_ERRORS.inc()
        raise HTTPException(status_code=503, detail="No model available. Train a model first.")

    PREDICT_REQUESTS.inc()

    try:
        with RESPONSE_DELAY_SECONDS.time():
            X = np.array(request.features).reshape(1, -1)
            prediction = int(_model.predict(X)[0])
            proba      = _model.predict_proba(X)[0]
            confidence = float(np.max(proba))

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
    """Prometheus-compatible metrics endpoint — all metrics from all threads."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)