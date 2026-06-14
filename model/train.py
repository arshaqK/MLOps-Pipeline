"""
train.py — Binary classifier training with accuracy threshold and versioning.

Loads ingested data from the CSV produced by ingestion.py, trains a classifier,
and serializes the best model with a versioned filename.

Design decisions:
  • RandomForest chosen for robustness on small/noisy tabular data with no
    hyperparameter tuning required out of the box.
  • Training loops up to MAX_ITERATIONS times, each time re-fitting on a fresh
    random train/val split; keeps whichever iteration hits the accuracy target
    first, or the best seen if the target is never reached.
  • Model version is derived by scanning existing model files so versions never
    collide across runs.
  • Prometheus Gauge (not Counter) for accuracy because accuracy can go up or
    down; a Counter can only increase.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import glob
import logging
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from prometheus_client import start_http_server
from exporter.metrics import MODEL_ACCURACY, MODEL_VERSION, RETRAIN_COUNT

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE          = os.getenv("DATA_FILE", "data/records.csv")
MODEL_DIR          = os.getenv("MODEL_DIR", "model/artifacts")
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.80"))
MAX_ITERATIONS     = int(os.getenv("MAX_ITERATIONS", "10"))
VAL_SPLIT          = float(os.getenv("VAL_SPLIT", "0.2"))
METRICS_PORT       = int(os.getenv("TRAIN_METRICS_PORT", "8002"))  # separate port from ingestion
LABEL_COL          = os.getenv("LABEL_COL", "label")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
# MODEL_ACCURACY, MODEL_VERSION, RETRAIN_COUNT imported from exporter.metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_next_version() -> int:
    """Scan MODEL_DIR for existing model files and return the next version int."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(MODEL_DIR, "model_v*.pkl"))
    if not existing:
        return 1
    # Extract version numbers from filenames like model_v3.pkl
    versions = []
    for path in existing:
        match = re.search(r"model_v(\d+)\.pkl", os.path.basename(path))
        if match:
            versions.append(int(match.group(1)))
    return max(versions) + 1 if versions else 1


def _load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the ingested CSV; return feature matrix X and labels y."""
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(
            f"Data file not found: {DATA_FILE}. Run ingestion.py first."
        )

    df = pd.read_csv(DATA_FILE)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in data.")

    # Drop rows with any NaN — can appear after schema changes mid-stream
    df = df.dropna()

    if len(df) < 20:
        raise ValueError(f"Not enough data to train ({len(df)} rows). Need at least 20.")

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feature_cols].values.astype(float)
    y = df[LABEL_COL].values
    log.info("Loaded %d rows, %d features from %s", len(df), len(feature_cols), DATA_FILE)
    return X, y


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(start_metrics_server: bool = False) -> tuple[str, float, int]:
    """
    Train a classifier, iterating until accuracy >= ACCURACY_THRESHOLD or
    MAX_ITERATIONS is exhausted.

    Returns
    -------
    model_path : str   — path to the saved .pkl file
    accuracy   : float — validation accuracy of the saved model
    version    : int   — version number assigned to this model
    """
    if start_metrics_server:
        # Only start if called as a standalone script; retrain_trigger manages this
        try:
            start_http_server(METRICS_PORT)
            log.info("Prometheus metrics server started on port %d", METRICS_PORT)
        except OSError:
            log.warning("Metrics port %d already in use — skipping.", METRICS_PORT)

    X, y = _load_data()
    RETRAIN_COUNT.inc()

    best_model    = None
    best_accuracy = 0.0

    for iteration in range(1, MAX_ITERATIONS + 1):
        # Fresh random split each iteration to reduce variance
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VAL_SPLIT, random_state=iteration  # different seed each iter
        )

        # RandomForest: robust, handles small datasets well, no feature scaling needed
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,       # let trees grow fully on small data
            random_state=iteration,
            n_jobs=-1,            # use all CPU cores
        )
        clf.fit(X_train, y_train)

        preds    = clf.predict(X_val)
        accuracy = accuracy_score(y_val, preds)

        log.info("Iteration %d/%d — val accuracy: %.4f", iteration, MAX_ITERATIONS, accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model    = clf

        # Stop early as soon as we hit the target — no need to iterate further
        if accuracy >= ACCURACY_THRESHOLD:
            log.info("Accuracy threshold %.2f reached at iteration %d.", ACCURACY_THRESHOLD, iteration)
            break
    else:
        # Loop exhausted without hitting threshold
        log.warning(
            "Could not reach accuracy threshold %.2f in %d iterations. "
            "Best accuracy: %.4f. Saving best model.",
            ACCURACY_THRESHOLD, MAX_ITERATIONS, best_accuracy,
        )

    # ------------------------------------------------------------------
    # Serialize the best model with a versioned filename
    # ------------------------------------------------------------------
    version    = _get_next_version()
    model_path = os.path.join(MODEL_DIR, f"model_v{version}.pkl")
    joblib.dump(best_model, model_path)
    log.info("Model v%d saved to %s (accuracy=%.4f)", version, model_path, best_accuracy)

    # ------------------------------------------------------------------
    # Update Prometheus gauges
    # ------------------------------------------------------------------
    MODEL_ACCURACY.set(best_accuracy)
    MODEL_VERSION.set(version)

    return model_path, best_accuracy, version


# ---------------------------------------------------------------------------
# Entry point — allows running train.py directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model_path, accuracy, version = train(start_metrics_server=True)
    print(f"\nDone. Model v{version} saved to: {model_path}")
    print(f"Validation accuracy: {accuracy:.4f}")