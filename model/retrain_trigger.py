"""
retrain_trigger.py — Auto-retraining orchestration.

Watches for three retraining signals:
  1. Model accuracy drops below the configured threshold (evaluated on latest data).
  2. Distribution drift detected in incoming data (polled from ingestion state).
  3. Schema change requiring feature pipeline re-engineering.

When triggered, it calls train.train(), sends a Slack notification with the
reason and new accuracy, and optionally kicks off AWS redeployment (Part 3 stub).

Design decisions:
  • Trigger reasons are accumulated per evaluation cycle so one retrain handles
    multiple simultaneous signals rather than firing three back-to-back retrains.
  • The accuracy check loads the latest saved model and re-evaluates it on the
    most recent data — this catches real degradation, not just training noise.
  • Redeployment is a stub here; Part 3 will replace it with a real AWS call.
  • A minimum interval between retrains (RETRAIN_COOLDOWN_SEC) prevents thrashing
    when drift is persistent.
"""

import os
import glob
import time
import logging
import joblib
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from prometheus_client import Counter, Gauge, start_http_server

import train as trainer  # reuse training logic from train.py

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
DATA_FILE             = os.getenv("DATA_FILE",         "data/records.csv")
MODEL_DIR             = os.getenv("MODEL_DIR",         "model/artifacts")
ACCURACY_THRESHOLD    = float(os.getenv("ACCURACY_THRESHOLD",  "0.80"))
EVAL_INTERVAL_SEC     = int(os.getenv("EVAL_INTERVAL",         "60"))   # how often to evaluate
RETRAIN_COOLDOWN_SEC  = int(os.getenv("RETRAIN_COOLDOWN",      "120"))  # min gap between retrains
VAL_SPLIT             = float(os.getenv("VAL_SPLIT",           "0.2"))
LABEL_COL             = os.getenv("LABEL_COL",         "label")
SLACK_WEBHOOK_URL     = os.getenv("SLACK_WEBHOOK_URL", "")
RETRAIN_METRICS_PORT  = int(os.getenv("RETRAIN_METRICS_PORT",  "8003"))

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
RETRAIN_TRIGGERED  = Counter("retrain_triggered_total",  "Times retraining was triggered")
CURRENT_ACCURACY   = Gauge("current_model_accuracy",     "Live accuracy of deployed model")

# ---------------------------------------------------------------------------
# Slack helper
# ---------------------------------------------------------------------------
def _send_slack(message: str) -> None:
    if not SLACK_WEBHOOK_URL:
        log.warning("SLACK_WEBHOOK_URL not set — skipping alert: %s", message)
        return
    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            json={"text": f":robot_face: *MLOps Retrain Alert*\n{message}"},
            timeout=5,
        )
        resp.raise_for_status()
    except Exception as exc:
        log.error("Slack alert failed: %s", exc)


# ---------------------------------------------------------------------------
# Load the latest saved model
# ---------------------------------------------------------------------------
def _load_latest_model():
    """Return the most recently versioned model, or None if none exist."""
    files = glob.glob(os.path.join(MODEL_DIR, "model_v*.pkl"))
    if not files:
        return None, None

    # Sort by version number extracted from filename
    def _version(path):
        import re
        m = re.search(r"model_v(\d+)\.pkl", os.path.basename(path))
        return int(m.group(1)) if m else 0

    latest = max(files, key=_version)
    version = _version(latest)
    model = joblib.load(latest)
    log.info("Loaded model v%d from %s", version, latest)
    return model, version


# ---------------------------------------------------------------------------
# Evaluate current model accuracy on latest data
# ---------------------------------------------------------------------------
def _evaluate_current_model() -> float | None:
    """
    Load latest model and latest data, evaluate accuracy on a held-out split.
    Returns accuracy float, or None if evaluation is not possible.
    """
    model, version = _load_latest_model()
    if model is None:
        log.warning("No trained model found — cannot evaluate accuracy.")
        return None

    if not os.path.isfile(DATA_FILE):
        log.warning("Data file not found — cannot evaluate accuracy.")
        return None

    df = pd.read_csv(DATA_FILE).dropna()
    if len(df) < 20:
        log.warning("Not enough data (%d rows) to evaluate.", len(df))
        return None

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feature_cols].values.astype(float)
    y = df[LABEL_COL].values

    # Use a fixed random_state so the val split is consistent across evaluations
    _, X_val, _, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=42)
    preds    = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)

    log.info("Current model v%d accuracy on latest data: %.4f", version, accuracy)
    CURRENT_ACCURACY.set(accuracy)
    return accuracy


# ---------------------------------------------------------------------------
# AWS redeployment stub — replaced in Part 3
# ---------------------------------------------------------------------------
def _redeploy_to_aws(model_path: str, version: int) -> None:
    """
    Stub for AWS redeployment.
    Part 3 will replace this with a real SageMaker / EC2 deployment call.
    """
    log.info("[STUB] Redeploying model v%d (%s) to AWS …", version, model_path)
    # TODO Part 3: trigger GitHub Actions workflow or call AWS SDK here


# ---------------------------------------------------------------------------
# Core retrain decision function
# ---------------------------------------------------------------------------
def maybe_retrain(
    drift_detected: bool = False,
    schema_changed: bool = False,
) -> None:
    """
    Evaluate all retraining signals and retrain if any are triggered.

    Parameters
    ----------
    drift_detected : bool — set True when ingestion signals distribution drift
    schema_changed : bool — set True when ingestion signals a schema change
    """
    reasons: list[str] = []

    # Signal 1 — accuracy degradation
    accuracy = _evaluate_current_model()
    if accuracy is not None and accuracy < ACCURACY_THRESHOLD:
        reasons.append(f"accuracy dropped to {accuracy:.4f} (threshold {ACCURACY_THRESHOLD})")

    # Signal 2 — distribution drift passed in from ingestion layer
    if drift_detected:
        reasons.append("distribution drift detected in incoming data")

    # Signal 3 — schema change passed in from ingestion layer
    if schema_changed:
        reasons.append("schema change detected (feature added or removed)")

    if not reasons:
        log.info("No retraining signals — model is healthy.")
        return

    # ------------------------------------------------------------------
    # At least one signal fired — retrain
    # ------------------------------------------------------------------
    reason_str = "; ".join(reasons)
    log.info("Retraining triggered: %s", reason_str)
    RETRAIN_TRIGGERED.inc()

    try:
        model_path, new_accuracy, version = trainer.train(start_metrics_server=False)

        # Notify Slack with reason + new accuracy
        _send_slack(
            f":arrows_counterclockwise: *Model retrained* (v{version})\n"
            f"*Reason:* {reason_str}\n"
            f"*New accuracy:* {new_accuracy:.4f}"
        )

        # Redeploy to AWS (stub until Part 3)
        _redeploy_to_aws(model_path, version)

        log.info(
            "Retraining complete — model v%d, accuracy=%.4f", version, new_accuracy
        )

    except Exception as exc:
        log.error("Retraining failed: %s", exc)
        _send_slack(f":x: Retraining *failed*: `{exc}`")


# ---------------------------------------------------------------------------
# Standalone watch loop — runs as a background service
# ---------------------------------------------------------------------------
def run_watch_loop() -> None:
    """
    Periodically evaluates the model and triggers retraining as needed.
    Drift/schema signals are injected by ingestion.py in a real deployment
    (e.g. via a shared Redis flag or a lightweight HTTP call).
    In standalone mode, only accuracy degradation is checked.
    """
    try:
        start_http_server(RETRAIN_METRICS_PORT)
        log.info("Retrain metrics server on port %d", RETRAIN_METRICS_PORT)
    except OSError:
        log.warning("Port %d already in use.", RETRAIN_METRICS_PORT)

    log.info("Retrain watch loop started (eval every %ds)", EVAL_INTERVAL_SEC)
    last_retrain_time = 0.0

    while True:
        now = time.time()

        # Enforce cooldown so we don't thrash on persistent drift/accuracy issues
        if now - last_retrain_time < RETRAIN_COOLDOWN_SEC:
            log.debug("Within cooldown window — skipping evaluation.")
            time.sleep(EVAL_INTERVAL_SEC)
            continue

        maybe_retrain()  # drift/schema signals False in standalone mode
        last_retrain_time = time.time()
        time.sleep(EVAL_INTERVAL_SEC)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_watch_loop()