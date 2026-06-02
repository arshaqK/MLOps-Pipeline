"""
ingestion.py — Data fetching, schema monitoring, and pipeline orchestration.

Polls the /records endpoint at regular intervals, persists batches locally,
detects schema changes, delegates drift detection, and triggers retraining.
"""

import os
import csv
import time
import logging
import requests
import pandas as pd
from datetime import datetime
from prometheus_client import Counter, start_http_server

from drift_detector import DriftDetector

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging — structured, timestamped output so every event is traceable
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all tunables in one place; override via environment vars
# ---------------------------------------------------------------------------
ENDPOINT_URL      = os.getenv("INGEST_ENDPOINT", "http://149.40.228.124:6500/records")
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL", "30"))       # seconds between polls
DATA_FILE         = os.getenv("DATA_FILE", "data/records.csv")  # local persistence path
DRIFT_THRESHOLD   = float(os.getenv("DRIFT_THRESHOLD", "0.5"))  # z-score-like threshold
RETRAIN_ROW_COUNT = int(os.getenv("RETRAIN_ROW_COUNT", "500"))  # rows before auto-retrain
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")          # set in env for live alerts
METRICS_PORT      = int(os.getenv("METRICS_PORT", "8001"))      # Prometheus scrape port

# ---------------------------------------------------------------------------
# Prometheus counters — one per notable event so Grafana can graph each
# ---------------------------------------------------------------------------
FEATURE_ADDED          = Counter("feature_added_total",        "Schema: new feature appeared")
FEATURE_REMOVED        = Counter("feature_removed_total",      "Schema: existing feature disappeared")
DRIFT_DETECTED         = Counter("distribution_drift_detected_total", "Drift: stat shift beyond threshold")
DATALAKE_UNAVAILABLE   = Counter("datalake_unavailable_total", "HTTP 503 responses from data source")
RECORDS_INGESTED       = Counter("records_ingested_total",     "Total rows successfully ingested")
RETRAIN_TRIGGERED      = Counter("retrain_triggered_total",    "Times retraining pipeline was invoked")


# ---------------------------------------------------------------------------
# Slack helper — fire-and-forget; failures are logged but never crash the loop
# ---------------------------------------------------------------------------
def _send_slack(message: str) -> None:
    """Post a plain-text alert to the configured Slack webhook."""
    if not SLACK_WEBHOOK_URL:
        log.warning("SLACK_WEBHOOK_URL not set — skipping Slack alert: %s", message)
        return
    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            json={"text": f":robot_face: *MLOps Ingestion Alert*\n{message}"},
            timeout=5,
        )
        resp.raise_for_status()
    except Exception as exc:  # network issues must not kill the ingestion loop
        log.error("Slack alert failed: %s", exc)


# ---------------------------------------------------------------------------
# Schema comparison — returns (added_features, removed_features)
# ---------------------------------------------------------------------------
def _compare_schemas(
    previous: list[str] | None, current: list[str]
) -> tuple[list[str], list[str]]:
    """Diff two schema lists; return newly appeared and disappeared columns."""
    if previous is None:
        return [], []  # first batch has no baseline to compare against
    prev_set, curr_set = set(previous), set(current)
    return sorted(curr_set - prev_set), sorted(prev_set - curr_set)


# ---------------------------------------------------------------------------
# CSV persistence — append-only so nothing is ever overwritten
# ---------------------------------------------------------------------------
def _persist_batch(records: list[dict], schema: list[str]) -> int:
    """Append records to the local CSV, creating it with a header if needed."""
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    
    # Always check if header exists by reading first line, not just file existence
    has_header = False
    if os.path.isfile(DATA_FILE):
        with open(DATA_FILE, "r") as fh:
            first_line = fh.readline().strip()
            # Header exists if first line matches schema columns
            has_header = first_line == ",".join(schema)

    with open(DATA_FILE, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=schema, extrasaction="ignore")
        if not has_header:
            writer.writeheader()
        for row in records:
            writer.writerow({col: row.get(col, "") for col in schema})

    return len(records)


# ---------------------------------------------------------------------------
# Retraining trigger — decoupled so it can be swapped for an API call later
# ---------------------------------------------------------------------------
def _trigger_retraining(reason: str) -> None:
    """Invoke the retraining pipeline; currently shell-based, easily swappable."""
    log.info("Triggering retraining: %s", reason)
    RETRAIN_TRIGGERED.inc()
    _send_slack(f":arrows_counterclockwise: Retraining triggered — {reason}")

    # Replace the os.system call with an HTTP POST to a training service or
    # a Kubernetes Job submission when the pipeline is containerised.
    ret = os.system("python model/train.py")
    if ret != 0:
        log.error("Retraining script exited with code %d", ret)


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------
def run_ingestion_loop() -> None:
    """
    Continuously polls the data endpoint, persists records, monitors schema
    and drift, fires alerts, and triggers retraining when warranted.
    """
    start_http_server(METRICS_PORT)  # expose /metrics for Prometheus scraping
    log.info("Prometheus metrics server started on port %d", METRICS_PORT)

    previous_schema: list[str] | None = None  # no baseline on first iteration
    rows_since_last_retrain: int = 0
    drift_detector = DriftDetector(threshold=DRIFT_THRESHOLD)

    log.info("Starting ingestion loop (poll every %ds) …", POLL_INTERVAL_SEC)

    while True:
        try:
            # ------------------------------------------------------------------
            # 1. Fetch a batch from the upstream API
            # ------------------------------------------------------------------
            response = requests.get(ENDPOINT_URL, timeout=10)

            # ------------------------------------------------------------------
            # 2. Handle intentional 503 downtime gracefully
            # ------------------------------------------------------------------
            if response.status_code == 503:
                log.warning("Data source returned 503 — upstream unavailable")
                DATALAKE_UNAVAILABLE.inc()
                _send_slack(":warning: Data source returned *503 Service Unavailable*. Retrying …")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            response.raise_for_status()  # surface any other unexpected HTTP errors
            
            payload = response.json()
            # API returns a list of {"features": [...], "label": ...} objects
            # We flatten each record into {"feature_0": val, "feature_1": val, ..., "label": val}
            if isinstance(payload, list):
                if not payload:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                n_features = len(payload[0]["features"])
                feature_cols = [f"feature_{i}" for i in range(n_features)]
                current_schema = feature_cols + ["label"]

                records = []
                for row in payload:
                    flat = {f"feature_{i}": v for i, v in enumerate(row["features"])}
                    flat["label"] = row["label"]
                    records.append(flat)
            else:
                current_schema = payload["schema"]
                records = payload["records"]


            # ------------------------------------------------------------------
            # 3. Schema change detection
            # ------------------------------------------------------------------
            added, removed = _compare_schemas(previous_schema, current_schema)

            for feat in added:
                log.info("Schema change — new feature appeared: '%s'", feat)
                FEATURE_ADDED.inc()
                _send_slack(f":heavy_plus_sign: New feature appeared in schema: `{feat}`")

            for feat in removed:
                log.warning("Schema change — feature removed: '%s'", feat)
                FEATURE_REMOVED.inc()
                _send_slack(f":heavy_minus_sign: Feature *removed* from schema: `{feat}`")

            # Any schema change is a strong signal to retrain immediately
            if added or removed:
                _trigger_retraining(f"schema changed (added={added}, removed={removed})")
                rows_since_last_retrain = 0

            previous_schema = current_schema

            # ------------------------------------------------------------------
            # 4. Persist the batch to disk
            # ------------------------------------------------------------------
            if records:
                n = _persist_batch(records, current_schema)
                rows_since_last_retrain += n
                RECORDS_INGESTED.inc(n)
                log.info("Ingested %d records (session total: %d)", n, rows_since_last_retrain)

            # ------------------------------------------------------------------
            # 5. Distribution drift detection
            # ------------------------------------------------------------------
            if records:
                df_batch = pd.DataFrame(records, columns=current_schema)
                drifted_features = drift_detector.check(df_batch)

                if drifted_features:
                    log.warning("Drift detected in features: %s", drifted_features)
                    DRIFT_DETECTED.inc()
                    _send_slack(
                        f":chart_with_downwards_trend: Distribution drift detected in: "
                        f"`{'`, `'.join(drifted_features)}`"
                    )
                    # Drift in the live distribution also warrants a retrain
                    _trigger_retraining(f"drift in {drifted_features}")
                    rows_since_last_retrain = 0

            # ------------------------------------------------------------------
            # 6. Volume-based retraining — retrain once enough new data arrives
            # ------------------------------------------------------------------
            if rows_since_last_retrain >= RETRAIN_ROW_COUNT:
                _trigger_retraining(f"{rows_since_last_retrain} new rows accumulated")
                rows_since_last_retrain = 0

        except requests.exceptions.RequestException as exc:
            # Network-level errors (timeout, DNS, etc.) — log and continue
            log.error("Request failed: %s", exc)
            _send_slack(f":x: Ingestion request failed: `{exc}`")

        except Exception as exc:
            # Catch-all so the loop never silently dies
            log.exception("Unexpected error in ingestion loop: %s", exc)

        time.sleep(POLL_INTERVAL_SEC)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_ingestion_loop()