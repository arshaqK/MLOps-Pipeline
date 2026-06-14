"""
exporter/metrics.py — Single source of truth for all Prometheus metrics.

All other modules (ingestion, training, serving) import from here instead of
defining their own metrics. This prevents duplicate registration errors when
modules are imported together and makes the full metric inventory visible in
one place.

Usage:
    from exporter.metrics import RECORDS_INGESTED, MODEL_ACCURACY, ...
"""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Ingestion metrics
# ---------------------------------------------------------------------------

# Total records successfully ingested from the API
RECORDS_INGESTED = Counter(
    "records_processed_total",
    "Total number of records ingested from the API since startup."
)

# Number of times the /records endpoint returned 503
DATALAKE_UNAVAILABLE = Counter(
    "datalake_unavailable_total",
    "Number of times the /records endpoint returned 503."
)

# Number of features added to the schema since startup
FEATURE_ADDED = Counter(
    "feature_added_total",
    "Number of features added to the schema since startup."
)

# Number of features removed from the schema since startup
FEATURE_REMOVED = Counter(
    "feature_removed_total",
    "Number of features removed from the schema since startup."
)

# ---------------------------------------------------------------------------
# Drift metrics
# ---------------------------------------------------------------------------

# Set to 1 when drift is detected in the current batch, 0 otherwise
DISTRIBUTION_DRIFT_DETECTED = Gauge(
    "distribution_drift_detected",
    "Set to 1 when drift is detected in the current batch, 0 otherwise."
)

# ---------------------------------------------------------------------------
# Training / retraining metrics
# ---------------------------------------------------------------------------

# Total number of times the model has been retrained
RETRAIN_COUNT = Counter(
    "retrain_count_total",
    "Total number of times the model has been retrained."
)

# Current validation accuracy of the deployed model (0.0 - 1.0)
MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current validation accuracy of the deployed model (0.0 - 1.0)."
)

# Current model version number
MODEL_VERSION = Gauge(
    "model_version",
    "Current model version number."
)

# ---------------------------------------------------------------------------
# Inference / serving metrics
# ---------------------------------------------------------------------------

# Total /predict requests
PREDICT_REQUESTS = Counter(
    "predict_requests_total",
    "Total number of /predict API calls."
)

# Total /predict errors
PREDICT_ERRORS = Counter(
    "predict_errors_total",
    "Total number of /predict errors."
)

# Latency of each /predict call in seconds
RESPONSE_DELAY_SECONDS = Histogram(
    "response_delay_seconds",
    "Latency of each /predict API call in seconds.",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)