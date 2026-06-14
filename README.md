# End-to-End MLOps Pipeline

**Course:** Machine Learning Operations (MLOps) — Spring 2026
**Institution:** FAST NUCES

---

## Team Members

| Name | Roll Number |
|------|-------------|
| Syed Arshaq Hussain Kirmani | 22i-0834 |

---

## Project Description

A production-grade MLOps pipeline that:

- Ingests live data from an HTTP API, detecting schema changes and distribution drift
- Trains and auto-retrains a RandomForest classifier when data quality degrades
- Deploys the model as a REST API on AWS EC2 inside a Docker container
- Exposes Prometheus-compatible metrics for real-time observability
- Visualizes metrics in Grafana dashboards
- Routes 7 alert rules through Alertmanager to Slack
- Automates build, test, and deployment via GitHub Actions CI/CD

---

## Project Structure

```
mlops-project/
├── .github/workflows/mlops-ci.yml   # GitHub Actions CI/CD
├── ingestion/
│   ├── ingestion.py                 # Data fetching & schema monitoring
│   └── drift_detector.py           # Distribution drift detection
├── model/
│   ├── train.py                     # Training script
│   ├── retrain_trigger.py           # Auto-retraining orchestration
│   └── artifacts/                   # Versioned .pkl model files
├── serve/
│   └── app.py                       # FastAPI inference API
├── exporter/
│   └── metrics.py                   # Prometheus metric definitions
├── prometheus/
│   ├── prometheus.yml               # Prometheus config
│   └── alert_rules.yml             # Alerting rules
├── alertmanager/
│   └── alertmanager.yml            # Alertmanager Slack routing
├── grafana/
│   ├── dashboards/mlops_dashboard.json
│   └── provisioning/
│       ├── datasources/prometheus.yml
│       └── dashboards/dashboards.yml
├── deploy/
│   └── deploy.sh                    # AWS deployment script
├── tests/
│   ├── test_schema.py              # Unit test: schema change detection
│   ├── test_drift.py               # Unit test: drift detection
│   └── test_predict.py             # Unit test: /predict endpoint
├── docker-compose.yml              # Observability stack
├── Dockerfile                      # ML inference service container
├── requirements.txt
├── .env.example
└── README.md
```

---

## AWS EC2

**Public IP:** `13.57.126.183`
**Port:** `8000`

---

## Setup and Run Instructions

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- AWS account with EC2 instance running
- Docker Hub account
- Slack webhook URL

---

### Local Setup

**1. Clone the repository:**
```bash
git clone https://github.com/arshaqK/MLOps-Pipeline.git
cd MLOps-Pipeline
```

**2. Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your values
```

**5. Train initial model:**
```bash
python model/train.py
```

**6. Run ingestion and retrain watcher (separate terminals):**
```bash
# Terminal 1
python ingestion/ingestion.py

# Terminal 2
python model/retrain_trigger.py
```

**7. Run inference server locally:**
```bash
uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload
```

**8. Start observability stack:**
```bash
docker-compose --env-file .env up -d
```

---

### AWS EC2 Deployment

**1. Configure `.env` with your credentials:**
```env
DOCKERHUB_USERNAME=your_username
DOCKERHUB_TOKEN=your_token
EC2_HOST=13.57.126.183
EC2_USER=ubuntu
EC2_SSH_KEY=./ec2_key.pem
APP_PORT=8000
```

**2. Run deployment script:**
```bash
bash deploy/deploy.sh
```

This will:
- Build the Docker image locally
- Push it to Docker Hub
- SSH into EC2 and pull + run the container

---

## Testing the API

### Health Check
```bash
curl http://13.57.126.183:8000/health
```
Expected response:
```json
{"status": "ok"}
```

### Prediction
```bash
curl -X POST http://13.57.126.183:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.5, 2.0]}'
```
Expected response:
```json
{"prediction": 1, "confidence": 0.95, "model_version": 3}
```

### Prometheus Metrics
```bash
curl http://13.57.126.183:8000/metrics
```

Or open in browser:
```
http://13.57.126.183:8000/metrics
```

### Interactive API Docs
```
http://13.57.126.183:8000/docs
```

---

## Observability Stack

Start locally with:
```bash
docker-compose --env-file .env up -d
```

| Service | URL |
|---------|-----|
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |
| Alertmanager | http://localhost:9093 |

---

## Configuring Slack Webhook

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Create a new app → **From scratch**
3. Enable **Incoming Webhooks**
4. Add webhook to your channel (e.g. `#mlops-alerts`)
5. Copy the webhook URL
6. Add to `.env`:
```env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
```
7. Add to `alertmanager/slack_webhook.txt`:
```bash
echo -n "https://hooks.slack.com/services/XXX/YYY/ZZZ" > alertmanager/slack_webhook.txt
```

---

## Alert Rules

All 7 alerts are defined in `prometheus/alert_rules.yml` and routed to Slack via Alertmanager.

| # | Alert | Trigger | Slack Message | How to Trigger |
|---|-------|---------|---------------|----------------|
| 1 | `DataLakeUnavailable` | `increase(datalake_unavailable_total[1m]) > 0` | "Data source returned 503. Check API availability." | Wait for natural 503 from API or temporarily change `expr` to use `records_processed_total > 0` |
| 2 | `FeatureAdded` | `increase(feature_added_total[1m]) > 0` | "New feature detected in schema. Retraining may be required." | API schema change or temporarily change `expr` to use `records_processed_total > 0` |
| 3 | `FeatureRemoved` | `increase(feature_removed_total[1m]) > 0` | "Feature dropped from schema. Verify pipeline compatibility." | API schema change or temporarily change `expr` to use `records_processed_total > 0` |
| 4 | `DistributionDrift` | `distribution_drift_detected == 1` | "Data distribution drift detected. Model may be stale." | Happens naturally with `DRIFT_THRESHOLD=0.5` |
| 5 | `FeatureDriftDetected` | `distribution_drift_detected > 0` | "Feature-level drift flagged. Investigate upstream data." | Happens naturally with `DRIFT_THRESHOLD=0.5` |
| 6 | `HighResponseLatency` | `histogram_quantile(0.95, rate(response_delay_seconds_bucket[5m])) > 1.0` | "P95 response latency exceeded 1 second." | Temporarily lower threshold to `> 0.001` in `alert_rules.yml` |
| 7 | `LowModelAccuracy` | `model_accuracy < 0.8` | "Model accuracy dropped below threshold. Auto-retraining triggered." | Temporarily raise threshold to `< 0.99` in `alert_rules.yml` |

After modifying `alert_rules.yml`, restart Prometheus:
```bash
docker-compose restart prometheus
```

---

## Running Unit Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

Expected output:
```
tests/test_schema.py::test_feature_added PASSED
tests/test_schema.py::test_feature_removed PASSED
tests/test_schema.py::test_feature_added_and_removed PASSED
tests/test_schema.py::test_no_schema_change PASSED
tests/test_schema.py::test_first_batch_no_baseline PASSED
tests/test_drift.py::test_drift_detected_on_large_shift PASSED
tests/test_drift.py::test_no_drift_on_similar_distribution PASSED
tests/test_drift.py::test_warmup_batches_no_drift PASSED
tests/test_drift.py::test_drift_reset PASSED
tests/test_predict.py::test_predict_returns_correct_structure PASSED
tests/test_predict.py::test_predict_returns_valid_types PASSED
tests/test_predict.py::test_predict_correct_values PASSED
tests/test_predict.py::test_health_endpoint PASSED
tests/test_predict.py::test_predict_missing_features PASSED
```

---

## GitHub Actions CI/CD

The workflow runs automatically on every push to `main`:

1. **Lint and Test** — flake8 linting + pytest unit tests
2. **Build and Push** — builds Docker image, pushes to Docker Hub with `:latest` and `:<git-sha>` tags
3. **Deploy to EC2** — SSHs into EC2, pulls latest image, restarts container, verifies `/health` returns 200

### Required GitHub Secrets

Go to **Settings → Secrets and Variables → Actions** and add:

| Secret | Value |
|--------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Your Docker Hub access token |
| `EC2_SSH_KEY` | Full contents of your `.pem` file |
| `EC2_HOST` | `13.57.126.183` |
| `EC2_USER` | `ubuntu` |
| `SLACK_WEBHOOK_URL` | Your Slack webhook URL |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```env
# Data ingestion
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
POLL_INTERVAL=30
DRIFT_THRESHOLD=0.5
RETRAIN_ROW_COUNT=500
DATA_FILE=data/records.csv
INGEST_ENDPOINT=http://149.40.228.124:6500/records

# Model training
MODEL_DIR=model/artifacts
ACCURACY_THRESHOLD=0.80
MAX_ITERATIONS=10
LABEL_COL=label

# Retraining
EVAL_INTERVAL=60
RETRAIN_COOLDOWN=120

# Deployment
DOCKERHUB_USERNAME=your_username
DOCKERHUB_TOKEN=your_token
EC2_HOST=13.57.126.183
EC2_USER=ubuntu
EC2_SSH_KEY=./ec2_key.pem
APP_PORT=8000
```

> **Never commit `.env` or `*.pem` files to Git.**

## Demo Video
https://drive.google.com/file/d/18-ydrFnsWe97Mg9bxA7-UIqf-7QPg41P/view?usp=sharing
