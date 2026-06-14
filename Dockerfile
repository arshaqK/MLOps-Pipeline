FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all modules
COPY server/app.py ./server/app.py
COPY exporter/ ./exporter/
COPY ingestion/ ./ingestion/
COPY model/ ./model/

# Runtime environment defaults
ENV MODEL_DIR=model/artifacts
ENV LABEL_COL=label

EXPOSE 8000

# Only run app.py — ingestion and retrain watcher run as background threads inside it
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]