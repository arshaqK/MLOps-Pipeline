FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY server/app.py ./server/app.py

COPY model/artifacts/ ./model/artifacts/

ENV MODEL_DIR=model/artifacts
ENV LABEL_COL=label

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]