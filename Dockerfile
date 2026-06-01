# syntax=docker/dockerfile:1.2
FROM python:3.11-slim

WORKDIR /app


COPY requirements.txt requirements-test.txt requirements-dev.txt ./


RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-test.txt \
    && pip install --no-cache-dir -r requirements-dev.txt


COPY challenge/ ./challenge/
COPY data/ ./data/
COPY Makefile ./


EXPOSE 8080


CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]