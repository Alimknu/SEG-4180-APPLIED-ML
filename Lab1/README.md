# Lab1: Sentiment Analysis Model Service

This is a containerized sentiment analysis API using DistilBERT pretrained model served with Flask.

## Quick Start

### Run Locally (Docker)

```bash
docker run -p 5000:5000 makinu/lab1-model-service:latest
```

The API will be available at `http://localhost:5000`

### Test the API

Health check:
```bash
curl http://localhost:5000/health
```

Single prediction:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"I love this!\"}"
```

Batch prediction:
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d "{\"texts\":[\"I love this!\", \"This is terrible\"]}"
```

## Build Locally

```bash
docker build -t makinu/lab1-model-service:latest .
```

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single text prediction
- `POST /predict_batch` - Batch text predictions

## Model

Uses pretrained `distilbert-base-uncased-finetuned-sst-2-english` from Hugging Face for sentiment classification.

## Docker Hub

Image available at: https://hub.docker.com/r/makinu/lab1-model-service
