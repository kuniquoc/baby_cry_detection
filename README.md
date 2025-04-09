# Baby Cry Detection API

This API allows you to detect baby cries in audio recordings.

## Setup and Installation

1. Make sure you have the base project installed
2. Install the additional API dependencies:

```bash
pip install -r requirements_api.txt
```

## Running the API

Start the API server with:

```bash
python -m src.api.server --host 0.0.0.0 --port 8000
```

Options:
- `--host`: Host address (default: 0.0.0.0)
- `--port`: Port number (default: 8000)
- `--reload`: Enable auto-reload for development
- `--model`: Path to model checkpoint (default: runs/latest/checkpoints/best_model.pth)

## API Documentation

When the server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### GET /
Health check endpoint that returns API status.

### GET /model-status
Checks if the model is loaded and ready for predictions.

### POST /predict
Upload a single WAV file for cry detection.

Example using curl:
```bash
curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/audio.wav"
```

### POST /predict-batch
Upload multiple WAV files for batch prediction.

Example using curl:
```bash
curl -X POST "http://localhost:8000/predict-batch" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "files=@file1.wav" -F "files=@file2.wav"
```

## Response Format

```json
{
  "filename": "audio.wav",
  "prediction": "cry",
  "confidence": 0.95,
  "is_crying": true
}
```

- `filename`: Name of the uploaded file
- `prediction`: Prediction class ("cry" or "not_cry")
- `confidence`: Confidence score (0-1)
- `is_crying`: Boolean indicating if a baby cry was detected
