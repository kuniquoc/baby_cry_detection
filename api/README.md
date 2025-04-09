# Baby Cry Detection API

A FastAPI application for detecting baby cry sounds in audio files using machine learning.

## Features

- API endpoint for predicting baby cry in 3-second audio clips
- Support for analyzing longer audio files by segmenting
- Interactive UI for uploading and analyzing audio files
- Visualization of audio waveform and prediction results
- Detailed API documentation with Swagger UI

## Requirements

- Python 3.8+
- FFmpeg
- libsndfile

## Installation

1. Clone this repository
2. Install required packages:

```bash
pip install -r api/requirements.txt
```

3. Make sure you have a trained model in the default location or specify a custom path

## Running the API

### Direct Method

```bash
cd /path/to/baby_cry_detection
uvicorn api.app:app --reload
```

### Using Docker

```bash
cd /path/to/baby_cry_detection
docker build -t baby-cry-detection -f api/Dockerfile .
docker run -p 8000:8000 baby-cry-detection
```

## API Documentation

API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### `/predict/`

Predict whether a 3-second audio clip contains baby crying.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: form field 'audio' with WAV file (approx. 3 seconds)

**Response:**
```json
{
  "predicted_class": "cry",
  "confidence": 0.9876
}
```

### `/analyze/`

Analyze a longer audio file by segmenting it and making predictions on each segment.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: form field 'audio' with WAV file (any length)

**Response:**
```json
{
  "filename": "example.wav",
  "segments": [
    {
      "segment_index": 0,
      "start_time": 0.0,
      "end_time": 3.0,
      "predicted_class": "cry",
      "confidence": 0.9876,
      "mean_f0": 450.25
    },
    ...
  ],
  "summary": {
    "total_segments": 10,
    "cry_segments": 7,
    "not_cry_segments": 3,
    "cry_percentage": 0.7,
    "audio_length": 30.5
  }
}
```

## UI Usage

1. Open http://localhost:8000 in your web browser
2. Upload a WAV audio file
3. Click "Analyze Audio"
4. View the analysis results, including:
   - Audio waveform with marked segments
   - Timeline visualization
   - Pie chart showing cry vs. not cry segments
   - Detailed segment information table
