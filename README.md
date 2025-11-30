# Baby Cry Detection System

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/kuniquoc/baby_cry_detection/actions)

A comprehensive machine learning-based system for detecting baby cries in real-time using audio processing and deep learning techniques. This project includes data preprocessing, model training, a REST API, web interface, and Firebase integration for notifications.

## 🚀 Features

- **Real-time Audio Processing**: Process audio streams to detect baby cries instantly
- **Machine Learning Model**: Convolutional Neural Network (CNN) trained on audio spectrograms
- **REST API**: Flask-based API for easy integration
- **Web Interface**: User-friendly web dashboard for monitoring and testing
- **Firebase Integration**: Push notifications via Firebase Cloud Messaging (FCM)
- **WebSocket Support**: Real-time communication for live updates
- **Data Pipeline**: Complete workflow from raw audio to trained model
- **Model Management**: Version control and deployment of ML models

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Training the Model](#training-the-model)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- Firebase project (for notifications)

### Clone the Repository

```bash
git clone https://github.com/kuniquoc/baby_cry_detection.git
cd baby_cry_detection
```

### Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# On Windows:
env\Scripts\activate
# On macOS/Linux:
# source env/bin/activate
```

### Install Dependencies

```bash
# Install main requirements
pip install -r requirements.txt

# Install API-specific requirements
pip install -r api/requirements.txt
```

### Firebase Setup

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Firebase Cloud Messaging (FCM)
3. Download the service account key and save as `api/firebase-credentials.json`
4. Update FCM configuration in `api/fcm_service.py`

## 🚀 Usage

### Running the API Server

```bash
cd api
python app.py
```

The API will be available at `http://localhost:5000`

### Running the Web Interface

```bash
# From the root directory
python -m flask run --host=0.0.0.0 --port=8000
```

Access the web interface at `http://localhost:8000`

### Testing the Detection

```python
from src.inference import CryDetector

detector = CryDetector()
result = detector.detect('path/to/audio/file.wav')
print(f"Cry detected: {result}")
```

## 📁 Project Structure

```
baby_cry_detection/
├── api/                          # REST API
│   ├── app.py                   # Main Flask application
│   ├── fcm_service.py           # Firebase Cloud Messaging
│   ├── firebase_service.py      # Firebase integration
│   └── requirements.txt         # API dependencies
├── data/                        # Data management
│   ├── raw/                     # Raw audio files
│   ├── processed/               # Processed datasets
│   └── segmented/               # Audio segments
├── models/                      # ML models
│   ├── cnn_model.py            # Convolutional Neural Network
│   └── model_manager.py        # Model versioning
├── scripts/                     # Utility scripts
│   ├── segment_audio.py        # Audio segmentation
│   └── filter_false_predictions.py
├── src/                         # Source code
│   ├── inference.py            # Inference engine
│   ├── train.py                # Training script
│   ├── preprocess.py           # Data preprocessing
│   └── dataset_loader.py       # Data loading utilities
├── static/                      # Web assets
│   ├── css/
│   └── js/
├── templates/                   # HTML templates
├── utils/                       # Utility functions
├── websocket/                   # WebSocket handlers
├── ML_Notebook/                # Jupyter notebooks
│   └── train_model.ipynb       # Model training notebook
├── results/                     # Model predictions and results
├── runs/                        # Training runs and checkpoints
└── requirements.txt            # Main dependencies
```

## 🔌 API Endpoints

### Detection Endpoints

- `POST /detect` - Detect cry in uploaded audio file
- `POST /detect/stream` - Real-time audio stream detection
- `GET /health` - API health check

### Model Management

- `GET /models` - List available models
- `POST /models/deploy` - Deploy a new model version

### WebSocket

- `/ws/detection` - Real-time detection updates

## 🧠 Training the Model

### Using the Training Script

```bash
python src/train.py --config config/training_config.yaml
```

### Using the Jupyter Notebook

1. Start Jupyter:

   ```bash
   jupyter notebook ML_Notebook/train_model.ipynb
   ```

2. Follow the notebook steps for data preparation and training

### Data Preparation

```bash
# Segment audio files
python scripts/segment_audio.py

# Preprocess data
python src/preprocess.py
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
FLASK_ENV=development
FIREBASE_CREDENTIALS_PATH=api/firebase-credentials.json
MODEL_PATH=models/cry_detection_model.h5
AUDIO_SAMPLE_RATE=16000
```

### Model Configuration

Update `services/cry_detection_config.py` for detection parameters:

```python
DETECTION_THRESHOLD = 0.8
AUDIO_CHUNK_SIZE = 1.0  # seconds
OVERLAP = 0.5
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Run linting
flake8 src/ api/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Audio processing techniques inspired by various ML research papers
- Dataset sources and preprocessing methods
- Open-source community for libraries and tools

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project is for educational and research purposes. Always consult medical professionals for baby care decisions.
