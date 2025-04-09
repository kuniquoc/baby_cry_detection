# Baby Cry Detection

This project aims to detect baby cries from audio recordings using machine learning techniques. It includes preprocessing, training, inference, and evaluation pipelines.

## Features

- **Audio Preprocessing**: Segmentation and feature extraction (MFCC).
- **Model Training**: Deep learning models for cry detection.
- **Inference**: Batch and single-file prediction support.
- **Evaluation**: Metrics like accuracy, precision, recall, and F1 score.
- **Web API**: FastAPI-based API for deployment.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kuniquoc/baby_cry_detection
   cd baby_cry_detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Preprocessing

Segment raw audio files:
```bash
python src/data_processing/split_audio.py
```

### Prepare dataset
```bash
python src/data_processing/prepare_dataset.py
```

### Extract features
```bash
python src/utils/dataset_loader.py
```

### Training

Train the model using the prepared dataset:
```bash
python src/train.py 
```

### Inference

#### Single File
Run inference on a single audio file:
```bash
python src/inference.py --model runs/latest/checkpoints/best_model.pth --audio path/to/audio_file.wav
```

#### Batch Processing
Run inference on a directory of audio files:
```bash
python src/inference.py --model runs/latest/checkpoints/best_model.pth --dir path/to/audio_folder
```

#### Structured Directory
Run inference on a structured directory with `cry` and `not_cry` subdirectories:
```bash
python src/inference.py --model runs/latest/checkpoints/best_model.pth --dir path/to/test_directory --structured
```

## Project Structure

```
baby_cry_detection/        
├── data/                  # Raw and processed data
├── results/               # Predictions and evaluation results
├── runs/                  # Model checkpoints and logs
├── src/                   # Source code
│   ├── data_processing/   # Preprocessing scripts
│   ├── models/            # Model definitions
│   ├── utils/             # Utility functions
│   ├── train.py           # Train script
│   └── inference.py       # Inference script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Dependencies

Key dependencies include:
- **Core Libraries**: `numpy`, `pandas`, `matplotlib`
- **Audio Processing**: `librosa`, `torchaudio`
- **Deep Learning**: `torch`, `torchvision`
- **Web API**: `fastapi`, `uvicorn`

See the full list in `requirements.txt`.
