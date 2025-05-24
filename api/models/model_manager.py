import torch
import logging
import librosa
from typing import Tuple

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cnn_model import MobileNetV2_Crying
from src.dataset_loader import DatasetLoader
from src.preprocess import extract_mfcc

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_path: str = "D:/Git/baby_cry_detection/api/runs/20250515_003039/checkpoints/best_model_acc.pth"):
        self.model = None
        self.device = None
        self.label_encoder = None
        self.model_path = model_path
        self.loader = DatasetLoader()
    
    def load_model(self) -> Tuple[torch.nn.Module, torch.device]:
        """Load the trained model from checkpoint"""
        if self.model is not None:
            return self.model, self.device
        
        try:
            self.label_encoder = self.loader.label_encoder
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Initialize model
            self.model = MobileNetV2_Crying().to(self.device)
            
            # Load model weights
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            
            # Remove 'module.' prefix if it exists (handle DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # Remove first 7 chars ('module.')
                else:
                    new_state_dict[k] = v
                    
            self.model.load_state_dict(new_state_dict)
            
            # Print model information
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
            logger.info(f"Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
            
            self.model.eval()
            return self.model, self.device
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict(self, audio_data: torch.Tensor, sr: int) -> Tuple[str, float]:
        """Make prediction on audio data"""
        if self.model is None:
            self.load_model()
        
        target_sr = self.loader.sample_rate
        n_mels = self.loader.n_mels
        n_fft = self.loader.n_fft
        hop_length = self.loader.hop_length
        
        # Resample if necessary
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        # Extract MFCC features
        mfccs = extract_mfcc(
            audio_data,
            sr=sr,
            n_mfcc=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )

        # Convert to tensor
        features = torch.from_numpy(mfccs).float().unsqueeze(0)
        
        # Add channel dimension if needed
        if len(features.shape) == 3:  # If [batch, n_mels, time]
            features = features.unsqueeze(1)  # Make it [batch, channel, n_mels, time]
        
        features = features.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(features) 
            probability = torch.sigmoid(logits)
            
            predicted_key = (probability > 0.5).int().item()
            confidence = probability.item() if predicted_key == 1 else 1 - probability.item()
        
        predicted_class = self.label_encoder.inverse_transform([predicted_key])[0]
        
        return predicted_class, confidence