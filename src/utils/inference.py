import torch
import librosa
import numpy as np
from pathlib import Path
from models.cnn_model import CryingCNN
from data_processing.preprocess import extract_mfcc

def load_model(model_path):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CryingCNN().to(device)  # Remove num_classes parameter
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def predict(audio_path, model, device, sr=16000):
    """Make prediction on audio file"""
    # Load and preprocess audio
    y, _ = librosa.load(audio_path, sr=sr)
    
    # Extract MFCC features
    mfccs = extract_mfcc(y, sr=sr)
    
    # Convert to tensor and add batch and channel dimensions
    features = torch.from_numpy(mfccs).float().unsqueeze(0).unsqueeze(0)
    features = features.to(device)
    
    # Make prediction
    with torch.no_grad():
        probability = model(features)
        predicted_class = (probability > 0.5).int().item()
        confidence = probability.item() if predicted_class == 1 else 1 - probability.item()
    
    return predicted_class, confidence

def main():
    # Configuration
    model_path = Path('runs/latest/checkpoints/best_model.pth')
    test_audio_path = Path('data/test/test_audio.wav')
    
    # Load model
    model, device = load_model(model_path)
    
    # Make prediction
    predicted_class, confidence = predict(test_audio_path, model, device)
    
    # Print results
    class_names = ['not_cry', 'cry']
    print(f'Predicted class: {class_names[predicted_class]}')
    print(f'Confidence: {confidence:.2%}')

if __name__ == '__main__':
    main()