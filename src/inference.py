import os
import sys
import torch
import librosa
from pathlib import Path
import argparse
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import from project structure
from src.models.cnn_model import CryingCNN
from src.utils.dataset_loader import DatasetLoader
from src.data_processing.preprocess import extract_mfcc


# Define class names for predictions
labelEncoder = DatasetLoader().label_encoder;

def load_model(model_path):
    """
    Load trained model from checkpoint
    
    Args:
        model_path (str): Path to the model checkpoint
        
    Returns:
        tuple: (model, device) - Loaded model and device
    """
    # Initialize DatasetLoader to get parameters
    loader = DatasetLoader()
    n_mels = loader.n_mels
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model with same parameters as during training
    model = CryingCNN().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print model information
        print(f"Model loaded successfully from {model_path}")
        print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
        print(f"Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
        
        model.eval()
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def predict(audio_path, model, device):
    """
    Make prediction on a single audio file
    
    Args:
        audio_path (str): Path to audio file
        model (torch.nn.Module): Loaded model
        device (torch.device): Device to run inference on
        
    Returns:
        tuple: (predicted_class, confidence) - Predicted class (0=not_cry, 1=cry) and confidence score
    """
    # Initialize DatasetLoader to get same parameters as training
    loader = DatasetLoader()
    sr = loader.sample_rate
    n_mels = loader.n_mels
    n_fft = loader.n_fft
    hop_length = loader.hop_length
    
    try:
        # Load and preprocess audio
        y, _ = librosa.load(audio_path, sr=sr)
        
        # Extract MFCC features using the same approach as in DatasetLoader
        mfccs = extract_mfcc(
            y, 
            sr=sr,
            n_mfcc=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )

        # Apply the same tensor transformation as in DatasetLoader.extract_features
        features = torch.from_numpy(mfccs).float().unsqueeze(0)  # [1, n_mels, time]
        
        # DatasetLoader adds only one channel dimension, but the model may expect [batch, channel, n_mels, time]
        # Check if we need to add another dimension for the channel
        if len(features.shape) == 3:  # If [batch, n_mels, time]
            features = features.unsqueeze(1)  # Make it [batch, channel, n_mels, time]
        
        features = features.to(device)
        
        # Make prediction
        with torch.no_grad():  # Disable gradient calculation for inference
            logits = model(features) 
            probability = torch.sigmoid(logits)  # Apply sigmoid to convert logits to probability (0-1)
            
            predicted_key = (probability > 0.5).int().item()
            
            confidence = probability.item() if predicted_key == 1 else 1 - probability.item()
        
        return predicted_key, confidence
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

def batch_predict(audio_dir, model, device):
    """
    Process all audio files in a directory and make predictions
    
    Args:
        audio_dir (str): Directory containing audio files
        model (torch.nn.Module): Loaded model
        device (torch.device): Device to run inference on
        
    Returns:
        dict: Dictionary mapping filenames to predictions
    """
    audio_dir = Path(audio_dir)
    results = {}
    audio_files = list(audio_dir.glob('*.wav'))
    
    if not audio_files:
        print(f"No .wav files found in {audio_dir}")
        return results
    
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        predicted_key, confidence = predict(audio_path, model, device)
        if predicted_key is not None:
            results[audio_path.name] = {
                'predicted_class': labelEncoder.inverse_transform([predicted_key])[0],
                'confidence': confidence
            }
    
    return results

def structured_batch_predict(test_dir, model, device):
    """
    Process audio files in a structured test directory with cry/not_cry subdirectories
    
    Args:
        test_dir (str): Directory containing cry and not_cry subdirectories
        model (torch.nn.Module): Loaded model
        device (torch.device): Device to run inference on
        
    Returns:
        dict: Dictionary with prediction results and evaluation metrics
    """
    test_dir = Path(test_dir)
    cry_dir = test_dir / 'cry'
    not_cry_dir = test_dir / 'not_cry'
    
    results = {}
    metrics = {
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0,
    }
    
    # Process cry directory 
    if cry_dir.exists():
        cry_files = list(cry_dir.glob('*.wav'))
        print(f"Found {len(cry_files)} files in cry directory")
        for audio_path in tqdm(cry_files, desc="Processing 'cry' files"):
            predicted_key, confidence = predict(audio_path, model, device)
            if predicted_key is not None:
                results[str(audio_path.relative_to(test_dir))] = {
                    'true_class': 'cry',
                    'predicted_class': labelEncoder.inverse_transform([predicted_key])[0],
                    'confidence': confidence
                }
                
                if predicted_key == labelEncoder.transform(['cry'])[0]:
                    metrics['true_positives'] += 1
                else:
                    metrics['false_negatives'] += 1
    
    # Process not_cry directory
    if not_cry_dir.exists():
        not_cry_files = list(not_cry_dir.glob('*.wav'))
        print(f"Found {len(not_cry_files)} files in not_cry directory")
        for audio_path in tqdm(not_cry_files, desc="Processing 'not_cry' files"):
            predicted_key, confidence = predict(audio_path, model, device)
            if predicted_key is not None:
                results[str(audio_path.relative_to(test_dir))] = {
                    'true_class': 'not_cry',
                    'predicted_class': labelEncoder.inverse_transform([predicted_key])[0],
                    'confidence': confidence
                }
                
                if predicted_key == labelEncoder.transform(['not_cry'])[0]:
                    metrics['true_negatives'] += 1
                else:
                    metrics['false_positives'] += 1
    
    # Calculate evaluation metrics
    total_predictions = sum(metrics.values())
    if total_predictions > 0:
        metrics['accuracy'] = (metrics['true_positives'] + metrics['true_negatives']) / total_predictions
        
        if (metrics['true_positives'] + metrics['false_positives']) > 0:
            metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
        else:
            metrics['precision'] = 0
            
        if (metrics['true_positives'] + metrics['false_negatives']) > 0:
            metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
        else:
            metrics['recall'] = 0
            
        if (metrics['precision'] + metrics['recall']) > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0
    
    return {'results': results, 'metrics': metrics}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Baby Cry Detection Inference')
    parser.add_argument('--model', type=str, default='runs/latest/checkpoints/best_model.pth',
                        help='Path to model checkpoint (default: runs/latest/checkpoints/best_model.pth)')
    parser.add_argument('--audio', type=str, default=None,
                        help='Path to audio file for single prediction')
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory containing audio files for batch prediction')
    parser.add_argument('--structured', action='store_true',
                        help='Process test directory with cry/not_cry subdirectory structure')
    parser.add_argument('--output', type=str, default='results/predictions.csv',
                        help='Path to output CSV file for batch predictions (default: predictions.csv)')
    parser.add_argument('--example', action='store_true',
                        help='Show example usage for testing an audio file')
    
    args = parser.parse_args()
    
    # Show example usage if requested
    if args.example:
        print("\nExample usage for testing a single audio file:")
        print("python src/inference.py --model runs/latest/checkpoints/best_model.pth --audio path/to/audio_file.wav")
        print("\nExample usage for batch processing:")
        print("python src/inference.py --model runs/latest/checkpoints/best_model.pth --dir path/to/audio_folder")
        print("\nExample usage for structured directory with cry/not_cry subdirectories:")
        print("python src/inference.py --model runs/latest/checkpoints/best_model.pth --dir path/to/test_directory --structured")
        
        print("\n--- Các ví dụ sử dụng ---")
        print("\nKiểm tra một file âm thanh:")
        print("python src/inference.py --model runs/latest/checkpoints/best_model.pth --audio data/samples/baby_cry_sample.wav")
        #python src/inference.py --model D:\Git\baby_cry_detection\runs\20250330_203609\checkpoints\best_model.pth --audio D:\Git\baby_cry_detection\data\dataset\test\cry\0c8f14a9-6999-485b-97a2-913c1cbf099c-1430760379259-1.7-m-26-hu_part1.wav
        print("\nKiểm tra nhiều file âm thanh trong một thư mục:")
        print("python src/inference.py --model runs/latest/checkpoints/best_model.pth --dir data/samples")
        #python src/inference.py --model D:\Git\baby_cry_detection\runs\20250330_203609\checkpoints\best_model.pth  --dir D:\Git\baby_cry_detection\data\dataset\test\cry
        print("\nKiểm tra thư mục có cấu trúc chứa nhãn (cry/not_cry):")
        print("python src/inference.py --model runs/latest/checkpoints/best_model.pth --dir data/test_dataset --structured --output ketqua.csv")
        #python src/inference.py --model D:\Git\baby_cry_detection\runs\20250330_203609\checkpoints\best_model.pth  --dir D:\Git\baby_cry_detection\data\dataset\test --structured --output D:\Git\baby_cry_detection\results\test_predictions.csv

        return
    
    # Validate arguments
    if args.audio is None and args.dir is None:
        print("Error: Either --audio or --dir must be specified.")
        print("For single file test: python src/inference.py --model MODEL_PATH --audio AUDIO_PATH")
        print("To see examples, use: python src/inference.py --example")
        parser.print_help()
        return
    
    # Ensure model path exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model path {args.model} does not exist.")
        return
    
    # Load model
    try:
        model, device = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Single prediction
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"Error: Audio file {args.audio} does not exist.")
            return
        
        predicted_key, confidence = predict(audio_path, model, device)
        
        if predicted_key is not None:
            print("\n----- Prediction Result -----")
            print(f"File: {audio_path.name}")
            print(f"Predicted class: {labelEncoder.inverse_transform([predicted_key])[0]}")
            print(f"Confidence: {confidence:.2%}")
            print("----------------------------")
    
    # Batch prediction
    if args.dir:
        audio_dir = Path(args.dir)
        if not audio_dir.exists() or not audio_dir.is_dir():
            print(f"Error: Directory {args.dir} does not exist or is not a directory.")
            return
        
        if args.structured:
            # Process structured test directory
            output = structured_batch_predict(audio_dir, model, device)
            results = output['results']
            metrics = output['metrics']
            
            # Print summary with evaluation metrics
            if results:
                print("\n----- Structured Batch Prediction Summary -----")
                print(f"Total files processed: {len(results)}")
                print(f"\nEvaluation Metrics:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1_score']:.4f}")
                print(f"\nConfusion Matrix:")
                print(f"True Positives: {metrics['true_positives']}")
                print(f"False Positives: {metrics['false_positives']}")
                print(f"True Negatives: {metrics['true_negatives']}")
                print(f"False Negatives: {metrics['false_negatives']}")
                
                # Save results to CSV
                try:
                    import pandas as pd
                    df = pd.DataFrame([
                        {
                            'filename': filename,
                            'true_class': result['true_class'],
                            'predicted_class': result['predicted_class'],
                            'confidence': f"{result['confidence']:.4f}",
                            'correct': result['true_class'] == result['predicted_class']
                        }
                        for filename, result in results.items()
                    ])
                    df.to_csv(args.output, index=False)
                    print(f"\nResults saved to {args.output}")
                except Exception as e:
                    print(f"Error saving results to CSV: {e}")
        else:
            # Standard batch prediction on flat directory
            results = batch_predict(audio_dir, model, device)
            
            # Print summary
            if results:
                cry_count = sum(1 for r in results.values() if r['predicted_class'] == 'cry')
                not_cry_count = len(results) - cry_count
                
                print("\n----- Batch Prediction Summary -----")
                print(f"Total files processed: {len(results)}")
                print(f"Cry detected: {cry_count} files ({cry_count/len(results):.2%})")
                print(f"No cry detected: {not_cry_count} files ({not_cry_count/len(results):.2%})")
                
                # Save results to CSV
                try:
                    import pandas as pd
                    df = pd.DataFrame([
                        {
                            'filename': filename,
                            'predicted_class': result['predicted_class'],
                            'confidence': f"{result['confidence']:.4f}"
                        }
                        for filename, result in results.items()
                    ])
                    df.to_csv(args.output, index=False)
                    print(f"\nResults saved to {args.output}")
                except Exception as e:
                    print(f"Error saving results to CSV: {e}")

if __name__ == '__main__':
    main()