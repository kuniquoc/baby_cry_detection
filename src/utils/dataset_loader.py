import os
import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Replace relative imports with absolute imports
from src.data_processing import extract_mfcc

class DatasetLoader:
    """
    DatasetLoader class for loading and processing audio data for baby cry detection.
    Uses pre-defined functions from data_processing directory.
    """
    
    def __init__(self, 
                 data_dir='data/dataset', 
                 processed_dir='data/processed', 
                 sample_rate=16000, 
                 n_mels=40, 
                 n_fft=512, 
                 hop_length=160, 
                 duration=3.0):
        """
        Initialize DatasetLoader with configuration parameters.
        
        Args:
            data_dir: Path to data directory
            processed_dir: Path to store processed features
            sample_rate: Audio sampling rate (Hz)
            n_mels: Number of mel bands for feature extraction
            n_fft: FFT window size
            hop_length: Hop length for FFT
            duration: Audio segment length in seconds
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.classes = ['not_cry', 'cry']  # Class definitions
        self.label_encoder = LabelEncoder().fit(self.classes)
        
        # Create directory structure for processed data
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create directory structure for processed data."""
        for split in ['train', 'val', 'test']:
            for label in self.classes:
                os.makedirs(self.processed_dir / split / label, exist_ok=True)
        
    def load_metadata(self):
        """
        Create metadata from directory structure.
        
        Returns:
            DataFrame with audio file information and labels
        """
        metadata = []
        
        # Scan through train, val, test directories
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue
                
            # Scan through class directories
            for label in self.classes:
                label_dir = split_dir / label
                if not label_dir.exists():
                    continue
                    
                # Get all wav files
                for audio_file in label_dir.glob('*.wav'):
                    metadata.append({
                        'file_path': str(audio_file),
                        'label': label,
                        'split': split
                    })
        
        # Create DataFrame
        df = pd.DataFrame(metadata)
        
        # Log dataset statistics
        if not df.empty:
            print(f"Loaded metadata with {len(df)} files")
            print(f"Split distribution: {df['split'].value_counts().to_dict()}")
            print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        else:
            print("No files found in the data directory")
        
        return df
        
    def extract_features(self, audio_path, augment=False):
        """
        Extract MFCC features from audio file.
        
        Args:
            audio_path: Path to audio file
            augment: Whether to apply data augmentation
            
        Returns:
            torch.Tensor containing extracted features or None if extraction fails
        """
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract MFCCs
            mfccs = extract_mfcc(
                y, 
                sr=self.sample_rate,
                n_mfcc=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Apply data augmentation if requested
            if augment:
                mfccs = self._apply_augmentation(mfccs)
                
            # Convert to PyTorch tensor and add channel dimension
            features = torch.from_numpy(mfccs).float().unsqueeze(0)  # [1, n_mels, time]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def get_processed_path(self, file_path):
        """
        Generate path for processed feature file.
        
        Args:
            file_path: Path to original audio file
            
        Returns:
            Path to processed feature file (.npy)
        """
        orig_path = Path(file_path)
        parts = orig_path.parts
        
        # Find location of split directory in path
        for i, part in enumerate(parts):
            if part in ['train', 'val', 'test']:
                split = part
                label = parts[i+1]  # Assume label is directory after split
                filename = orig_path.stem
                
                return self.processed_dir / split / label / f"{filename}.npy"
                
        # Fallback if path structure doesn't match expected format
        return self.processed_dir / f"{orig_path.stem}.npy"
    
    def save_features(self, features, file_path):
        """
        Save extracted features to processed directory.
        
        Args:
            features: Extracted features to save
            file_path: Path to original audio file
        """
        processed_path = self.get_processed_path(file_path)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert torch tensor to numpy if needed
            features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
            np.save(processed_path, features_np)
        except Exception as e:
            print(f"Error saving feature file {processed_path}: {e}")
            # Remove potentially corrupted file
            if processed_path.exists():
                try:
                    os.remove(processed_path)
                except:
                    pass
    
    def load_features(self, file_path):
        """
        Load pre-processed features from disk.
        
        Args:
            file_path: Path to original audio file
            
        Returns:
            torch.Tensor of loaded features or None if file doesn't exist
        """
        processed_path = self.get_processed_path(file_path)
        
        if processed_path.exists():
            try:
                features_np = np.load(processed_path)
                return torch.from_numpy(features_np).float()
            except Exception as e:
                print(f"Error loading feature file {processed_path}: {e}")
                # Delete corrupted file for recomputation
                try:
                    os.remove(processed_path)
                    print(f"Removed corrupted feature file: {processed_path}")
                except Exception as ex:
                    print(f"Could not remove file: {ex}")
                return None
        
        return None
    
    def convert_audio_to_mfcc(self, force_recompute=False):
        """
        Convert all audio files in dataset to MFCC features and save as .npy files.
        
        Args:
            force_recompute: Force recomputation of all features
        
        Returns:
            Number of processed files
        """
        print("Converting audio files to MFCC features...")
        
        df = self.load_metadata()
        
        if df.empty:
            print("No audio files found in metadata.")
            return 0
        
        # Initialize counters
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Process each file
        file_paths = df['file_path'].values
        
        for file_path in tqdm(file_paths, desc="Converting to MFCCs"):
            processed_path = self.get_processed_path(file_path)
            
            # Skip if already processed and not forcing recomputation
            if processed_path.exists() and not force_recompute:
                skipped_count += 1
                continue
                
            # Extract features
            features = self.extract_features(file_path, augment=False)
            
            if features is not None:
                self.save_features(features, file_path)
                processed_count += 1
            else:
                failed_count += 1
                
        print(f"MFCC conversion completed:")
        print(f"- Processed: {processed_count} files")
        print(f"- Skipped (already exists): {skipped_count} files")
        print(f"- Failed: {failed_count} files")
        
        return processed_count
    
    def _precompute_mfccs(self, df):
        """
        Load pre-computed MFCCs from processed directory.
        
        Args:
            df: DataFrame with file information
            
        Returns:
            Tuple of (features_list, labels_list)
        """
        if df.empty:
            return [], []
        
        features_list = []
        labels_list = []
        
        # Get file paths and labels
        file_paths = df['file_path'].values
        labels = self.label_encoder.transform(df['label'].values)
        
        # Load features from processed directory
        for file_path, label in tqdm(zip(file_paths, labels), 
                                    total=len(file_paths), 
                                    desc="Loading MFCCs"):
            features = self.load_features(file_path)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(label)
            else:
                print(f"Warning: Missing processed feature for {file_path}")
        
        return features_list, labels_list

    def prepare_dataset(self, batch_size=32, num_workers=4):
        """
        Prepare PyTorch datasets by loading pre-computed MFCC features.
        
        Args:
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes for DataLoader
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        df = self.load_metadata()
        
        # Split data by set
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']
        
        print("Loading MFCC features from processed directory...")
        
        # Load pre-computed features
        train_features, train_labels = self._precompute_mfccs(train_df)
        val_features, val_labels = self._precompute_mfccs(val_df)
        test_features, test_labels = self._precompute_mfccs(test_df)
        
        print(f"Loaded features - Train: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")
        
        # Create TensorDatasets
        train_loader = self._create_data_loader(train_features, train_labels, batch_size, num_workers, shuffle=True)
        val_loader = self._create_data_loader(val_features, val_labels, batch_size, num_workers, shuffle=False)
        test_loader = self._create_data_loader(test_features, test_labels, batch_size, num_workers, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def _create_data_loader(self, features, labels, batch_size, num_workers, shuffle):
        """Helper method to create DataLoader from features and labels."""
        if not features or len(features) == 0:
            return None
            
        dataset = TensorDataset(
            torch.stack(features),
            torch.tensor(labels, dtype=torch.float)
        )
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers, 
            pin_memory=True
        )
    
    def _apply_augmentation(self, features):
        """
        Apply data augmentation to features.
        
        Args:
            features: Input features (MFCC)
            
        Returns:
            Augmented features
        """
        choice = np.random.randint(0, 3)
        
        if choice == 0:  
            # Frequency Masking (reduced intensity)
            freq_mask_param = 5
            freq_mask_idx = np.random.randint(0, features.shape[0] - freq_mask_param)
            features[freq_mask_idx:freq_mask_idx + freq_mask_param, :] = 0

        elif choice == 1:  
            # Time Masking (reduced intensity)
            time_mask_param = 5
            time_mask_idx = np.random.randint(0, features.shape[1] - time_mask_param)
            features[:, time_mask_idx:time_mask_idx + time_mask_param] = 0

        elif choice == 2:  
            # Small Frequency Shift
            shift = np.random.randint(-1, 2)
            if shift > 0:
                features[shift:, :] = features[:-shift, :]
                features[:shift, :] = 0
            elif shift < 0:
                features[:shift, :] = features[-shift:, :]
                features[shift:, :] = 0
        
        return features
        
    def get_class_weights(self):
        """
        Calculate class weights for handling imbalanced data.
        
        Returns:
            torch.Tensor: Tensor containing positive class weight for BCEWithLogitsLoss
        """
        df = self.load_metadata()
        train_df = df[df['split'] == 'train']
        
        class_counts = train_df['label'].value_counts().to_dict()
        print("Class counts:", class_counts)
        
        # Calculate pos_weight: negative samples / positive samples
        pos_weight = class_counts.get('not_cry', 0) / max(class_counts.get('cry', 0), 1)
        print(f"Using pos_weight={pos_weight} for BCEWithLogitsLoss")
        
        return torch.tensor([pos_weight], dtype=torch.float)


# Example usage
if __name__ == "__main__":
    # Initialize DatasetLoader
    loader = DatasetLoader(data_dir='data/dataset')
    
    # Load metadata and display dataset statistics
    metadata = loader.load_metadata()
    
    # Convert audio to MFCC features (uncomment to run)
    loader.convert_audio_to_mfcc()
    
    # # Prepare datasets for training
    # train_loader, val_loader, test_loader = loader.prepare_dataset(batch_size=32)
    
    # # Check a batch from train_loader
    # if train_loader:
    #     for batch_features, batch_labels in train_loader:
    #         print(f"Batch shape: {batch_features.shape}")
    #         print(f"Labels shape: {batch_labels.shape}")
    #         break

