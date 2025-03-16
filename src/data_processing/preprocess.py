import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def apply_vad(audio, sr=16000, frame_length=2048, hop_length=512,
              energy_threshold=0.01, zcr_threshold_low=0.1, zcr_threshold_high=0.5):
    """Advanced Voice Activity Detection optimized for children's crying.
    
    Args:
        audio: Input audio signal
        sr: Sampling rate (default: 16kHz)
        frame_length: Length of each frame
        hop_length: Number of samples between frames
        energy_threshold: Threshold for RMS energy
        zcr_threshold_low: Lower threshold for zero crossing rate
        zcr_threshold_high: Upper threshold for zero crossing rate
    
    Returns:
        numpy.ndarray: Filtered audio signal with crying segments
    """
    # 1. Compute frame-wise features
    # RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True
    )[0]
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True
    )[0]
    
    # Fundamental frequency (F0)
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=200,  # Min frequency for crying
        fmax=600,  # Max frequency for crying
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # 2. Create masks based on each feature
    # Energy mask
    energy_mask = rms > energy_threshold
    
    # ZCR mask (exclude segments with too low or too high ZCR)
    zcr_mask = (zcr > zcr_threshold_low) & (zcr < zcr_threshold_high)
    
    # F0 mask (typical crying frequency range: 200-600 Hz)
    f0_mask = np.zeros_like(energy_mask)
    valid_f0 = ~np.isnan(f0)
    f0_mask[valid_f0] = (f0[valid_f0] >= 200) & (f0[valid_f0] <= 600)
    
    # 3. Combine masks
    combined_mask = energy_mask & zcr_mask & (f0_mask | voiced_flag)
    
    # 4. Convert frame-wise mask to sample-wise mask
    mask_samples = np.repeat(combined_mask, hop_length)
    
    # Pad or truncate mask to match audio length
    if len(mask_samples) < len(audio):
        mask_samples = np.pad(mask_samples, (0, len(audio) - len(mask_samples)))
    else:
        mask_samples = mask_samples[:len(audio)]
    
    # 5. Apply mask to audio
    audio_vad = audio * mask_samples
    
    # 6. Trim silent edges
    audio_vad, _ = librosa.effects.trim(audio_vad, top_db=20)
    
    return audio_vad

def extract_mfcc(audio, sr=16000, n_mfcc=40, n_fft=512, hop_length=160):
    """Extract MFCC features from audio signal.
    
    Args:
        audio: Input audio signal
        sr: Sampling rate
        n_mfcc: Number of MFCC coefficients
        n_fft: Length of FFT window
        hop_length: Number of samples between successive frames
    
    Returns:
        numpy.ndarray: MFCC features
    """
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Normalize features
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
    
    return mfccs

def process_audio_file(file_path, output_dir, label):
    """Process a single audio file with VAD and feature extraction.
    
    Args:
        file_path: Path to the audio file
        output_dir: Directory to save processed features
        label: Class label (cry/not_cry)
    
    Returns:
        bool: True if processing was successful
    """
    try:
        # Load and normalize audio
        audio, sr = librosa.load(str(file_path), sr=16000)
        audio = librosa.util.normalize(audio)
        
        # Apply VAD
        audio_vad = apply_vad(audio, sr)
        
        # Only process if VAD returns valid audio
        if len(audio_vad) > 0:
            # Extract MFCC features
            features = extract_mfcc(audio_vad, sr)
            
            # Create output path
            output_path = output_dir / label / f"{file_path.stem}.npy"
            os.makedirs(output_path.parent, exist_ok=True)
            
            # Save features
            np.save(output_path, features)
            return True
            
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    # Set up directories
    base_dir = Path('data')
    segmented_dir = base_dir / 'segmented'
    processed_dir = base_dir / 'processed'
    
    # Initialize counters
    processed_counts = defaultdict(lambda: defaultdict(int))
    
    # Process files for each split and label
    for split in ['train', 'val']:
        split_dir = segmented_dir / split
        if not split_dir.exists():
            continue
            
        print(f"\nProcessing {split} split...")
        
        for label in ['cry', 'not_cry']:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue
                
            # Get all wav files in the directory
            wav_files = list(label_dir.glob('*.wav'))
            print(f"Found {len(wav_files)} {label} files")
            
            # Process each file
            successful = 0
            for file_path in tqdm(wav_files, desc=f"Processing {split}/{label}"):
                if process_audio_file(file_path, processed_dir / split, label):
                    successful += 1
                    processed_counts[split][label] += 1
            
            print(f"Successfully processed {successful}/{len(wav_files)} files")
    
    # Print final statistics
    print("\nProcessing Statistics:")
    print("--------------------")
    total_processed = 0
    
    for split in ['train', 'val']:
        print(f"\n{split.upper()} Split:")
        split_total = 0
        for label in ['cry', 'not_cry']:
            count = processed_counts[split][label]
            print(f"  {label}: {count} files")
            split_total += count
        print(f"  Total {split}: {split_total} files")
        total_processed += split_total
    
    print(f"\nTotal files processed: {total_processed}")

if __name__ == "__main__":
    main()
