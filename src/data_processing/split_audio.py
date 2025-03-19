import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from .preprocess import apply_vad

def analyze_audio_segments(audio_data, sr=16000, segment_length=3, hop_length=1):
    """Analyze audio data by splitting into segments and applying VAD.
    
    Args:
        audio_data: Audio signal array
        sr: Sample rate (default: 16000)
        segment_length: Length of each segment in seconds
        hop_length: Hop length between segments in seconds
    
    Returns:
        list: List of dictionaries containing segment information
    """
    # Calculate samples per segment & step size
    segment_samples = segment_length * sr
    step_samples = hop_length * sr
    
    # Split file into segments using sliding window
    segments_info = []
    start_sample = 0
    segment_index = 0
    
    while start_sample + segment_samples <= len(audio_data):
        # Extract segment
        segment = audio_data[start_sample:start_sample + segment_samples]
        
        # Apply VAD to segment
        audio_vad, is_crying, mean_f0 = apply_vad(segment, sr)
        
        if is_crying:
            # Calculate time range for this segment
            start_time = start_sample / sr
            end_time = (start_sample + segment_samples) / sr
            
            segment_info = {
                'audio': audio_vad,
                'start_time': start_time,
                'end_time': end_time,
                'index': segment_index,
                'f0': mean_f0
            }
            segments_info.append(segment_info)
        
        start_sample += step_samples
        segment_index += 1
    
    return segments_info

def pad_or_trim_segment(audio, sr, target_length=3):
    """Ensure segment is exactly the target length"""
    target_samples = int(sr * target_length)
    if len(audio) > target_samples:
        return audio[:target_samples]
    elif len(audio) < target_samples:
        return np.pad(audio, (0, target_samples - len(audio)))
    return audio

def split_audio(file_path, segment_length=3, hop_length=1):
    """Split audio file into segments using sliding window.
    
    Args:
        file_path: Path to the audio file
        segment_length: Length of each segment in seconds
        hop_length: Hop length between segments in seconds
    
    Returns:
        list: List of (audio_segment, segment_name, sample_rate) tuples
    """
    # Get label (cry/not_cry) from the path
    path_parts = str(file_path).split(os.sep)
    label = path_parts[-2]  # cry or not_cry
    
    # Load audio file
    y, sr = librosa.load(file_path, sr=16000)
    
    # Calculate samples per segment & step size
    segment_samples = segment_length * sr
    step_samples = hop_length * sr
    
    # Get base filename
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Split file into segments using sliding window
    segments = []
    start = 0
    index = 1
    while start + segment_samples <= len(y):
        segment = y[start:start + segment_samples]
        name = f"{base_filename}_part{index}"
        segments.append((segment, name, sr))
        start += step_samples
        index += 1
    
    return segments, label

def save_segments(segments, output_dir, label):
    """Save segments to the specified directory.
    
    Args:
        segments: List of (audio, name, sr) tuples
        output_dir: Base output directory
        label: Label directory (cry/not_cry)
    """
    save_dir = os.path.join(output_dir, label)
    os.makedirs(save_dir, exist_ok=True)
    
    for segment, name, sr in segments:
        segment_path = os.path.join(save_dir, f"{name}.wav")
        sf.write(segment_path, segment, sr)
    
    return len(segments)

def main():
    # Set up directories
    base_dir = Path('data')
    raw_root = base_dir / "raw"
    output_root = base_dir / "segmented"
    
    # Initialize counters and segment storage
    segment_counts = defaultdict(int)
    all_segments = defaultdict(list)
    
    # Process files from raw directory
    for label in ["cry", "not_cry"]:
        print(f"\nProcessing {label} files from raw directory...")
        label_dir = raw_root / label
        if label_dir.exists():
            for file_name in os.listdir(label_dir):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(label_dir, file_name)
                    segments, _ = split_audio(file_path)
                    all_segments[label].extend(segments)
                    print(f"Created {len(segments)} segments from {file_name}")
    
    # Save segments
    for label in ["cry", "not_cry"]:
        if len(all_segments[label]) > 0:
            count = save_segments(all_segments[label], output_root, label)
            segment_counts[label] += count
            print(f"Saved {count} {label} segments to segmented/{label}")
    
    # Print final statistics
    print("\nSegmentation Statistics:")
    print("-----------------------")
    total_segments = 0
    
    for label in ['cry', 'not_cry']:
        count = segment_counts[label]
        print(f"  {label}: {count} segments")
        total_segments += count
    
    print(f"\nTotal segments created: {total_segments}")

if __name__ == "__main__":
    main()
