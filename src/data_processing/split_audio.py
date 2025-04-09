import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from collections import defaultdict

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
    file_path = Path(file_path)
    
    # Get label (cry/not_cry) from the path
    label = file_path.parent.name  # cry or not_cry
    
    # Load audio file
    y, sr = librosa.load(file_path, sr=16000)
    
    # Calculate samples per segment & step size
    segment_samples = segment_length * sr
    step_samples = hop_length * sr
    
    # Get base filename
    base_filename = file_path.stem
    
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
    
    print(f"Created {len(segments)} segments from {file_path.name} (regular segmentation)")
    return segments, label

def save_segments(segments, output_dir, label):
    """Save segments to the specified directory.
    
    Args:
        segments: List of (audio, name, sr) tuples
        output_dir: Base output directory
        label: Label directory (cry/not_cry)
    """
    save_dir = output_dir / label
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for segment, name, sr in segments:
        segment_path = save_dir / f"{name}.wav"
        sf.write(segment_path, segment, sr)
    
    return len(segments)

def process_directory(input_dir, output_dir, segment_length=3, hop_length=1):
    """Process all audio files in a directory
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save segmented files
        segment_length: Length of each segment in seconds
        hop_length: Hop length between segments in seconds
    
    Returns:
        dict: Counts of segments by label
    """
    segment_counts = defaultdict(int)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all valid directories
    for label_dir in ['cry', 'not_cry']:
        current_dir = input_dir / label_dir
        if not current_dir.exists():
            continue
            
        print(f"\nProcessing {label_dir} files from {current_dir}...")
        
        # Process each audio file
        segments_list = []
        for file_path in current_dir.glob('*.wav'):
            segments, _ = split_audio(file_path, segment_length, hop_length)
            segments_list.extend(segments)
        
        # Save segments
        if segments_list:
            count = save_segments(segments_list, output_dir, label_dir)
            segment_counts[label_dir] += count
            print(f"Saved {count} {label_dir} segments to {output_dir / label_dir}")
    
    return segment_counts

def main():
    # Set up directories
    base_dir = Path('data')
    raw_root = base_dir / "raw"
    output_root = base_dir / "segmented"
    
    # Process all directories
    segment_counts = process_directory(raw_root, output_root)
    
    # Print final statistics
    print("\nSegmentation Statistics:")
    print("-----------------------")
    total_segments = 0
    
    for label in ['cry', 'not_cry']:
        count = segment_counts[label]
        print(f"  {label}: {count} segments")
        total_segments += count
    
    print(f"\nTotal segments created: {total_segments}")
    print("(All segments were created with regular segmentation)")

if __name__ == "__main__":
    main()
