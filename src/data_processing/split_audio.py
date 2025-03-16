import os
import librosa
import soundfile as sf
import numpy as np
from collections import defaultdict
from pathlib import Path
import random

def split_audio(file_path, output_dir, segment_length=3, overlap=1, is_augmented=False):
    """Split audio file into segments.
    
    Args:
        file_path: Path to the audio file
        output_dir: Directory to save segments
        segment_length: Length of each segment in seconds
        overlap: Overlap between segments in seconds
        is_augmented: Whether this is an augmented file
    
    Returns:
        tuple: (segments, split) where segments is list of segments and split is 'train' or 'val'
    """
    # Get label (cry/not_cry) and split (train/val) from the path
    path_parts = file_path.split(os.sep)
    label = path_parts[-2]  # cry or not_cry
    split = path_parts[-3] if not is_augmented else 'train'  # Use path for original, 'train' for augmented
    
    # Load audio file
    y, sr = librosa.load(file_path, sr=16000)
    
    # Calculate samples per segment & step size
    segment_samples = segment_length * sr
    step_samples = (segment_length - overlap) * sr
    
    # Get base filename
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Split file into segments
    segments = []
    start = 0
    index = 1
    while start + segment_samples <= len(y):
        segment = y[start:start + segment_samples]
        name = f"{base_filename}_aug_part{index}" if is_augmented else f"{base_filename}_part{index}"
        segments.append((segment, name, sr))
        start += step_samples
        index += 1
    
    return segments, split

def save_segments(segments, output_dir, label, split='train'):
    """Save segments to the specified directory.
    
    Args:
        segments: List of (audio, name, sr) tuples
        output_dir: Base output directory
        label: Label directory (cry/not_cry)
        split: Split directory (train/val)
    """
    save_dir = os.path.join(output_dir, split, label)
    os.makedirs(save_dir, exist_ok=True)
    
    for segment, name, sr in segments:
        segment_path = os.path.join(save_dir, f"{name}.wav")
        sf.write(segment_path, segment, sr)
    
    return len(segments)

def main():
    # Set up directories
    base_dir = Path('data')
    dataset_root = base_dir / "dataset"
    augmented_root = base_dir / "augmented"
    output_root = base_dir / "segmented"
    
    # Initialize counters and segment storage
    segment_counts = defaultdict(lambda: defaultdict(int))
    all_segments = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Process original files
    for split in ['train', 'val']:
        for label in ["cry", "not_cry"]:
            print(f"\nProcessing original {split}/{label} files...")
            label_dir = dataset_root / split / label
            if label_dir.exists():
                for file_name in os.listdir(label_dir):
                    if file_name.endswith(".wav"):
                        file_path = os.path.join(label_dir, file_name)
                        segments, _ = split_audio(file_path, output_root)
                        all_segments[label][split]['original'].extend(segments)
                        print(f"Created {len(segments)} segments from {file_name}")
    
    # Process augmented files
    for label in ["cry", "not_cry"]:
        aug_dir = augmented_root / label
        if aug_dir.exists():
            print(f"\nProcessing augmented {label} files...")
            for file_name in os.listdir(aug_dir):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(aug_dir, file_name)
                    segments, _ = split_audio(file_path, output_root, is_augmented=True)
                    all_segments[label]['train']['augmented'].extend(segments)
                    print(f"Created {len(segments)} segments from augmented {file_name}")
    
    # Distribute segments
    for label in ["cry", "not_cry"]:
        # Save original segments for each split
        for split in ['train', 'val']:
            if len(all_segments[label][split]['original']) > 0:
                # Save original segments
                count = save_segments(all_segments[label][split]['original'], output_root, label, split)
                segment_counts[split][label] += count
                print(f"Saved {count} original {label} segments to {split}")
        
        # Handle augmented segments
        if label == "not_cry" and len(all_segments[label]['train']['augmented']) > 0:
            # Shuffle augmented segments
            random.shuffle(all_segments[label]['train']['augmented'])
            
            # Select required number of augmented segments for train only
            aug_train = all_segments[label]['train']['augmented'][:2300]
            
            # Save augmented segments to train only
            train_count = save_segments(aug_train, output_root, label, 'train')
            segment_counts['train'][label] += train_count
            print(f"Added {train_count} augmented {label} segments to train")
        
        elif label == "cry" and len(all_segments[label]['train']['augmented']) > 0:
            # For cry, add all augmented segments to train
            train_count = save_segments(all_segments[label]['train']['augmented'], output_root, label, 'train')
            segment_counts['train'][label] += train_count
            print(f"Added {train_count} augmented {label} segments to train")
    
    # Print final statistics
    print("\nSegmentation Statistics:")
    print("-----------------------")
    total_segments = 0
    
    for split in ['train', 'val']:
        print(f"\n{split.upper()} Split:")
        split_total = 0
        for label in ['cry', 'not_cry']:
            count = segment_counts[split][label]
            print(f"  {label}: {count} segments")
            split_total += count
        print(f"  Total {split}: {split_total} segments")
        total_segments += split_total
    
    print(f"\nTotal segments created: {total_segments}")

if __name__ == "__main__":
    main()
