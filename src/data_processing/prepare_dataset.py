import os
import shutil
import random

def create_dataset_splits(segmented_dir, dataset_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into train, validation and test sets.
    test_ratio is implicitly 1 - (train_ratio + val_ratio)
    """
    # Create main dataset directory
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create train, val, test directories
    splits = ['train', 'val', 'test']
    for split in splits:
        for label in ['cry', 'not_cry']:
            os.makedirs(os.path.join(dataset_dir, split, label), exist_ok=True)
    
    # Process each class (cry and not_cry)
    for label in ['cry', 'not_cry']:
        # Get all files for this class
        source_dir = os.path.join(segmented_dir, label)
        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} does not exist")
            continue
            
        files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
        random.shuffle(files)
        
        # Calculate split indices
        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        
        # Split files
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Copy files to respective directories
        for file_name in train_files:
            src = os.path.join(source_dir, file_name)
            dst = os.path.join(dataset_dir, 'train', label, file_name)
            shutil.copy2(src, dst)
            
        for file_name in val_files:
            src = os.path.join(source_dir, file_name)
            dst = os.path.join(dataset_dir, 'val', label, file_name)
            shutil.copy2(src, dst)
            
        for file_name in test_files:
            src = os.path.join(source_dir, file_name)
            dst = os.path.join(dataset_dir, 'test', label, file_name)
            shutil.copy2(src, dst)
        
        print(f"{label}:")
        print(f"  Train: {len(train_files)} files")
        print(f"  Validation: {len(val_files)} files")
        print(f"  Test: {len(test_files)} files")

if __name__ == "__main__":
    segmented_dir = "data/segmented"
    dataset_dir = "data/dataset"
    create_dataset_splits(segmented_dir, dataset_dir) 