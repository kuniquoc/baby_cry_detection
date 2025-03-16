import os
import numpy as np
import librosa
import torchaudio
import torchaudio.transforms as T
import soundfile as sf

# Original data directory & Augmented data directory
SEGMENTED_DATA_DIR = "data/segmented"
AUGMENTED_DATA_DIR = "data/augmented"

# Function to create augmentations and save new files
def augment_and_save(file_path, label, augment_id):
    # Read audio file
    waveform, sr = librosa.load(file_path, sr=16000)

    # 1. Add Gaussian noise
    noise = np.random.normal(0, 0.005, waveform.shape)
    noisy_waveform = waveform + noise

    # 2. Time Shifting
    shift = np.random.randint(100, 1000)
    shifted_waveform = np.roll(waveform, shift)

    # 3. Time Stretching
    stretched_waveform = librosa.effects.time_stretch(waveform, rate=np.random.uniform(0.8, 1.2))

    # Create directory for augmented data
    save_dir = os.path.join(AUGMENTED_DATA_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    # Save each augmented variant to new files
    sf.write(os.path.join(save_dir, f"{os.path.basename(file_path).replace('.wav', '')}_noise_{augment_id}.wav"), noisy_waveform, sr)
    print(f"Saved: {os.path.join(save_dir, f"{os.path.basename(file_path).replace('.wav', '')}_noise_{augment_id}.wav")}")
    sf.write(os.path.join(save_dir, f"{os.path.basename(file_path).replace('.wav', '')}_shift_{augment_id}.wav"), shifted_waveform, sr)
    print(f"Saved: {os.path.join(save_dir, f"{os.path.basename(file_path).replace('.wav', '')}_shift_{augment_id}.wav")}")
    sf.write(os.path.join(save_dir, f"{os.path.basename(file_path).replace('.wav', '')}_stretch_{augment_id}.wav"), stretched_waveform, sr)
    print(f"Saved: {os.path.join(save_dir, f"{os.path.basename(file_path).replace('.wav', '')}_stretch_{augment_id}.wav")}")

# Iterate through original data and create augmentations
for label in ["cry", "not_cry"]:
    segmented_files = os.listdir(os.path.join(SEGMENTED_DATA_DIR, label))
    for i, file in enumerate(segmented_files):
        augment_and_save(os.path.join(SEGMENTED_DATA_DIR, label, file), label, i)

print("Augmentation & Saving Completed!")
