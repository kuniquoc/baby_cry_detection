import numpy as np
import librosa

def apply_vad(audio, sr, min_freq=200, max_freq=600, frame_length=0.025, hop_length=0.01):
    """Apply voice activity detection to detect baby crying.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        min_freq: Minimum frequency for baby cry (default: 200 Hz)
        max_freq: Maximum frequency for baby cry (default: 600 Hz)
        frame_length: Frame length in seconds
        hop_length: Hop length in seconds
        
    Returns:
        tuple: (voiced flag, is crying boolean, mean f0)
    """
    # Compute fundamental frequency (f0)
    f0, voiced_flag, _ = librosa.pyin(audio, fmin=min_freq, fmax=max_freq, sr=sr, frame_length=int(frame_length * sr), hop_length=int(hop_length * sr))
    
    # Create mask based on f0
    valid_f0 = ~np.isnan(f0)
    
    # Calculate mean f0 (excluding NaN values)
    f0_clean = f0[~np.isnan(f0)]
    mean_f0 = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
    
    # Calculate percentage of frames with crying
    total_frames = len(f0)
    crying_frames = np.sum(valid_f0 | voiced_flag)
    crying_percentage = (crying_frames / total_frames) * 100 if total_frames > 0 else 0
    
    # Determine if baby crying is detected (any presence of crying)
    isPassed = crying_percentage > 50
    
    return audio, isPassed, mean_f0


def extract_mfcc(audio, sr=16000, n_mfcc=40, n_fft=512, hop_length=160, duration=3.0):
    """Extract MFCC features from audio signal with consistent length.
    
    Args:
        audio: Input audio signal
        sr: Sampling rate
        n_mfcc: Number of MFCC coefficients
        n_fft: Length of FFT window
        hop_length: Number of samples between successive frames
        duration: Target duration in seconds (default: 3.0)
    
    Returns:
        numpy.ndarray: MFCC features with consistent length
    """
    # 1. Remove DC offset
    audio = audio - np.mean(audio)
    
    # 2. Apply pre-emphasis filter to enhance high frequencies
    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # 3. Normalize audio signal to have max amplitude of 1.0
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # 4. Simple noise reduction by applying a threshold-based gate
    noise_threshold = 0.01
    audio = np.where(np.abs(audio) < noise_threshold, 0, audio)
    
    # 5. Trim silence from the beginning and end
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
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
    
    # Calculate target number of frames for the specified duration
    target_frames = int((duration * sr - n_fft + hop_length) / hop_length)
    
    # Ensure consistent output length (3 seconds)
    if mfccs.shape[1] < target_frames:
        # If shorter than 3 seconds, pad with zeros
        padding = np.zeros((n_mfcc, target_frames - mfccs.shape[1]))
        mfccs = np.hstack((mfccs, padding))
    elif mfccs.shape[1] > target_frames:
        # If longer than 3 seconds, truncate
        mfccs = mfccs[:, :target_frames]
    
    return mfccs


