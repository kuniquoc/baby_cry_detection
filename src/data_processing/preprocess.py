import numpy as np
import librosa

def apply_vad(audio, sr=16000, frame_length=400, hop_length=160):
    """Advanced Voice Activity Detection optimized for children's crying using f0.

    Args:
        audio: Input audio signal
        sr: Sampling rate (default: 16kHz)
        frame_length: Length of each frame
        hop_length: Number of samples between frames

    Returns:
        numpy.ndarray: Original audio signal
        bool: Whether potential crying is detected
        float: Mean f0 of detected segments
    """
    # Compute fundamental frequency (f0)
    f0, voiced_flag, _ = librosa.pyin(audio, fmin=200, fmax=600, sr=sr, frame_length=frame_length, hop_length=hop_length)
    
    # Create mask based on f0
    valid_f0 = ~np.isnan(f0)
    
    # Calculate mean f0 (excluding NaN values)
    f0_clean = f0[~np.isnan(f0)]
    mean_f0 = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
    
    # Calculate percentage of frames with crying
    total_frames = len(f0)
    crying_frames = np.sum(valid_f0 | voiced_flag)
    crying_percentage = (crying_frames / total_frames) * 100 if total_frames > 0 else 0
    
    # Determine if baby crying is detected (>50% of frames)
    isBabyCrying = crying_percentage > 50
    
    return audio, isBabyCrying, mean_f0


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

