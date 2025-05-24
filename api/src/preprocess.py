import numpy as np
import librosa

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
