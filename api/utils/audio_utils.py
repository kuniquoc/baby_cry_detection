import io
import wave
import logging
import numpy as np
import librosa
from typing import Tuple, Optional
import soundfile as sf
import base64

logger = logging.getLogger(__name__)

def load_audio_file(file_content: bytes, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio file from bytes content"""
    try:
        audio_io = io.BytesIO(file_content)
        audio_data, sr = librosa.load(audio_io, sr=target_sr)
        return audio_data, sr
    except Exception as e:
        logger.error(f"Error loading audio file: {str(e)}")
        raise ValueError(f"Invalid audio file format: {str(e)}")

def decode_base64_audio(audio_data: str, expected_channels: int, expected_sr: int) -> Tuple[np.ndarray, int]:
    """Decode base64 encoded audio data and validate format"""
    try:
        audio_data_bytes = base64.b64decode(audio_data)
        with wave.open(io.BytesIO(audio_data_bytes), 'rb') as wf:
            # Validate audio format
            if wf.getnchannels() != expected_channels:
                raise ValueError(f"Expected {expected_channels} channels, got {wf.getnchannels()}")
            if wf.getframerate() != expected_sr:
                raise ValueError(f"Expected {expected_sr}Hz sample rate, got {wf.getframerate()}Hz")
            
            # Read audio data
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            
            # Convert to float32
            audio = audio.astype(np.float32) / 32768.0
            
            return audio, wf.getframerate()
            
    except Exception as e:
        logger.error(f"Error decoding audio data: {str(e)}")
        raise ValueError(f"Invalid audio data: {str(e)}")

def save_audio_segment(audio_data: np.ndarray, sr: int, filepath: str) -> None:
    """Save audio segment to file"""
    try:
        sf.write(filepath, audio_data, sr)
    except Exception as e:
        logger.error(f"Error saving audio segment: {str(e)}")
        raise IOError(f"Failed to save audio file: {str(e)}")

def validate_audio_length(audio_length: float, min_length: float = 0.5, max_length: float = 6.0) -> bool:
    """Validate audio length is within acceptable range"""
    if audio_length < min_length or audio_length > max_length:
        logger.warning(
            f"Audio length ({audio_length:.2f}s) is not optimal. "
            f"Expected between {min_length}s and {max_length}s"
        )
        return False
    return True