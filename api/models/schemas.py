from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float

class SegmentPrediction(BaseModel):
    segment_index: int
    start_time: float
    end_time: float
    predicted_class: str
    confidence: float

class ConsecutiveCryInfo(BaseModel):
    detected: bool
    segments: List[int] = []
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class AudioAnalysisResult(BaseModel):
    filename: str
    segments: List[SegmentPrediction]
    consecutive_cry_info: ConsecutiveCryInfo
    summary: Dict[str, Any]

class WebSocketMessage(BaseModel):
    timestamp: float
    sample_rate: int = 16000
    channels: int = 1
    audio_data: str  # Base64 encoded WAV data