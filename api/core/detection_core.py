import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_manager import ModelManager
from utils.error_handling import ModelPredictionError, AudioProcessingError
from utils.audio_utils import validate_audio_length

logger = logging.getLogger(__name__)

class CryDetectionCore:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.confidence_threshold = 0.8
        self.min_consecutive_segments = 2  # Minimum number of consecutive cry segments to confirm crying

    def process_segment(self, audio_data: np.ndarray, sr: int) -> Tuple[str, float]:
        """Process a single audio segment for cry detection"""
        try:
            # Validate audio length
            audio_length = len(audio_data) / sr
            validate_audio_length(audio_length)
            
            # Get prediction
            predicted_class, confidence = self.model_manager.predict(audio_data, sr)
            
            return predicted_class, confidence
            
        except Exception as e:
            logger.error(f"Error in process_segment: {str(e)}")
            raise ModelPredictionError("Failed to process audio segment", {"error": str(e)})

    def analyze_segments(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a list of segment predictions to detect crying patterns"""
        try:
            consecutive_info = {
                "detected": False,
                "segments": [],
                "start_time": None,
                "end_time": None,
                "max_confidence": 0.0,
                "avg_confidence": 0.0
            }
            
            if len(segments) < self.min_consecutive_segments:
                return consecutive_info
            
            cry_segments = []
            confidences = []
            
            # Find segments with high confidence cry predictions
            for segment in segments:
                if (segment["predicted_class"] == "cry" and 
                    segment["confidence"] > self.confidence_threshold):
                    cry_segments.append(segment)
                    confidences.append(segment["confidence"])
            
            if not cry_segments:
                return consecutive_info
            
            # Find consecutive segments
            current_sequence = [cry_segments[0]]
            longest_sequence = current_sequence
            
            for i in range(1, len(cry_segments)):
                if cry_segments[i]["segment_index"] == cry_segments[i-1]["segment_index"] + 1:
                    current_sequence.append(cry_segments[i])
                    if len(current_sequence) > len(longest_sequence):
                        longest_sequence = current_sequence
                else:
                    current_sequence = [cry_segments[i]]
            
            # Update consecutive info if we found a valid sequence
            if len(longest_sequence) >= self.min_consecutive_segments:
                consecutive_info.update({
                    "detected": True,
                    "segments": [seg["segment_index"] for seg in longest_sequence],
                    "start_time": longest_sequence[0]["start_time"],
                    "end_time": longest_sequence[-1]["end_time"],
                    "max_confidence": max(seg["confidence"] for seg in longest_sequence),
                    "avg_confidence": sum(seg["confidence"] for seg in longest_sequence) / len(longest_sequence)
                })
            
            return consecutive_info
            
        except Exception as e:
            logger.error(f"Error in analyze_segments: {str(e)}")
            raise AudioProcessingError("Failed to analyze segments", {"error": str(e)})

    def should_trigger_notification(self, 
                                 prediction: str, 
                                 confidence: float) -> bool:
        """Determine if a notification should be triggered based on prediction results"""
        
        # Basic confidence threshold check
        if prediction != "cry" or confidence < self.confidence_threshold:
            return False
        
        return True