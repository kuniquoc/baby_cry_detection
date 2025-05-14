from typing import Dict, Any, Optional
from fastapi import HTTPException, status

class AudioProcessingError(Exception):
    """Base exception for audio processing errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class AudioFormatError(AudioProcessingError):
    """Exception for invalid audio format"""
    pass

class ModelPredictionError(AudioProcessingError):
    """Exception for model prediction errors"""
    pass

class FirebaseError(Exception):
    """Base exception for Firebase related errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

def handle_audio_error(error: AudioProcessingError) -> HTTPException:
    """Convert audio processing errors to HTTPException"""
    status_code = status.HTTP_400_BAD_REQUEST
    
    if isinstance(error, AudioFormatError):
        status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    elif isinstance(error, ModelPredictionError):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return HTTPException(
        status_code=status_code,
        detail={
            "message": str(error),
            "details": error.details,
            "error_type": error.__class__.__name__
        }
    )

def handle_firebase_error(error: FirebaseError) -> HTTPException:
    """Convert Firebase errors to HTTPException"""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "message": str(error),
            "details": error.details,
            "error_type": "FirebaseError"
        }
    )

def format_error_response(message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Format error response for websocket messages"""
    return {
        "type": "error",
        "message": message,
        "details": details or {}
    }