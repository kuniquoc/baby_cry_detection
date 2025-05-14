import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from models.schemas import PredictionResult, AudioAnalysisResult, WebSocketMessage
from models.model_manager import ModelManager
from websocket.connection_manager import ConnectionManager
from services.cry_detection_service import CryDetectionService
from core.detection_core import CryDetectionCore
from utils.audio_utils import load_audio_file, decode_base64_audio, save_audio_segment
from utils.error_handling import (
    AudioProcessingError, 
    AudioFormatError,
    handle_audio_error,
    format_error_response
)

# Initialize FastAPI app
app = FastAPI(
    title="Baby Cry Detection API",
    description="API for detecting baby cry sounds in audio files",
    version="1.0.0",
)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize services and managers
model_manager = ModelManager()
connection_manager = ConnectionManager()
cry_detection_service = CryDetectionService(connection_manager)
detection_core = CryDetectionCore(model_manager)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        model_manager.load_model()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_model=PredictionResult)
async def predict(audio: UploadFile = File(...)):
    """Make a prediction on a 3-second audio file."""
    if not audio.filename.lower().endswith(('.wav')):
        raise AudioFormatError("Only WAV files are supported")
    
    try:
        # Load and process audio
        audio_content = await audio.read()
        audio_data, sr = load_audio_file(audio_content)
        
        # Get prediction using detection core
        predicted_class, confidence = detection_core.process_segment(audio_data, sr)
        
        return {"predicted_class": predicted_class, "confidence": confidence}
    
    except AudioProcessingError as e:
        raise handle_audio_error(e)
    except Exception as e:
        logger.error(f"Unexpected error in predict: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/", response_model=AudioAnalysisResult)
async def analyze_audio(audio: UploadFile = File(...)):
    """Analyze a longer audio file by splitting it into segments."""
    if not audio.filename.lower().endswith(('.wav')):
        raise AudioFormatError("Only WAV files are supported")
    
    try:
        # Load audio file
        audio_content = await audio.read()
        audio_data, sr = load_audio_file(audio_content)
        audio_length = len(audio_data) / sr
        
        # Process in segments
        segment_length = 3  # seconds
        hop_length = 1     # seconds
        segments = []
        
        for start in range(0, len(audio_data), int(hop_length * sr)):
            end = start + int(segment_length * sr)
            if end > len(audio_data):
                end = len(audio_data)
            
            segment = audio_data[start:end]
            if len(segment) < sr * 0.5:  # Skip segments shorter than 0.5s
                continue
                
            # Process segment
            predicted_class, confidence = detection_core.process_segment(segment, sr)
            
            segments.append({
                "segment_index": len(segments),
                "start_time": start / sr,
                "end_time": end / sr,
                "predicted_class": predicted_class,
                "confidence": confidence
            })
            
            if end == len(audio_data):
                break
        
        # Analyze segments for patterns
        consecutive_info = detection_core.analyze_segments(segments)
        
        # Calculate summary statistics
        cry_segments = sum(1 for seg in segments if seg["predicted_class"] == "cry")
        
        summary = {
            "total_segments": len(segments),
            "cry_segments": cry_segments,
            "not_cry_segments": len(segments) - cry_segments,
            "cry_percentage": cry_segments / len(segments) if segments else 0,
            "audio_length": audio_length,
            "has_consecutive_cry": consecutive_info["detected"]
        }
        
        return {
            "filename": audio.filename,
            "segments": segments,
            "consecutive_cry_info": consecutive_info,
            "summary": summary
        }
        
    except AudioProcessingError as e:
        raise handle_audio_error(e)
    except Exception as e:
        logger.error(f"Unexpected error in analyze_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time audio processing."""
    logger.info(f"New WebSocket connection request from client {client_id}")
    
    try:
        await connection_manager.connect(websocket, client_id)
        cry_detection_service.init_device_tracking(client_id)
        
        while True:
            try:
                # Receive and validate audio data
                data = await websocket.receive_json()
                message = WebSocketMessage(**data)
                
                try:
                    # Decode and process audio
                    audio_data, sr = decode_base64_audio(
                        message.audio_data,
                        message.channels,
                        message.sample_rate
                    )
                    
                    # Get prediction
                    predicted_class, confidence = detection_core.process_segment(audio_data, sr)
                    
                    # Send prediction result
                    await connection_manager.send_message(client_id, {
                        "type": "prediction",
                        "timestamp": message.timestamp,
                        "predicted_class": predicted_class,
                        "confidence": confidence
                    })
                    
                    # Check if we should trigger cry detection processing
                    if detection_core.should_trigger_notification(
                        predicted_class, 
                        confidence,
                        cry_detection_service.cry_status_tracker.get(client_id, {}).get('last_cry_time')
                    ):
                        await cry_detection_service.process_cry_detection(
                            client_id=client_id,
                            timestamp=message.timestamp,
                            confidence=confidence
                        )
                    
                except AudioProcessingError as e:
                    await connection_manager.send_message(
                        client_id, 
                        format_error_response("Error processing audio", e.details)
                    )
                    
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")
                break
            except Exception as e:
                logger.error(f"Error in websocket loop: {str(e)}")
                await connection_manager.send_message(
                    client_id,
                    format_error_response("Internal server error", {"error": str(e)})
                )
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
    finally:
        cry_detection_service.cleanup_device_tracking(client_id)
        connection_manager.disconnect(client_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model_manager.model is None:
        return {"status": "warning", "message": "Model not loaded"}
    return {"status": "ok", "message": "Service is healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
