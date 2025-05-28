import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
import sys
import json
import io
import wave
import base64
import numpy as np
from datetime import datetime

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
        await cry_detection_service.start_periodic_check()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up services on shutdown"""
    try:
        # Send final no-cry events before stopping the service
        await cry_detection_service.send_final_no_cry_events()
        # Stop the periodic check
        await cry_detection_service.stop_periodic_check()
        logger.info("Services cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during service cleanup: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_model=PredictionResult)
async def predict(audio: UploadFile = File(...)):
    """Make a prediction on a 3-second audio file."""
    start_time = datetime.now()
    
    if not audio.filename.lower().endswith(('.wav')):
        raise AudioFormatError("Only WAV files are supported")
    
    try:
        # Load and process audio
        audio_content = await audio.read()
        audio_data, sr = load_audio_file(audio_content)
        
        # Get prediction using detection core
        predicted_class, confidence = detection_core.process_segment(audio_data, sr)
        
        result = {"predicted_class": predicted_class, "confidence": confidence}
        
        # Log response time
        response_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"API /predict response time: {response_time:.3f}s")
        
        return result
    
    except AudioProcessingError as e:
        raise handle_audio_error(e)
    except Exception as e:
        logger.error(f"Unexpected error in predict: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/", response_model=AudioAnalysisResult)
async def analyze_audio(audio: UploadFile = File(...)):
    """Analyze a longer audio file by splitting it into segments."""
    start_time = datetime.now()
    
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
        
        result = {
            "filename": audio.filename,
            "segments": segments,
            "consecutive_cry_info": consecutive_info,
            "summary": summary
        }
        
        # Log response time
        response_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"API /analyze response time: {response_time:.3f}s for file {audio.filename} (length: {audio_length:.1f}s)")
        
        return result
        
    except AudioProcessingError as e:
        raise handle_audio_error(e)
    except Exception as e:
        logger.error(f"Unexpected error in analyze_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time audio processing with cry detection.
    
    Parameters:
    ----------
    websocket : WebSocket
        The WebSocket connection object
    client_id : str
        Unique identifier for the client connection, also used as deviceId for Firebase
        
    Expected Message Format:
    ---------------------
    JSON object containing:
    {
        "timestamp": float (required)
            Unix timestamp in seconds
        "sample_rate": int (optional, default=16000)
            Audio sample rate in Hz
        "channels": int (optional, default=1)
            Number of audio channels
        "audio_data": str (required)
            Base64-encoded WAV audio data
    }
    
    Sent Message Types:
    -----------------
    1. Prediction Result:
    {
        "type": "prediction",
        "timestamp": float,
        "predicted_class": str,
        "confidence": float
    }
    
    2. Alert (on cry detection):
    {
        "type": "alert",
        "timestamp": float,
        "message": str,
        "confidence": float,
        "deviceId": str
    }
    
    3. Error:
    {
        "type": "error",
        "message": str,
        "details": dict (optional)
    }
    
    Notes:
    -----
    - Connection remains open for continuous real-time processing
    - Audio chunks should be exactly 3 seconds in length
    - Cry detection triggers when confidence > 0.8
    - Firebase events and notifications are processed according to the history
    - No-cry event is triggered if no crying is detected for 10 seconds after a cry
    """
    logger.info(f"New WebSocket connection request from client {client_id}")
    
    try:
        # Initialize connection and tracking
        await connection_manager.connect(websocket, client_id)
        cry_detection_service.init_device_tracking(client_id)
        
        while True:
            try:
                # Receive and validate message
                data = await websocket.receive_json()
                timestamp = data.get("timestamp")
                sample_rate = data.get("sample_rate", 16000)
                channels = data.get("channels", 1)
                audio_data_base64 = data.get("audio_data")
                
                # Validate required fields
                if not all([timestamp, audio_data_base64]):
                    await connection_manager.send_message(client_id, {
                        "type": "error",
                        "message": "Missing required fields in payload",
                        "details": {
                            "timestamp": "missing" if not timestamp else "ok",
                            "audio_data": "missing" if not audio_data_base64 else "ok"
                        }
                    })
                    continue

                try:
                    # Decode and validate audio data
                    try:
                        audio_data_bytes = base64.b64decode(audio_data_base64)
                    except Exception as e:
                        await connection_manager.send_message(client_id, {
                            "type": "error",
                            "message": "Invalid base64 audio data",
                            "details": str(e)
                        })
                        continue

                    # Validate audio metadata
                    with wave.open(io.BytesIO(audio_data_bytes), 'rb') as wf:
                        if wf.getnchannels() != channels or wf.getframerate() != sample_rate:
                            await connection_manager.send_message(client_id, {
                                "type": "error",
                                "message": "Audio metadata mismatch",
                                "details": {
                                    "expected_channels": channels,
                                    "actual_channels": wf.getnchannels(),
                                    "expected_sample_rate": sample_rate,
                                    "actual_sample_rate": wf.getframerate()
                                }
                            })
                            continue
                            
                        # Verify audio length
                        audio_length = wf.getnframes() / wf.getframerate()
                        if not (2.9 <= audio_length <= 3.1):
                            await connection_manager.send_message(client_id, {
                                "type": "error",
                                "message": "Audio length must be 3 seconds",
                                "details": {
                                    "expected_length": 3.0,
                                    "actual_length": audio_length
                                }
                            })
                            continue
                            
                        try:
                            audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                        except Exception as e:
                            await connection_manager.send_message(client_id, {
                                "type": "error",
                                "message": "Failed to read audio frames",
                                "details": str(e)
                            })
                            continue

                    # Get prediction
                    try:
                        predicted_class, confidence = detection_core.process_segment(audio_data, sample_rate)
                    except Exception as e:
                        await connection_manager.send_message(client_id, {
                            "type": "error",
                            "message": "Prediction failed",
                            "details": str(e)
                        })
                        continue

                    # Send prediction result
                    await connection_manager.send_message(client_id, {
                        "type": "prediction",
                        "timestamp": timestamp,
                        "predicted_class": predicted_class,
                        "confidence": confidence
                    })

                    # Process high-confidence cry detection
                    if predicted_class == "cry" and confidence > 0.8:
                        # Save audio file for record
                        try:
                            save_dir = os.path.join("api", "data", "cry_detections")
                            os.makedirs(save_dir, exist_ok=True)
                            timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
                            audio_filename = f"cry_detected_{timestamp_str}.wav"
                            audio_filepath = os.path.join(save_dir, audio_filename)
                            
                            with open(audio_filepath, 'wb') as f:
                                f.write(audio_data_bytes)
                            logger.info(f"Saved cry detection audio to: {audio_filepath}")
                        except Exception as e:
                            logger.error(f"Failed to save audio file: {str(e)}")

                        # Process cry detection through service
                        await cry_detection_service.process_cry_detection(
                            client_id=client_id,
                            timestamp=timestamp,
                            confidence=confidence
                        )

                except Exception as e:
                    logger.error(f"Error processing audio chunk: {str(e)}")
                    await connection_manager.send_message(client_id, {
                        "type": "error",
                        "message": "Internal processing error",
                        "details": str(e)
                    })

            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {str(e)}")
                await connection_manager.send_message(client_id, {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "details": str(e)
                })
            except Exception as e:
                logger.error(f"Error in websocket loop: {str(e)}")
                await connection_manager.send_message(client_id, {
                    "type": "error",
                    "message": "Internal server error",
                    "details": str(e)
                })
                break

    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
    finally:
        # Cleanup resources
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
