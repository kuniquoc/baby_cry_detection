import os
import sys
from pathlib import Path
import uvicorn
import tempfile
import numpy as np
import io
import librosa
import soundfile as sf
import torch
import requests
import base64
import wave
import aiohttp
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from project structure
from src.models.cnn_model import MobileNetV2_Crying
from src.utils.dataset_loader import DatasetLoader
from src.data_processing.preprocess import extract_mfcc
from src.data_processing.split_audio import split_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="Baby Cry Detection API",
    description="API for detecting baby cry sounds in audio files",
    version="1.0.0",
)

# Set up templates and static files
templates = Jinja2Templates(directory="api/templates")
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Default model path
DEFAULT_MODEL_PATH = "runs/20250406_182137/checkpoints/last_model.pth"

# Define response model for prediction results
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

# Global variables for model and device
model = None
device = None
label_encoder = None

def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load the trained model from checkpoint"""
    global model, device, label_encoder
    
    if model is not None:
        return model, device
    
    try:
        # Initialize DatasetLoader to get parameters
        loader = DatasetLoader()
        label_encoder = loader.label_encoder
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize model with same parameters as during training
        model = MobileNetV2_Crying().to(device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print model information
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
        logger.info(f"Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
        
        model.eval()
        return model, device
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def predict_on_audio(audio_data, sr):
    """Make prediction on audio data"""
    global model, device, label_encoder
    
    if model is None:
        model, device = load_model()
    
    # Initialize DatasetLoader to get same parameters as training
    loader = DatasetLoader()
    target_sr = loader.sample_rate
    n_mels = loader.n_mels
    n_fft = loader.n_fft
    hop_length = loader.hop_length
    
    # Resample if necessary
    if sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Extract MFCC features
    mfccs = extract_mfcc(
        audio_data,
        sr=sr,
        n_mfcc=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Convert to tensor
    features = torch.from_numpy(mfccs).float().unsqueeze(0)
    
    # Add channel dimension if needed
    if len(features.shape) == 3:  # If [batch, n_mels, time]
        features = features.unsqueeze(1)  # Make it [batch, channel, n_mels, time]
    
    features = features.to(device)
    
    # Make prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        logits = model(features) 
        probability = torch.sigmoid(logits)  # Apply sigmoid to convert logits to probability (0-1)
        
        predicted_key = (probability > 0.5).int().item()
        confidence = probability.item() if predicted_key == 1 else 1 - probability.item()
    
    predicted_class = label_encoder.inverse_transform([predicted_key])[0]
    
    return predicted_class, confidence

@app.on_event("startup")
async def startup_db_client():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        # We'll continue anyway and try to load model when needed

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_model=PredictionResult)
async def predict(audio: UploadFile = File(...)):
    """
    Make a prediction on a 3-second audio file.
    
    - **audio**: WAV audio file (approximately 3 seconds in length)
    
    Returns:
    - **predicted_class**: The predicted class (cry or not_cry)
    - **confidence**: Confidence score (0-1)
    """
    if not audio.filename.lower().endswith(('.wav')):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    try:
        # Read audio file
        audio_content = await audio.read()
        audio_io = io.BytesIO(audio_content)
        
        # Load audio using librosa
        audio_data, sr = librosa.load(audio_io, sr=None)
        
        # Check audio length - should be around 3 seconds
        audio_length = len(audio_data) / sr
        if audio_length < 0.5 or audio_length > 6:
            logger.warning(f"Audio length ({audio_length:.2f}s) is not optimal. Ideal length is 3 seconds.")
        
        # Get prediction
        predicted_class, confidence = predict_on_audio(audio_data, sr)
        
        return {"predicted_class": predicted_class, "confidence": confidence}
    
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/predict_with_timestamp")
async def predict_with_timestamp(payload: Dict[str, Any]):
    """
    Predict if the audio contains baby crying and send a notification if confidence > 80%.
    
    - **payload**: JSON payload containing:
        - **timestamp**: Timestamp associated with the audio
        - **sample_rate**: Sample rate of the audio
        - **channels**: Number of audio channels
        - **audio_data**: Base64-encoded WAV audio data
    
    Returns:
    - **predicted_class**: The predicted class (cry or not_cry)
    - **confidence**: Confidence score (0-1)
    """
    try:
        # Extract and decode audio data from payload
        timestamp = payload.get("timestamp")
        sample_rate = payload.get("sample_rate") 
        if sample_rate is None: 
            sample_rate = 16000
        channels = payload.get("channels")
        if channels is None:
            channels = 1
        audio_data_base64 = payload.get("audio_data")
        
        if not all([timestamp, audio_data_base64]):
            raise HTTPException(status_code=400, detail="Missing required fields in payload")
        
        # Decode base64 audio data
        audio_data_bytes = base64.b64decode(audio_data_base64)
        with wave.open(io.BytesIO(audio_data_bytes), 'rb') as wf:
            if wf.getnchannels() != channels or wf.getframerate() != sample_rate:
                raise HTTPException(status_code=400, detail="Audio metadata mismatch")
            audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        
        # Get prediction
        predicted_class, confidence = predict_on_audio(audio_data, sample_rate)

        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
        
        # If prediction is "cry" with confidence > 80%, send notification
        if predicted_class == "cry" and confidence > 0.8:
            # Save audio file
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

            notification_payload = {"timestamp": timestamp, "probability": confidence, "audio_file": audio_filename}
            try:
                response = requests.post("http://localhost:3000/api/notifications", json=notification_payload)
                response.raise_for_status()
                logger.info(f"Notification sent successfully: {notification_payload}")
            except requests.RequestException as e:
                logger.error(f"Failed to send notification: {str(e)}")
        
        return {"predicted_class": predicted_class, "confidence": confidence}
    
    except Exception as e:
        logger.error(f"Error processing prediction with timestamp: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/analyze/", response_model=AudioAnalysisResult)
async def analyze_audio(audio: UploadFile = File(...)):
    """
    Analyze a longer audio file by splitting it into segments and making predictions.
    
    - **audio**: WAV audio file (can be any length)
    
    Returns:
    - **filename**: Original filename
    - **segments**: List of analyzed segments with predictions
    - **consecutive_cry_info**: Information about consecutive cry segments with high confidence
    - **summary**: Summary of analysis results
    """
    if not audio.filename.lower().endswith(('.wav')):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    try:
        # Read audio file
        audio_content = await audio.read()
        audio_io = io.BytesIO(audio_content)
        
        # Load audio using librosa
        audio_data, sr = librosa.load(audio_io, sr=16000)  # Load at 16kHz
        audio_length = len(audio_data) / sr
        
        # Create temporary file path for split_audio function
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filepath = temp_file.name
            sf.write(temp_filepath, audio_data, sr)
        
        try:
            # Split audio into 3-second segments with 1-second hop length
            segments, _ = split_audio(temp_filepath, segment_length=3, hop_length=1)
            
            # Process each segment for prediction
            segment_predictions = []
            for idx, (segment_audio, name, segment_sr) in enumerate(segments):
                # Calculate start and end times
                start_time = idx * 1  # 1-second hop length
                end_time = min(start_time + 3, audio_length)  # 3-second segments or shorter at the end
                
                # Get prediction for segment
                predicted_class, confidence = predict_on_audio(segment_audio, segment_sr)
                
                # Add prediction to segment info
                segment_predictions.append({
                    "segment_index": idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "predicted_class": predicted_class,
                    "confidence": confidence
                })
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
        
        # If no segments were found, create a single "not_cry" segment for the whole audio
        if not segment_predictions:
            segment_predictions = [{
                "segment_index": 0,
                "start_time": 0,
                "end_time": audio_length,
                "predicted_class": "not_cry",
                "confidence": 1.0
            }]
            
        # Detect consecutive cry segments with high confidence
        consecutive_cry_info = detect_consecutive_cry_segments(segment_predictions)
        
        # Calculate summary statistics
        cry_segments = sum(1 for seg in segment_predictions if seg["predicted_class"] == "cry")
        not_cry_segments = len(segment_predictions) - cry_segments
        
        summary = {
            "total_segments": len(segment_predictions),
            "cry_segments": cry_segments,
            "not_cry_segments": not_cry_segments,
            "cry_percentage": cry_segments / len(segment_predictions) if segment_predictions else 0,
            "audio_length": audio_length,
            "has_consecutive_cry": consecutive_cry_info["detected"]
        }
        
        return {
            "filename": audio.filename,
            "segments": segment_predictions,
            "consecutive_cry_info": consecutive_cry_info,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing audio: {str(e)}")

def detect_consecutive_cry_segments(segments):
    """
    Detect if there are two or more consecutive segments with 'cry' prediction 
    and confidence > 0.8
    
    Returns:
    - Dictionary with consecutive cry information
    """
    consecutive_info = {
        "detected": False,
        "segments": [],
        "start_time": None,
        "end_time": None
    }
    
    if len(segments) < 2:
        return consecutive_info
    
    # Find consecutive cry segments with high confidence
    for i in range(len(segments) - 1):
        current_seg = segments[i]
        next_seg = segments[i + 1]
        
        if (current_seg["predicted_class"] == "cry" and 
            next_seg["predicted_class"] == "cry" and
            current_seg["confidence"] > 0.8 and 
            next_seg["confidence"] > 0.8):
            
            consecutive_info["detected"] = True
            if not consecutive_info["segments"]:
                consecutive_info["segments"] = [current_seg["segment_index"], next_seg["segment_index"]]
                consecutive_info["start_time"] = current_seg["start_time"]
                consecutive_info["end_time"] = next_seg["end_time"]
            elif next_seg["segment_index"] not in consecutive_info["segments"]:
                consecutive_info["segments"].append(next_seg["segment_index"])
                consecutive_info["end_time"] = next_seg["end_time"]
    
    return consecutive_info

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_message(self, client_id: str, message: Dict[str, Any]):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time audio processing.
    Expects audio chunks of exactly 3 seconds in the following format:
    {
        "timestamp": float,
        "sample_rate": int,
        "channels": int,
        "audio_data": string (base64 encoded WAV, must be exactly 3 seconds)
    }
    """
    logger.info(f"New WebSocket connection request from client {client_id}")
    try:
        await manager.connect(websocket, client_id)
        
        while True:
            try:
                # Receive audio data
                data = await websocket.receive_json()
                
                # Extract and validate data
                timestamp = data.get("timestamp")
                sample_rate = data.get("sample_rate", 16000)
                channels = data.get("channels", 1)
                audio_data_base64 = data.get("audio_data")
                
                if not all([timestamp, audio_data_base64]):
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": "Missing required fields in payload",
                        "details": {
                            "timestamp": "missing" if not timestamp else "ok",
                            "audio_data": "missing" if not audio_data_base64 else "ok"
                        }
                    })
                    continue
                
                try:
                    # Decode audio data
                    try:
                        audio_data_bytes = base64.b64decode(audio_data_base64)
                    except Exception as e:
                        await manager.send_message(client_id, {
                            "type": "error",
                            "message": "Invalid base64 audio data",
                            "details": str(e)
                        })
                        continue

                    with wave.open(io.BytesIO(audio_data_bytes), 'rb') as wf:
                        if wf.getnchannels() != channels or wf.getframerate() != sample_rate:
                            await manager.send_message(client_id, {
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
                            
                        # Verify audio length (should be exactly 3 seconds)
                        audio_length = wf.getnframes() / wf.getframerate()
                        if not (2.9 <= audio_length <= 3.1):  # Allow small tolerance
                            await manager.send_message(client_id, {
                                "type": "error",
                                "message": f"Audio length must be 3 seconds",
                                "details": {
                                    "expected_length": 3.0,
                                    "actual_length": audio_length
                                }
                            })
                            continue
                            
                        try:
                            audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                        except Exception as e:
                            await manager.send_message(client_id, {
                                "type": "error",
                                "message": "Failed to read audio frames",
                                "details": str(e)
                            })
                            continue

                    # Get prediction
                    try:
                        predicted_class, confidence = predict_on_audio(audio_data, sample_rate)
                    except Exception as e:
                        logger.error(f"Prediction error: {str(e)}")
                        await manager.send_message(client_id, {
                            "type": "error",
                            "message": "Prediction failed",
                            "details": str(e)
                        })
                        continue
                    
                    # Send prediction result
                    await manager.send_message(client_id, {
                        "type": "prediction",
                        "timestamp": timestamp,
                        "predicted_class": predicted_class,
                        "confidence": confidence
                    })
                    
                    # If crying detected with high confidence, save audio and send notification
                    if predicted_class == "cry" and confidence > 0.8:
                        # Save audio file
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

                        # Send alert through websocket
                        await manager.send_message(client_id, {
                            "type": "alert",
                            "timestamp": timestamp,
                            "message": "Baby crying detected!",
                            "confidence": confidence,
                            "audio_file": audio_filename
                        })
                        
                        # Send to notification API
                        try:
                            notification_payload = {
                                "timestamp": timestamp,
                                "probability": confidence,
                                "audio_file": audio_filename
                            }
                            async with aiohttp.ClientSession() as session:
                                async with session.post(
                                    "http://localhost:3000/api/notifications",
                                    json=notification_payload,
                                    timeout=5
                                ) as response:
                                    if response.status != 200:
                                        logger.error(f"Notification API error: {response.status}")
                                        
                            logger.info(f"Notification sent successfully: {notification_payload}")
                        except Exception as e:
                            logger.error(f"Failed to send notification: {str(e)}")
                
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {str(e)}")
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": "Error processing audio",
                        "details": str(e)
                    })
                    
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {str(e)}")
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "details": str(e)
                })
            except Exception as e:
                logger.error(f"Error in websocket loop: {str(e)}")
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": "Internal server error",
                    "details": str(e)
                })
                break
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected during handshake")
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
    finally:
        manager.disconnect(client_id)  # Use ConnectionManager to handle disconnection

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "warning", "message": "Model not loaded yet"}
    return {"status": "ok", "message": "Service is running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
