import os
import sys
import uvicorn
import tempfile
import numpy as np
import io
import librosa
import soundfile as sf
import torch
import base64
import wave
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import time
import threading

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from project structure
from src.models.cnn_model import MobileNetV2_Crying
from src.utils.dataset_loader import DatasetLoader
from src.data_processing.preprocess import extract_mfcc
from src.data_processing.split_audio import split_audio
from firebase_service import send_cry_notification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global variables to track cry status for each device
# Format: {device_id: {'last_cry_time': timestamp, 'checking_no_cry': bool, 'no_cry_timer': threading.Timer}}
cry_status_tracker = {}
NO_CRY_CHECK_SECONDS = 10  # Time window to check for no-crying event

class CryDetection:
    def __init__(self, timestamp: float, probability: float, audio_file: Optional[str] = None):
        self.timestamp = timestamp
        self.probability = probability
        self.audio_file = audio_file


async def check_firebase_events_and_notify(timestamp, device_id, confidence=0.8, audio_filename=None):
    """
    Check Firebase events and create notifications based on the event history
    
    Parameters:
    ----------
    timestamp : float
        Current timestamp in seconds
    device_id : str
        Device ID for Firebase
    confidence : float
        Confidence score for the current detection
    audio_filename : str, optional
        Audio file name for reference
        
    Returns:
    -------
    bool
        True if notification was processed, False otherwise
    """
    try:
        notification_data = {
            "timestamp": timestamp,
            "deviceId": device_id,
            "confidence": confidence
        }
        
        # Send to Firebase to process events and notifications
        result = await send_cry_notification(notification_data)
        if result:
            logger.info(f"Cry event created at {timestamp} for device {device_id}")
            return True
        else:
            logger.error(f"Failed to create cry event at {timestamp} for device {device_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking Firebase events: {str(e)}")
        return False

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
    
    Parameters:
    ----------
    audio : UploadFile
        A WAV audio file upload with the following requirements:
        - Format: WAV
        - Length: Approximately 3 seconds (0.5-6 seconds acceptable)
        - Sample Rate: Any (will be resampled if needed)
        - Channels: Any (will be converted if needed)
    
    Returns:
    -------
    PredictionResult:
        - predicted_class: str
            Either "cry" or "not_cry"
        - confidence: float
            Confidence score between 0 and 1
            
    Raises:
    ------
    HTTPException(400):
        If file format is not WAV
    HTTPException(500):
        If there's an error processing the audio
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



@app.post("/analyze/", response_model=AudioAnalysisResult)
async def analyze_audio(audio: UploadFile = File(...)):
    """
    Analyze a longer audio file by splitting it into segments and making predictions.
    
    Parameters:
    ----------
    audio : UploadFile
        A WAV audio file with the following requirements:
        - Format: WAV
        - Length: Any (will be split into 3-second segments)
        - Sample Rate: Any (will be resampled to 16kHz)
        - Channels: Any (will be converted if needed)
    
    Returns:
    -------
    AudioAnalysisResult:
        - filename: str
            Original filename
        - segments: List[SegmentPrediction]
            List of analyzed segments, each containing:
            * segment_index: int
            * start_time: float (seconds)
            * end_time: float (seconds)
            * predicted_class: str ("cry" or "not_cry")
            * confidence: float (0-1)
        - consecutive_cry_info: ConsecutiveCryInfo
            Information about consecutive cry segments:
            * detected: bool
            * segments: List[int] (segment indices)
            * start_time: Optional[float]
            * end_time: Optional[float]
        - summary: Dict[str, Any]
            Analysis summary including:
            * total_segments: int
            * cry_segments: int
            * not_cry_segments: int
            * cry_percentage: float
            * audio_length: float
            * has_consecutive_cry: bool
            
    Notes:
    -----
    - Audio is split into 3-second segments with 1-second overlap
    - Consecutive cry detection requires adjacent segments with:
        * Both predicted as "cry"
        * Both confidence scores > 0.8
            
    Raises:
    ------
    HTTPException(400):
        If file format is not WAV
    HTTPException(500):
        If there's an error analyzing the audio
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
        "details": str (optional)
    }
    
    Notes:
    -----
    - Connection remains open for continuous real-time processing
    - Audio chunks should be ~3 seconds in length
    - Cry detection triggers when confidence > 0.8
    - Firebase events and notifications are processed according to the history
    - No-cry event is triggered if no crying is detected for 10 seconds after a cry
    
    Error Handling:
    -------------
    - Missing fields: Sends error message, continues processing
    - Audio metadata mismatch: Sends error message, continues processing
    - Processing errors: Sends error message, continues processing
    - Connection errors: Logs error, closes connection
    """
    logger.info(f"New WebSocket connection request from client {client_id}")
    
    # Initialize or reset cry status tracking for this device
    global cry_status_tracker
    cry_status_tracker[client_id] = {
        'last_cry_time': None,  # Will be set when a cry is detected
        'checking_no_cry': False,  # True when we're checking for 10s of no crying
        'no_cry_timer': None  # Timer thread object
    }
    
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
                
                # Use client_id from URL as deviceId for Firebase
                device_id = client_id
                
                if not all([timestamp, audio_data_base64]):
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": "Missing required fields in payload"
                    })
                    continue
                
                try:
                    # Decode audio data
                    audio_data_bytes = base64.b64decode(audio_data_base64)
                    with wave.open(io.BytesIO(audio_data_bytes), 'rb') as wf:
                        if wf.getnchannels() != channels or wf.getframerate() != sample_rate:
                            await manager.send_message(client_id, {
                                "type": "error",
                                "message": "Audio metadata mismatch"
                            })
                            continue
                        audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

                    # Get prediction
                    predicted_class, confidence = predict_on_audio(audio_data, sample_rate)
                    
                    # Send prediction result
                    await manager.send_message(client_id, {
                        "type": "prediction",
                        "timestamp": timestamp,
                        "predicted_class": predicted_class,
                        "confidence": confidence
                    })
                    
                    # If crying detected with high confidence > 80%, process it
                    if predicted_class == "cry" and confidence > 0.8:
                        # Update cry status tracking
                        device_status = cry_status_tracker[client_id]
                        device_status['last_cry_time'] = timestamp
                        
                        # Cancel any existing no-cry timer
                        if device_status['checking_no_cry'] and device_status['no_cry_timer'] is not None:
                            if device_status['no_cry_timer'].is_alive():
                                device_status['no_cry_timer'].cancel()
                            device_status['checking_no_cry'] = False
                            device_status['no_cry_timer'] = None
                            logger.info(f"Cancelled no-cry timer for device {client_id} - new crying detected")

                        # Check Firebase events and send notification if needed
                        notification_sent = await check_firebase_events_and_notify(
                            timestamp=timestamp,
                            device_id=device_id,
                            confidence=confidence
                        )
                        
                        if notification_sent:
                            # Send alert through websocket
                            await manager.send_message(client_id, {
                                "type": "alert",
                                "timestamp": timestamp,
                                "message": "Crying detected!",
                                "confidence": confidence,
                                "deviceId": device_id
                            })
                            logger.info(f"Cry event created at timestamp {timestamp} for device {device_id}")
                            
                        # Start no-cry timer if not already checking
                        if device_status['last_cry_time'] is not None and not device_status['checking_no_cry']:
                            device_status['checking_no_cry'] = True
                            # Schedule the no-cry check after 10 seconds
                            no_cry_timer = threading.Timer(
                                NO_CRY_CHECK_SECONDS, 
                                await check_for_no_cry, 
                                args=[client_id, timestamp]
                            )
                            no_cry_timer.daemon = True
                            no_cry_timer.start()
                            device_status['no_cry_timer'] = no_cry_timer
                            logger.info(f"Started no-cry timer for device {client_id}")
                    
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
        # Cleanup cry status tracking
        if client_id in cry_status_tracker:
            if cry_status_tracker[client_id]['no_cry_timer'] is not None:
                try:
                    cry_status_tracker[client_id]['no_cry_timer'].cancel()
                except:
                    pass
            del cry_status_tracker[client_id]
        manager.disconnect(client_id)

async def check_for_no_cry(client_id, last_cry_timestamp):
    """
    Function that runs after 10 seconds to check if crying has stopped
    
    Parameters:
    ----------
    client_id : str
        Device ID to check
    last_cry_timestamp : float
        Timestamp when the last cry was detected
    """
    try:
        global cry_status_tracker
        
        # If device is no longer connected, do nothing
        if client_id not in cry_status_tracker:
            logger.warning(f"Device {client_id} not in tracker when checking for no-cry")
            return
            
        device_status = cry_status_tracker[client_id]
        
        # If this is an outdated timer (a newer cry was detected), do nothing
        if device_status['last_cry_time'] != last_cry_timestamp:
            logger.info(f"No-cry check skipped for device {client_id} - newer cry detected")
            return
            
        # If we get here, 10 seconds have passed with no new cry detection
        current_time = time.time()
        
        # Send no-cry notification to Firebase
        from api.firebase_service import send_nocry_notification
        notification_sent = await send_nocry_notification({
            'timestamp': current_time,
            'deviceId': client_id,
            'lastCryTimestamp': last_cry_timestamp
        })
        
        if notification_sent:
            logger.info(f"No-cry event detected at {current_time} for device {client_id}. "
                      f"Last cry was at {last_cry_timestamp}, {current_time - last_cry_timestamp:.2f}s ago")
        
        # Reset the checking flag
        device_status['checking_no_cry'] = False
        device_status['no_cry_timer'] = None
        
    except Exception as e:
        logger.error(f"Error in check_for_no_cry: {str(e)}")

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "warning", "message": "Model not loaded yet"}
    return {"status": "ok", "message": "Service is running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
