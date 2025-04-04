import os
import sys
import librosa
import soundfile as sf
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_processing import analyze_audio_segments  # Assuming this function is defined in data_processing.py
import uvicorn

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add global variable to store processed audio file path
processed_audio_path = None

@app.post("/api/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    """API endpoint to process audio file with VAD"""
    if not audio:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Create temp input file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_input.close()
    
    # Save uploaded file
    with open(temp_input.name, "wb") as buffer:
        content = await audio.read()
        buffer.write(content)
    
    try:
        # Load audio
        y, sr = librosa.load(temp_input.name, sr=16000)
        
        # Analyze segments
        crying_segments = analyze_audio_segments(y, sr)
        
        if not crying_segments:
            return {
                'is_baby_crying': False,
                'segments': [],
                'message': 'No crying detected in any segment'
            }
        
        # Store processed segments
        global processed_audio_path
        processed_audio_path = []
        
        # Prepare segments info
        segments_info = []
        for i, seg in enumerate(crying_segments):
            # Save segment audio
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{i}.wav')
            temp_output.close()
            sf.write(temp_output.name, seg['audio'], sr)
            processed_audio_path.append(temp_output.name)
            
            segments_info.append({
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'f0': float(seg['f0'])
            })
        
        return {
            'is_baby_crying': True,
            'segments': segments_info,
            'message': f'Found {len(crying_segments)} crying segments'
        }
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_input.name):
            os.unlink(temp_input.name)

@app.get("/api/get-processed-audio/{segment_index}")
async def get_processed_audio(segment_index: int):
    """API endpoint to get a specific processed audio segment"""
    global processed_audio_path
    
    if not processed_audio_path or not isinstance(processed_audio_path, list):
        raise HTTPException(status_code=404, detail="No processed audio available")
    
    if segment_index < 0 or segment_index >= len(processed_audio_path):
        raise HTTPException(status_code=404, detail="Invalid segment index")
    
    audio_path = processed_audio_path[segment_index]
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(audio_path, media_type="audio/wav")

def cleanup():
    """Clean up all temporary audio files"""
    global processed_audio_path
    if processed_audio_path and isinstance(processed_audio_path, list):
        for path in processed_audio_path:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

# Register cleanup function
import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)