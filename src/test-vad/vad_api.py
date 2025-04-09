import os
import sys
import librosa
import soundfile as sf
import io
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_processing import analyze_audio_segments  # Assuming this function is defined in data_processing.py
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    if not audio.filename.lower().endswith(('.wav')):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    # Read audio file
    audio_content = await audio.read()
    audio_io = io.BytesIO(audio_content)
    
    try:
        # Load audio
        logger.info(f"Processing audio file: {audio.filename}")
        y, sr = librosa.load(audio_io, sr=16000)
        
        # Analyze segments
        logger.info("Analyzing audio segments")
        try:
            crying_segments = analyze_audio_segments(y, sr)
            if crying_segments:
                logger.info(f"First segment structure: {list(crying_segments[0].keys())}")
        except Exception as analyze_error:
            logger.error(f"Error in analyze_audio_segments: {str(analyze_error)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error analyzing audio: {str(analyze_error)}")
        
        if not crying_segments:
            logger.info("No crying segments detected")
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
            try:
                # Extract the segment from the original audio using start_sample and end_sample
                if 'start_sample' in seg and 'end_sample' in seg:
                    start_sample = seg['start_sample']
                    end_sample = seg['end_sample']
                    # Ensure we don't go beyond array bounds
                    start_sample = max(0, min(start_sample, len(y)-1))
                    end_sample = max(start_sample+1, min(end_sample, len(y)))
                    seg_audio = y[start_sample:end_sample]
                else:
                    # Fallback to using start_time and end_time
                    start_sample = int(seg.get('start_time', 0) * sr)
                    end_sample = int(seg.get('end_time', 0) * sr)
                    start_sample = max(0, min(start_sample, len(y)-1))
                    end_sample = max(start_sample+1, min(end_sample, len(y)))
                    seg_audio = y[start_sample:end_sample]
                
                # Save segment audio
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{i}.wav')
                temp_output.close()
                sf.write(temp_output.name, seg_audio, sr)
                processed_audio_path.append(temp_output.name)
                
                # Create segment info with all available fields
                segment_info = {
                    'start_time': seg.get('start_time', 0),
                    'end_time': seg.get('end_time', 0),
                }
                
                # Add optional fields if they exist
                if 'mean_f0' in seg:
                    segment_info['f0'] = float(seg['mean_f0'])
                elif 'f0' in seg:
                    segment_info['f0'] = float(seg['f0'])
                
                segments_info.append(segment_info)
                logger.info(f"Successfully processed segment {i}")
                
            except Exception as seg_error:
                logger.error(f"Error processing segment {i}: {str(seg_error)}")
                logger.error(traceback.format_exc())
                continue
        
        logger.info(f"Successfully processed {len(segments_info)} crying segments")
        return {
            'is_baby_crying': len(segments_info) > 0,
            'segments': segments_info,
            'message': f'Found {len(segments_info)} crying segments'
        }
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    

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