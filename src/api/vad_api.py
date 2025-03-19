import os
import sys
import numpy as np
import librosa
import soundfile as sf
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_processing.split_audio import analyze_audio_segments, pad_or_trim_segment

app = Flask(__name__)
CORS(app)  # Cho phép CORS để frontend có thể gọi API

# Add global variable to store processed audio file path
processed_audio_path = None

@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    """API endpoint to process audio file with VAD"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    # Create temp input file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_input.close()
    file.save(temp_input.name)
    
    try:
        # Load audio
        y, sr = librosa.load(temp_input.name, sr=16000)
        
        # Analyze segments
        crying_segments = analyze_audio_segments(y, sr)
        
        if not crying_segments:
            return jsonify({
                'is_baby_crying': False,
                'segments': [],
                'message': 'No crying detected in any segment'
            })
        
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
        
        return jsonify({
            'is_baby_crying': True,
            'segments': segments_info,
            'message': f'Found {len(crying_segments)} crying segments'
        })
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        if os.path.exists(temp_input.name):
            os.unlink(temp_input.name)

@app.route('/api/get-processed-audio/<int:segment_index>', methods=['GET'])
def get_processed_audio(segment_index):
    """API endpoint to get a specific processed audio segment"""
    global processed_audio_path
    
    if not processed_audio_path or not isinstance(processed_audio_path, list):
        return jsonify({'error': 'No processed audio available'}), 404
    
    if segment_index < 0 or segment_index >= len(processed_audio_path):
        return jsonify({'error': 'Invalid segment index'}), 404
    
    audio_path = processed_audio_path[segment_index]
    if not os.path.exists(audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    
    return send_file(audio_path, mimetype='audio/wav')

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

import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    app.run(debug=True, port=5000)