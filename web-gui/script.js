document.addEventListener('DOMContentLoaded', function () {
    // DOM elements
    const audioFileInput = document.getElementById('audioFile');
    const fileNameSpan = document.getElementById('fileName');
    const processButton = document.getElementById('processButton');
    const errorMessage = document.getElementById('errorMessage');
    const originalAudio = document.getElementById('originalAudio');
    const processedAudio = document.getElementById('processedAudio');
    const processedAudioContainer = document.getElementById('processedAudioContainer');

    // Remove these lines
    const vadResultsContainer = document.getElementById('vadResultsContainer');
    const segmentsContainer = document.getElementById('segmentsContainer');

    // Result elements
    const isBabyCryingElement = document.getElementById('isBabyCrying');
    const rmsValueElement = document.getElementById('rmsValue');
    const zcrValueElement = document.getElementById('zcrValue');
    const f0ValueElement = document.getElementById('f0Value');

    // Chart objects
    let originalWaveformChart = null;
    let processedWaveformChart = null;

    // Event listeners
    audioFileInput.addEventListener('change', handleFileSelect);
    processButton.addEventListener('click', processAudio);

    // Handle file selection
    function handleFileSelect(event) {
        const file = event.target.files[0];

        if (file && file.type === 'audio/wav') {
            fileNameSpan.textContent = file.name;
            processButton.disabled = false;
            errorMessage.style.display = 'none';

            // Create URL for original audio
            const audioUrl = URL.createObjectURL(file);
            originalAudio.src = audioUrl;

            // Reset results
            if (vadResultsContainer) vadResultsContainer.style.display = 'none';
            if (segmentsContainer) segmentsContainer.innerHTML = '';
            if (isBabyCryingElement) isBabyCryingElement.textContent = '';
        } else {
            fileNameSpan.textContent = '';
            processButton.disabled = true;
            errorMessage.textContent = 'Vui lòng chọn file âm thanh .wav';
            errorMessage.style.display = 'block';
            originalAudio.src = '';
        }
    }

    // Process audio with VAD
    function processAudio() {
        const file = audioFileInput.files[0];

        if (!file) {
            errorMessage.textContent = 'Vui lòng chọn file âm thanh trước';
            errorMessage.style.display = 'block';
            return;
        }

        // Show loading state
        processButton.disabled = true;
        processButton.textContent = 'Đang xử lý...';
        errorMessage.style.display = 'none';

        // Create form data
        const formData = new FormData();
        formData.append('audio', file);

        // Send request to backend API
        fetch('http://localhost:5000/api/process-audio', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.error || 'Network response was not ok');
                    });
                }
                return response.json();
            })
            .then(data => {
                displayResults(data);
            })
            .catch(error => {
                console.error('Error:', error);
                errorMessage.textContent = `Lỗi khi xử lý âm thanh: ${error.message}`;
                errorMessage.style.display = 'block';
            })
            .finally(() => {
                processButton.disabled = false;
                processButton.textContent = 'Áp Dụng VAD';
            });
    }

    // Display VAD results
    function displayResults(data) {
        // Show VAD results
        isBabyCryingElement.textContent = data.is_baby_crying ? 'Có' : 'Không';
        isBabyCryingElement.className = data.is_baby_crying ? 'result-value positive' : 'result-value negative';
        
        if (data.segments && data.segments.length > 0) {
            // Create segments list
            const segmentsList = document.createElement('div');
            segmentsList.className = 'segments-list';
            
            data.segments.forEach((segment, index) => {
                const segmentDiv = document.createElement('div');
                segmentDiv.className = 'segment-item';
                
                // Create segment info with audio player
                segmentDiv.innerHTML = `
                    <div class="segment-info">
                        <span class="segment-time">Đoạn ${index + 1}: ${segment.start_time.toFixed(1)}s - ${segment.end_time.toFixed(1)}s</span>
                        <span class="segment-f0">F0: ${segment.f0.toFixed(1)} Hz</span>
                    </div>
                    <audio controls>
                        <source src="http://localhost:5000/api/get-processed-audio/${index}" type="audio/wav">
                    </audio>
                `;
                
                segmentsList.appendChild(segmentDiv);
            });
            
            // Clear and update segments container
            segmentsContainer.innerHTML = '';
            segmentsContainer.appendChild(segmentsList);
            segmentsContainer.style.display = 'block';
        }
        
        vadResultsContainer.style.display = 'block';
    }

    function resetResults() {
        processedAudioContainer.style.display = 'none';
        vadResultsContainer.style.display = 'none';
        segmentsContainer.innerHTML = '';

        isBabyCryingElement.textContent = '';
        rmsValueElement.textContent = '';
        zcrValueElement.textContent = '';
        f0ValueElement.textContent = '';
    }
});