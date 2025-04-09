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

    // Add timeline elements
    const timelineContainer = document.getElementById('timelineContainer');
    const timelineChart = document.getElementById('timelineChart');
    const timelineAxis = document.getElementById('timelineAxis');

    // Result elements
    const isBabyCryingElement = document.getElementById('isBabyCrying');
    const rmsValueElement = document.getElementById('rmsValue');
    const zcrValueElement = document.getElementById('zcrValue');
    const f0ValueElement = document.getElementById('f0Value');

    // Chart objects
    let originalWaveformChart = null;
    let processedWaveformChart = null;

    // Audio duration
    let audioDuration = 0;

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

            // Get audio duration when loaded
            originalAudio.onloadedmetadata = function () {
                audioDuration = originalAudio.duration;
            };

            // Reset results
            if (vadResultsContainer) vadResultsContainer.style.display = 'none';
            if (timelineContainer) timelineContainer.style.display = 'none';
            if (timelineChart) timelineChart.innerHTML = '';
            if (timelineAxis) timelineAxis.innerHTML = '';
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
            // Create timeline visualization
            createTimelineVisualization(data.segments);

            // Create segments list
            const segmentsList = document.createElement('div');
            segmentsList.className = 'segments-list';

            data.segments.forEach((segment, index) => {
                const segmentDiv = document.createElement('div');
                segmentDiv.className = 'segment-item';
                segmentDiv.dataset.segmentIndex = index;

                // Create segment info with audio player - fixing syntax errors
                segmentDiv.innerHTML = `
                    <div class="segment-info">
                        <span class="segment-time">Đoạn ${index + 1}: ${segment.start_time.toFixed(1)}s - ${segment.end_time.toFixed(1)}s</span>
                        <span class="segment-f0">F0: ${segment.f0.toFixed(1)} Hz</span>
                    </div>
                    <audio controls class="segment-audio" data-segment-index="${index}">
                        <source src="http://localhost:5000/api/get-processed-audio/${index}" type="audio/wav">
                    </audio>
                `;

                segmentsList.appendChild(segmentDiv);
            });

            // Clear and update segments container
            segmentsContainer.innerHTML = '';
            segmentsContainer.appendChild(segmentsList);
            segmentsContainer.style.display = 'block';

            // Add click events to timeline segments
            addTimelineInteractions();
        }

        vadResultsContainer.style.display = 'block';
    }

    function createTimelineVisualization(segments) {
        // Clear previous timeline
        timelineChart.innerHTML = '';
        timelineAxis.innerHTML = '';

        // Get audio duration from the loaded audio element
        const duration = audioDuration || 60; // Default to 60s if duration is unknown

        // Create timeline segments
        segments.forEach((segment, index) => {
            const startPercent = (segment.start_time / duration) * 100;
            const widthPercent = ((segment.end_time - segment.start_time) / duration) * 100;

            const segmentElement = document.createElement('div');
            segmentElement.className = 'timeline-segment';
            segmentElement.dataset.segmentIndex = index;
            segmentElement.style.left = startPercent + '%';
            segmentElement.style.width = widthPercent + '%';
            segmentElement.title = `Đoạn ${index + 1}: ${segment.start_time.toFixed(1)}s - ${segment.end_time.toFixed(1)}s (F0: ${segment.f0.toFixed(1)} Hz)`;

            timelineChart.appendChild(segmentElement);
        });

        // Create axis ticks
        const tickInterval = duration <= 30 ? 5 : 10; // 5s ticks for short audio, 10s for longer
        for (let i = 0; i <= duration; i += tickInterval) {
            const tickPercent = (i / duration) * 100;

            const tick = document.createElement('div');
            tick.className = 'timeline-tick';
            tick.style.left = tickPercent + '%';
            timelineAxis.appendChild(tick);

            const tickLabel = document.createElement('div');
            tickLabel.className = 'timeline-tick-label';
            tickLabel.style.left = tickPercent + '%';
            tickLabel.textContent = i + 's';
            timelineAxis.appendChild(tickLabel);
        }

        // Show timeline container
        timelineContainer.style.display = 'block';
    }

    function addTimelineInteractions() {
        // Add click events to timeline segments
        document.querySelectorAll('.timeline-segment').forEach(segment => {
            segment.addEventListener('click', function () {
                const index = this.dataset.segmentIndex;
                highlightSegment(index);

                // Play the corresponding audio
                const audio = document.querySelector(`.segment-audio[data-segment-index="${index}"]`);
                if (audio) {
                    audio.currentTime = 0;
                    audio.play();
                }
            });
        });

        // Add click events to segment items
        document.querySelectorAll('.segment-item').forEach(item => {
            item.addEventListener('click', function () {
                const index = this.dataset.segmentIndex;
                highlightSegment(index);
            });
        });
    }

    function highlightSegment(index) {
        // Remove highlight from all segments
        document.querySelectorAll('.timeline-segment').forEach(seg => {
            seg.classList.remove('active');
        });
        document.querySelectorAll('.segment-item').forEach(item => {
            item.classList.remove('active');
        });

        // Add highlight to selected segment
        const timelineSegment = document.querySelector(`.timeline-segment[data-segment-index="${index}"]`);
        if (timelineSegment) {
            timelineSegment.classList.add('active');
        }

        const segmentItem = document.querySelector(`.segment-item[data-segment-index="${index}"]`);
        if (segmentItem) {
            segmentItem.classList.add('active');
            segmentItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    function resetResults() {
        processedAudioContainer.style.display = 'none';
        vadResultsContainer.style.display = 'none';
        timelineContainer.style.display = 'none';
        segmentsContainer.innerHTML = '';
        timelineChart.innerHTML = '';
        timelineAxis.innerHTML = '';

        isBabyCryingElement.textContent = '';
        rmsValueElement.textContent = '';
        zcrValueElement.textContent = '';
        f0ValueElement.textContent = '';
    }
});