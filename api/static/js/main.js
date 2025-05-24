document.addEventListener('DOMContentLoaded', function () {
    // DOM elements
    const uploadForm = document.getElementById('uploadForm');
    const audioFile = document.getElementById('audioFile');
    const analyzeButton = document.getElementById('analyzeButton');
    const analyzeText = document.getElementById('analyzeText');
    const analyzeSpinner = document.getElementById('analyzeSpinner');
    const resultsContainer = document.getElementById('resultsContainer');

    // Upload form submission
    uploadForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        // Validate file
        if (!audioFile.files || audioFile.files.length === 0) {
            alert('Please select an audio file to upload.');
            return;
        }

        const file = audioFile.files[0];
        if (!file.name.toLowerCase().endsWith('.wav')) {
            alert('Please select a WAV audio file.');
            return;
        }

        // Set up audio player
        setupAudioPlayer(file);

        // Show loading state
        setLoading(true);

        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('audio', file);

            // Send to API
            const response = await fetch('/analyze/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Error analyzing audio');
            }

            // Process results
            const result = await response.json();
            displayResults(result, file);

        } catch (error) {
            console.error('Error:', error);
            alert('Error analyzing audio: ' + error.message);
        } finally {
            setLoading(false);
        }
    });

    // Set loading state
    function setLoading(isLoading) {
        if (isLoading) {
            analyzeText.classList.add('d-none');
            analyzeSpinner.classList.remove('d-none');
            analyzeButton.disabled = true;
        } else {
            analyzeText.classList.remove('d-none');
            analyzeSpinner.classList.add('d-none');
            analyzeButton.disabled = false;
        }
    }

    // Display analysis results
    function displayResults(result, audioFile) {
        // Show results container
        resultsContainer.classList.remove('d-none');

        // Fill in basic info
        document.getElementById('resultFilename').textContent = result.filename;
        document.getElementById('resultDuration').textContent = formatTime(result.summary.audio_length);
        document.getElementById('resultTotalSegments').textContent = result.summary.total_segments;

        // Show "No crying detected" message if applicable
        const noCryMessage = document.getElementById('noCryMessage');
        if (result.summary.cry_segments === 0) {
            noCryMessage.classList.remove('d-none');
        } else {
            noCryMessage.classList.add('d-none');
        }

        // Show consecutive cry message if applicable
        const consecutiveCryMessage = document.getElementById('consecutiveCryMessage');
        const consecutiveCryDetails = document.getElementById('consecutiveCryDetails');

        if (result.consecutive_cry_info.detected) {
            consecutiveCryMessage.classList.remove('d-none');

            // Format segment numbers
            const segments = result.consecutive_cry_info.segments.map(s => s + 1);
            const timeRange = `${formatTime(result.consecutive_cry_info.start_time)} - ${formatTime(result.consecutive_cry_info.end_time)}`;

            consecutiveCryDetails.innerHTML = `
                <strong>Segments:</strong> ${segments.join(', ')}<br>
                <strong>Time Range:</strong> ${timeRange}
            `;
        } else {
            consecutiveCryMessage.classList.add('d-none');
        }

        // Draw audio waveform and segments visualization
        drawWaveform(audioFile, result.segments);

        // Draw timeline visualization
        drawTimeline(result.segments, result.summary.audio_length, result.consecutive_cry_info);

        // Fill segments table
        populateSegmentsTable(result.segments, result.consecutive_cry_info);

        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    // Draw audio waveform
    function drawWaveform(audioFile, segments) {
        const canvas = document.getElementById('waveformCanvas');
        const ctx = canvas.getContext('2d');

        // Set canvas dimensions
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = 200;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Create audio context
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Create a FileReader to read the audio file
        const reader = new FileReader();
        reader.onload = function (event) {
            // Decode the audio data
            audioContext.decodeAudioData(event.target.result, function (buffer) {
                // Get raw audio data
                const rawData = buffer.getChannelData(0);

                // Downsample for visualization (we don't need every sample)
                const step = Math.ceil(rawData.length / canvas.width);
                const reducedData = [];

                for (let i = 0; i < canvas.width; i++) {
                    const start = i * step;
                    const end = start + step;
                    let min = 1.0;
                    let max = -1.0;

                    for (let j = start; j < end; j++) {
                        if (j < rawData.length) {
                            const val = rawData[j];
                            min = Math.min(min, val);
                            max = Math.max(max, val);
                        }
                    }
                    reducedData.push([min, max]);
                }

                // Draw the waveform
                ctx.strokeStyle = '#007bff';
                ctx.lineWidth = 1;
                ctx.beginPath();

                const middle = canvas.height / 2;

                for (let i = 0; i < reducedData.length; i++) {
                    const [min, max] = reducedData[i];

                    // Calculate y-coordinates (middle of canvas ± amplitude)
                    const minY = middle + (min * middle * 0.8);
                    const maxY = middle + (max * middle * 0.8);

                    ctx.moveTo(i, minY);
                    ctx.lineTo(i, maxY);
                }

                ctx.stroke();

                // Draw segment markers
                drawSegmentMarkersOnWaveform(ctx, segments, buffer.duration, canvas.width, canvas.height);

            }, function (err) {
                console.error('Error decoding audio data: ' + err);
            });
        };

        // Read the audio file as array buffer
        reader.readAsArrayBuffer(audioFile);
    }

    // Draw segment markers on waveform
    function drawSegmentMarkersOnWaveform(ctx, segments, totalDuration, width, height) {
        // Draw waveform title
        ctx.fillStyle = '#000';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Audio Waveform with Detected Segments', width / 2, 15);

        segments.forEach(segment => {
            const startX = (segment.start_time / totalDuration) * width;
            const endX = (segment.end_time / totalDuration) * width;

            // Draw rectangle for segment
            ctx.fillStyle = segment.predicted_class === 'cry' ?
                'rgba(220, 53, 69, 0.3)' :
                'rgba(25, 135, 84, 0.3)';

            ctx.fillRect(startX, 20, endX - startX, height - 20); // Adjusted to leave room for title

            // Draw segment border
            ctx.strokeStyle = segment.predicted_class === 'cry' ?
                'rgba(220, 53, 69, 0.8)' :
                'rgba(25, 135, 84, 0.8)';
            ctx.lineWidth = 1;
            ctx.strokeRect(startX, 20, endX - startX, height - 20);

            // Draw segment number
            ctx.fillStyle = '#000';
            ctx.font = '10px Arial';
            ctx.fillText(segment.segment_index + 1, startX + 2, 32);
        });

        // Draw legend
        createWaveformLegend();
    }

    // Create legend for waveform
    function createWaveformLegend() {
        const legendContainer = document.getElementById('waveformLegend');
        legendContainer.innerHTML = `
            <div class="visualization-legend">
                <div class="legend-item">
                    <div class="legend-color legend-cry"></div>
                    <span>Cry</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color legend-not-cry"></div>
                    <span>Not Cry</span>
                </div>
            </div>
        `;
    }

    // Draw timeline visualization
    function drawTimeline(segments, totalDuration, consecutiveCryInfo) {
        const canvas = document.getElementById('timelineCanvas');
        const ctx = canvas.getContext('2d');

        // Set canvas dimensions
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = 100;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw title
        ctx.fillStyle = '#000';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Audio Timeline with Cry Detection', canvas.width / 2, 15);

        // Draw timeline background
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 20, canvas.width, canvas.height - 20);

        // Draw time markers
        const timeMarkerCount = Math.min(Math.ceil(totalDuration), 20);
        const step = totalDuration / timeMarkerCount;

        ctx.strokeStyle = '#adb5bd';
        ctx.fillStyle = '#6c757d';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';

        for (let i = 0; i <= timeMarkerCount; i++) {
            const x = (i * step / totalDuration) * canvas.width;

            // Draw line
            ctx.beginPath();
            ctx.moveTo(x, 20);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();

            // Draw time
            ctx.fillText(formatTime(i * step), x, canvas.height - 5);
        }

        // If no segments, show "No segments detected" message
        if (segments.length === 0) {
            ctx.fillStyle = '#6c757d';
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('No segments detected in the audio', canvas.width / 2, 60);
        }

        // Draw segments
        segments.forEach(segment => {
            const startX = (segment.start_time / totalDuration) * canvas.width;
            const endX = (segment.end_time / totalDuration) * canvas.width;
            const segWidth = endX - startX;

            // Determine if this is a consecutive cry segment
            const isConsecutiveCry = consecutiveCryInfo.detected &&
                consecutiveCryInfo.segments.includes(segment.segment_index);

            // Draw segment
            ctx.fillStyle = segment.predicted_class === 'cry' ?
                isConsecutiveCry ? 'rgba(220, 53, 69, 1.0)' : 'rgba(220, 53, 69, 0.7)' :
                'rgba(25, 135, 84, 0.7)';

            const barHeight = 40;
            const barY = 30;
            ctx.fillRect(startX, barY, segWidth, barHeight);

            // Draw segment number if width permits
            if (segWidth > 15) {
                ctx.fillStyle = '#fff';
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(segment.segment_index + 1, startX + segWidth / 2, barY + barHeight / 2 + 4);
            }

            // Draw confidence indicator (height of a bar above the segment)
            const confHeight = 10;
            const confY = barY + barHeight + 2;

            ctx.fillStyle = getConfidenceColor(segment.confidence);
            const confWidth = segWidth * 0.8;
            const confX = startX + (segWidth - confWidth) / 2;
            ctx.fillRect(confX, confY, confWidth, confHeight);
        });

        // Create legend for timeline
        createTimelineLegend();
    }

    // Create legend for timeline
    function createTimelineLegend() {
        const legendContainer = document.getElementById('timelineLegend');
        legendContainer.innerHTML = `
            <div class="visualization-legend">
                <div class="legend-item">
                    <div class="legend-color legend-cry"></div>
                    <span>Cry</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color legend-consecutive-cry"></div>
                    <span>Consecutive Cry</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color legend-not-cry"></div>
                    <span>Not Cry</span>
                </div>
                <div class="legend-item ml-3">
                    <div style="width: 40px; height: 5px; background-color: rgba(25, 135, 84, 0.8); margin-right: 5px;"></div>
                    <span>High Confidence</span>
                </div>
                <div class="legend-item">
                    <div style="width: 40px; height: 5px; background-color: rgba(255, 193, 7, 0.8); margin-right: 5px;"></div>
                    <span>Medium Confidence</span>
                </div>
                <div class="legend-item">
                    <div style="width: 40px; height: 5px; background-color: rgba(220, 53, 69, 0.8); margin-right: 5px;"></div>
                    <span>Low Confidence</span>
                </div>
            </div>
        `;
    }

    // Populate segments table
    function populateSegmentsTable(segments, consecutiveCryInfo) {
        const tableBody = document.querySelector('#segmentsTable tbody');
        tableBody.innerHTML = '';

        segments.forEach(segment => {
            const row = document.createElement('tr');

            // Determine if this is a consecutive cry segment
            const isConsecutiveCry = consecutiveCryInfo.detected &&
                consecutiveCryInfo.segments.includes(segment.segment_index);

            // Apply appropriate class
            if (segment.predicted_class === 'cry') {
                row.className = isConsecutiveCry ? 'prediction-consecutive-cry' : 'prediction-cry';
            } else {
                row.className = 'prediction-not-cry';
            }

            // Segment number
            const numCell = document.createElement('td');
            numCell.textContent = segment.segment_index + 1;
            if (isConsecutiveCry) {
                numCell.innerHTML += ' <span class="badge bg-danger">Consecutive</span>';
            }
            row.appendChild(numCell);

            // Time range
            const timeCell = document.createElement('td');
            timeCell.textContent = `${formatTime(segment.start_time)} - ${formatTime(segment.end_time)}`;
            row.appendChild(timeCell);

            // Prediction
            const predCell = document.createElement('td');
            predCell.textContent = segment.predicted_class;
            predCell.style.fontWeight = 'bold';
            predCell.style.color = segment.predicted_class === 'cry' ? '#dc3545' : '#198754';
            row.appendChild(predCell);

            // Confidence
            const confCell = document.createElement('td');
            confCell.textContent = `${(segment.confidence * 100).toFixed(2)}%`;
            confCell.className = getConfidenceClass(segment.confidence);
            row.appendChild(confCell);

            tableBody.appendChild(row);
        });
    }

    // Helper function to format time as MM:SS
    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    // Get confidence color based on value
    function getConfidenceColor(confidence) {
        if (confidence >= 0.9) return 'rgba(25, 135, 84, 0.8)';  // Green
        if (confidence >= 0.7) return 'rgba(255, 193, 7, 0.8)';  // Yellow
        return 'rgba(220, 53, 69, 0.8)';  // Red
    }

    // Get confidence class based on value
    function getConfidenceClass(confidence) {
        if (confidence >= 0.9) return 'confidence-high';
        if (confidence >= 0.7) return 'confidence-medium';
        return 'confidence-low';
    }

    // Handle window resize to redraw charts
    window.addEventListener('resize', function () {
        // Window resize handler (pie chart removed)
    });

    // Function to set up the audio player
    function setupAudioPlayer(audioFile) {
        const audioPlayer = document.getElementById('audioPlayer');
        const playBtn = document.getElementById('playAudioBtn');
        const pauseBtn = document.getElementById('pauseAudioBtn');

        // Create object URL for the audio file
        const audioUrl = URL.createObjectURL(audioFile);
        audioPlayer.src = audioUrl;

        // Set up button event listeners
        playBtn.addEventListener('click', () => {
            audioPlayer.play();
        });

        pauseBtn.addEventListener('click', () => {
            audioPlayer.pause();
        });
    }
});
