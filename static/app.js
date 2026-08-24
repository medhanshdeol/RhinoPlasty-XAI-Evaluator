document.addEventListener('DOMContentLoaded', () => {
    const planInput = document.getElementById('plan-input');
    const actualInput = document.getElementById('actual-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    
    let planFile = null;
    let actualFile = null;
    let planDepthFile = null;
    let actualDepthFile = null;

    // Handle file selection
    planInput.addEventListener('change', (e) => handleFileSelect(e, 'plan'));
    actualInput.addEventListener('change', (e) => handleFileSelect(e, 'actual'));
    
    document.getElementById('plan-depth-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            planDepthFile = file;
            document.getElementById('plan-depth-status').textContent = `Loaded: ${file.name}`;
        }
    });
    
    document.getElementById('actual-depth-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            actualDepthFile = file;
            document.getElementById('actual-depth-status').textContent = `Loaded: ${file.name}`;
        }
    });

    // Camera UI Elements
    const captureCanvas = document.getElementById('capture-canvas');
    let activeStream = null;

    async function setupCamera(type, facingMode) {
        const video = document.getElementById(`${type}-video`);
        const captureBtn = document.getElementById(`${type}-capture-btn`);
        const previewContainer = document.getElementById(`${type}-preview`);
        
        const constraints = {
            video: { facingMode: { exact: facingMode } }
        };
        
        try {
            // First try exact match, if it fails, fallback to ideal
            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (exactError) {
                console.warn("Exact facingMode failed, trying ideal", exactError);
                stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facingMode } });
            }
            
            activeStream = stream;
            video.srcObject = stream;
            video.style.display = 'block';
            captureBtn.style.display = 'block';
            previewContainer.innerHTML = '';
        } catch (err) {
            alert(`Could not access ${facingMode} camera: ` + err.message);
            return;
        }

        captureBtn.onclick = () => {
            const context = captureCanvas.getContext('2d');
            captureCanvas.width = video.videoWidth;
            captureCanvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
            
            // Stop stream
            if (activeStream) {
                activeStream.getTracks().forEach(track => track.stop());
            }
            video.style.display = 'none';
            captureBtn.style.display = 'none';

            captureCanvas.toBlob((blob) => {
                const file = new File([blob], `captured_${type}.jpg`, { type: "image/jpeg" });
                if (type === 'plan') planFile = file;
                else actualFile = file;
                
                const url = URL.createObjectURL(blob);
                previewContainer.innerHTML = `<img src="${url}" alt="${type} preview">`;
                
                if (planFile && actualFile) {
                    analyzeBtn.disabled = false;
                }
            }, 'image/jpeg', 0.95);
        };
    }

    document.getElementById('plan-laptop-btn').addEventListener('click', () => setupCamera('plan', 'user'));
    document.getElementById('plan-phone-btn').addEventListener('click', () => setupCamera('plan', 'environment'));
    document.getElementById('actual-laptop-btn').addEventListener('click', () => setupCamera('actual', 'user'));
    document.getElementById('actual-phone-btn').addEventListener('click', () => setupCamera('actual', 'environment'));
    
    // Slider logic
    const slider = document.getElementById('satisfaction-slider');
    const sliderVal = document.getElementById('satisfaction-val');
    slider.addEventListener('input', (e) => {
        sliderVal.textContent = parseFloat(e.target.value).toFixed(1);
    });

    function handleFileSelect(event, type) {
        const file = event.target.files[0];
        if (file) {
            if (type === 'plan') planFile = file;
            else actualFile = file;

            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                const previewContainer = document.getElementById(`${type}-preview`);
                previewContainer.innerHTML = `<img src="${e.target.result}" alt="${type} preview">`;
            };
            reader.readAsDataURL(file);

            // Enable analyze button if both files are selected
            if (planFile && actualFile) {
                analyzeBtn.disabled = false;
            }
        }
    }

    // Handle Analysis
    analyzeBtn.addEventListener('click', async () => {
        if (!planFile || !actualFile) return;

        // UI updates
        analyzeBtn.disabled = true;
        resultsSection.classList.add('hidden');
        loader.classList.remove('hidden');

        const formData = new FormData();
        formData.append('planned_img', planFile);
        formData.append('actual_img', actualFile);
        formData.append('patient_satisfaction', slider.value);
        
        if (planDepthFile) formData.append('plan_depth', planDepthFile);
        if (actualDepthFile) formData.append('actual_depth', actualDepthFile);

        try {
            const response = await fetch('/analyze_differential', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Analysis failed');
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            loader.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    function displayResults(data) {
        // Set heatmap image
        const heatmapImg = document.getElementById('heatmap-image');
        // Add timestamp to bypass browser caching if the same path is reused
        heatmapImg.src = data.heatmap_url || `/download_heatmap?path=${data.heatmap_path}&t=${new Date().getTime()}`;

        // Populate metrics
        const metricsGrid = document.getElementById('metrics-grid');
        metricsGrid.innerHTML = '';

        // Add Holistic Score card
        if (data.holistic_score !== undefined) {
            const hCard = document.createElement('div');
            hCard.className = 'metric-card';
            hCard.style.border = '2px solid var(--mint)';
            hCard.innerHTML = `
                <div class="metric-name" style="color: var(--mint);">Holistic Outcome Score</div>
                <div class="metric-values">
                    <div style="font-size: 1.5rem; font-weight: bold;">${data.holistic_score.toFixed(1)} / 100</div>
                </div>
            `;
            metricsGrid.appendChild(hCard);
        }

        for (const [region, scores] of Object.entries(data.regional_scores)) {
            if (region === 'SFI_overall') continue;
            
            // Determine color based on mean error (just a visual enhancement)
            let colorStyle = '';
            if (scores.mean_error_mm > 0.05) colorStyle = 'color: var(--danger)';
            else if (scores.mean_error_mm > 0.02) colorStyle = 'color: var(--warning)';
            else colorStyle = 'color: var(--success)';

            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <div class="metric-name">${region.replace('_', ' ')}</div>
                <div class="metric-values">
                    <div>Mean Err: <span class="error-val" style="${colorStyle}">${scores.mean_error_mm.toFixed(4)}</span></div>
                    <div>Max Err: <span>${scores.max_error_mm.toFixed(4)}</span></div>
                </div>
            `;
            metricsGrid.appendChild(card);
        }

        // Show results
        resultsSection.classList.remove('hidden');
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
