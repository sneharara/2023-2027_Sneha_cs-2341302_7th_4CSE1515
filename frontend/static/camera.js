// Camera Node controller for IgnisGuard Edge Client (Streaming Mode)

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const video = document.getElementById("webcam");
    const canvas = document.getElementById("camera-canvas");
    const ctx = canvas.getContext("2d");
    
    const startBtn = document.getElementById("start-btn");
    const stopBtn = document.getElementById("stop-btn");
    const videoOverlay = document.getElementById("video-overlay");
    const overlayText = document.getElementById("overlay-text");
    const modelDot = document.getElementById("model-dot");
    const modelStatusText = document.getElementById("model-status-text");
    const detectionBadge = document.getElementById("detection-badge");
    
    const cameraNameInput = document.getElementById("camera-name-input");
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdVal = document.getElementById("threshold-val");
    
    const testFireBtn = document.getElementById("test-fire-btn");
    const testSmokeBtn = document.getElementById("test-smoke-btn");

    // State
    let isRunning = false;
    let activeStream = null;
    let threshold = 0.8;
    let ws = null;
    
    let lastFrameTime = 0;
    const frameIntervalMs = 250; // 4 Frames Per Second (suitable for real-time monitoring + low server load)

    // Initialize threshold slider
    thresholdSlider.addEventListener("input", (e) => {
        threshold = e.target.value / 100;
        thresholdVal.innerText = `${e.target.value}%`;
    });

    // In streaming mode, the ML model runs on the server, so edge node is immediately ready
    function initEdgeNode() {
        console.log("Edge node initialized. Server-side ONNX engine active.");
        modelDot.className = "status-dot green";
        modelStatusText.innerText = "Edge Stream Active";
        startBtn.disabled = false;
        videoOverlay.style.display = "none";
    }

    // Access Webcam API
    async function setupWebcam() {
        videoOverlay.style.display = "flex";
        overlayText.innerText = "Requesting Camera Access...";
        
        try {
            activeStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: "environment" },
                audio: false
            });
            
            video.srcObject = activeStream;
            
            return new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    video.play();
                    videoOverlay.style.display = "none";
                    resolve();
                };
            });
        } catch (e) {
            console.error("Webcam access error:", e);
            overlayText.innerHTML = `<span style="color:var(--color-fire);">Camera Access Denied.</span><br/><small style="font-size:12px; color:var(--text-secondary);">Ensure permissions are allowed.</small>`;
            throw e;
        }
    }

    // Connect to WebSocket Server for video streaming
    function connectWebSocket() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            console.log(`Connecting to WebSocket: ${wsUrl}`);
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log("WebSocket stream connected.");
                resolve();
            };

            ws.onerror = (err) => {
                console.error("WebSocket connection error:", err);
                reject(err);
            };

            ws.onclose = () => {
                console.warn("WebSocket stream closed.");
                if (isRunning) {
                    // Force stop if socket closes
                    stopDetection();
                }
            };
        });
    }

    // Toggle running state
    async function startDetection() {
        if (isRunning) return;
        
        try {
            // 1. Request camera
            await setupWebcam();
            // 2. Open WebSocket stream
            await connectWebSocket();
            
            isRunning = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            
            detectionBadge.innerText = "Streaming Active";
            detectionBadge.className = "badge badge-active";
            
            // Start the frame loop
            requestAnimationFrame(inferenceLoop);
        } catch (e) {
            console.error("Failed to start streaming detection:", e);
            alert("Failed to connect to server stream. Check server status.");
            stopDetection();
        }
    }

    function stopDetection() {
        if (!isRunning && !activeStream && !ws) return;
        
        isRunning = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        
        detectionBadge.innerText = "Inference Idle";
        detectionBadge.className = "badge badge-inactive";
        
        // Stop Camera
        if (activeStream) {
            activeStream.getTracks().forEach(track => track.stop());
            activeStream = null;
        }

        // Close WebSocket
        if (ws) {
            ws.close();
            ws = null;
        }
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#1e293b";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#94a3b8";
        ctx.font = "20px 'Outfit'";
        ctx.textAlign = "center";
        ctx.fillText("Camera Stream Stopped", canvas.width / 2, canvas.height / 2);
    }

    // Core Stream Loop
    function inferenceLoop() {
        if (!isRunning) return;

        // 1. Draw current video frame to canvas (mirrored)
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();

        // 2. Draw HUD styling locally
        drawHUD();

        // 3. Upload frame to server at set interval (e.g. 4 FPS)
        const now = Date.now();
        if (now - lastFrameTime >= frameIntervalMs) {
            lastFrameTime = now;
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                // Compress frame to JPEG (0.70 quality) for better image details
                const frameBase64 = canvas.toDataURL("image/jpeg", 0.70);
                const cameraName = cameraNameInput.value.trim() || "Remote Camera";
                
                ws.send(JSON.stringify({
                    type: "CAMERA_FRAME",
                    camera_name: cameraName,
                    frame: frameBase64,
                    threshold: threshold
                }));
            }
        }

        // Keep loop running
        if (isRunning) {
            requestAnimationFrame(inferenceLoop);
        }
    }

    // Render local status overlay on the edge canvas
    function drawHUD() {
        ctx.fillStyle = "rgba(15, 23, 42, 0.6)";
        ctx.fillRect(10, 10, 240, 50);
        ctx.strokeStyle = "rgba(51, 65, 85, 0.8)";
        ctx.lineWidth = 1;
        ctx.strokeRect(10, 10, 240, 50);

        ctx.fillStyle = "#3b82f6"; // Alerting system blue
        ctx.beginPath();
        ctx.arc(25, 35, 6, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = "#f8fafc";
        ctx.font = "bold 15px 'Outfit', sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("STREAMING LIVE DATA", 40, 40);

        // Draw corner scan lines (blue theme)
        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth = 4;
        const offset = 20;
        const len = 40;

        // Top Left
        ctx.beginPath();
        ctx.moveTo(offset, offset + len); ctx.lineTo(offset, offset); ctx.lineTo(offset + len, offset);
        ctx.stroke();

        // Top Right
        ctx.beginPath();
        ctx.moveTo(canvas.width - offset - len, offset); ctx.lineTo(canvas.width - offset, offset); ctx.lineTo(canvas.width - offset, offset + len);
        ctx.stroke();

        // Bottom Left
        ctx.beginPath();
        ctx.moveTo(offset, canvas.height - offset - len); ctx.lineTo(offset, canvas.height - offset); ctx.lineTo(offset + len, canvas.height - offset);
        ctx.stroke();

        // Bottom Right
        ctx.beginPath();
        ctx.moveTo(canvas.width - offset - len, canvas.height - offset); ctx.lineTo(canvas.width - offset, canvas.height - offset); ctx.lineTo(canvas.width - offset, canvas.height - offset - len);
        ctx.stroke();
    }

    // Dev Simulation Alert Tests (triggers REST API alert)
    async function triggerSimulationAlert(label) {
        const cameraName = cameraNameInput.value.trim() || "Simulator Node";
        const confidence = 0.942;

        // Setup mock canvas for snapshot
        ctx.fillStyle = label === "Fire" ? "#b91c1c" : "#78350f";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 36px 'Outfit'";
        ctx.textAlign = "center";
        ctx.fillText(`[SIMULATED ${label.toUpperCase()}]`, canvas.width / 2, canvas.height / 2 - 10);
        ctx.font = "18px 'Plus Jakarta Sans'";
        ctx.fillText("Sent to test email notifications and siren system.", canvas.width / 2, canvas.height / 2 + 30);

        const snapshotBase64 = canvas.toDataURL("image/jpeg", 0.6);

        alert(`Triggering simulated ${label} alert. Check your Dashboard!`);

        try {
            await fetch("/api/alert", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    camera_name: cameraName,
                    label: label,
                    confidence: confidence,
                    image_base64: snapshotBase64
                })
            });
        } catch (e) {
            console.error("Simulation request failed:", e);
        }
    }

    // Connect event listeners
    startBtn.addEventListener("click", startDetection);
    stopBtn.addEventListener("click", stopDetection);
    
    testFireBtn.addEventListener("click", () => triggerSimulationAlert("Fire"));
    testSmokeBtn.addEventListener("click", () => triggerSimulationAlert("Smoke"));

    // User Session / Login Checks
    const savedUser = JSON.parse(localStorage.getItem("user"));
    if (!savedUser) {
        // Redirect to dashboard login page if not logged in
        window.location.href = "/";
    } else {
        showUserProfile(savedUser);
    }

    function showUserProfile(user) {
        const userProfile = document.getElementById("user-profile");
        const profileName = document.getElementById("profile-name");
        const profileEmail = document.getElementById("profile-email");
        if (userProfile && profileName && profileEmail) {
            profileName.innerText = user.name;
            profileEmail.innerText = user.email;
            userProfile.style.display = "block";
        }
    }

    // Logout Click Event
    const logoutBtn = document.getElementById("logout-btn");
    if (logoutBtn) {
        logoutBtn.addEventListener("click", async () => {
            const user = JSON.parse(localStorage.getItem("user"));
            if (user && user.email) {
                try {
                    await fetch("/api/logout", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ email: user.email })
                    });
                } catch (err) {
                    console.error("Logout request failed:", err);
                }
            }
            localStorage.removeItem("user");
            window.location.href = "/";
        });
    }

    // Initialize edge node
    initEdgeNode();
});
