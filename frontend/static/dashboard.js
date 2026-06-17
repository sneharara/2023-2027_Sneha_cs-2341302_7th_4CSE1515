// Dashboard controller for IgnisGuard Central Monitoring

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const alertsFeed = document.getElementById("alerts-feed");
    const noAlertsMsg = document.getElementById("no-alerts-msg");
    const statSensors = document.getElementById("stat-sensors");
    const statAlerts = document.getElementById("stat-alerts");
    const logCount = document.getElementById("log-count");
    const threatLevelText = document.getElementById("threat-level-text");
    const threatDesc = document.getElementById("threat-desc");
    const statusCard = document.getElementById("status-card");
    const statusIcon = document.getElementById("status-icon");
    const alertOverlay = document.getElementById("alert-overlay");
    
    const sirenToggleBtn = document.getElementById("siren-toggle");
    const clearBtn = document.getElementById("clear-btn");
    const systemDot = document.getElementById("system-dot");
    const systemStatusText = document.getElementById("system-status-text");

    // Modal elements
    const alertModal = document.getElementById("alert-modal");
    const modalCamName = document.getElementById("modal-cam-name");
    const modalSnapshot = document.getElementById("modal-snapshot");
    const modalLabel = document.getElementById("modal-label");
    const modalConfidence = document.getElementById("modal-confidence");
    const modalDismissBtn = document.getElementById("modal-dismiss");

    // Login & Profile UI elements
    const loginOverlay = document.getElementById("login-overlay");
    const loginForm = document.getElementById("login-form");
    const loginName = document.getElementById("login-name");
    const loginEmail = document.getElementById("login-email");
    const userProfile = document.getElementById("user-profile");
    const profileName = document.getElementById("profile-name");
    const profileEmail = document.getElementById("profile-email");
    const logoutBtn = document.getElementById("logout-btn");
    const statSubscribers = document.getElementById("stat-subscribers");

    // Live Stream Elements
    const liveStreamView = document.getElementById("live-stream-view");
    const livePlaceholder = document.getElementById("live-placeholder");
    const liveIndicatorDot = document.getElementById("live-indicator-dot");
    const liveIndicatorText = document.getElementById("live-indicator-text");



    // Audio context for alarm synthesis
    let audioCtx = null;
    let alarmInterval = null;
    let isMuted = false;
    let activeAlertsCount = 0;
    let streamTimeout = null;


    // Track active camera sensors in memory
    const activeSensors = new Set();

    // Init Websocket connection
    let ws = null;

    function connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        console.log(`Connecting to WebSocket: ${wsUrl}`);
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("WebSocket connection established.");
            systemDot.className = "status-dot green";
            systemStatusText.innerText = "Monitoring Active";
        };

        ws.onmessage = (event) => {
            if (event.data === "pong") return;
            try {
                const message = JSON.parse(event.data);
                if (message.type === "NEW_ALERT") {
                    handleNewAlert(message.data);
                } else if (message.type === "ALERTS_CLEARED") {
                    clearLocalHistory();
                } else if (message.type === "SUBSCRIBERS_UPDATED") {
                    statSubscribers.innerText = message.data;
                } else if (message.type === "CAMERA_FRAME") {
                    handleCameraFrame(message);
                }
            } catch (e) {
                console.error("Error parsing WebSocket message:", e);
            }
        };

        ws.onclose = () => {
            console.warn("WebSocket disconnected. Retrying in 3 seconds...");
            systemDot.className = "status-dot orange";
            systemStatusText.innerText = "Reconnecting...";
            setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
            ws.close();
        };
    }

    // Load initial alerts from REST API
    async function loadAlertHistory() {
        try {
            const response = await fetch("/api/alerts");
            const data = await response.json();
            
            clearFeedUI();
            
            if (data && data.length > 0) {
                data.forEach(alert => {
                    addAlertToFeed(alert, false); // Add without triggering sound/modal
                });
                updateStats();
            }
        } catch (e) {
            console.error("Failed to load alert history:", e);
        }
    }

    // Web Audio API: Synthesizes a sweeping police/fire siren
    function startSirenSound() {
        if (isMuted) return;
        if (alarmInterval) return; // Already running

        try {
            // Create AudioContext on-demand (user interaction restriction)
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }

            if (audioCtx.state === "suspended") {
                audioCtx.resume();
            }

            let osc = audioCtx.createOscillator();
            let gain = audioCtx.createGain();

            osc.type = "sine";
            osc.frequency.setValueAtTime(450, audioCtx.currentTime); // starting freq
            
            // Gain configuration (volume)
            gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
            
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            
            osc.start();

            // Sweep oscillator frequency up and down
            let sweepDirection = 1;
            alarmInterval = setInterval(() => {
                let currentFreq = osc.frequency.value;
                if (currentFreq >= 800) sweepDirection = -1;
                else if (currentFreq <= 400) sweepDirection = 1;

                // Sweeping by 30Hz step
                osc.frequency.setValueAtTime(currentFreq + (sweepDirection * 35), audioCtx.currentTime);
            }, 30);

            // Store references on window/global object to stop them later
            window.activeSirenOsc = osc;
            window.activeSirenGain = gain;
        } catch (e) {
            console.error("Failed to play alarm audio synthesizer:", e);
        }
    }

    function stopSirenSound() {
        if (alarmInterval) {
            clearInterval(alarmInterval);
            alarmInterval = null;
        }
        try {
            if (window.activeSirenOsc) {
                window.activeSirenOsc.stop();
                window.activeSirenOsc.disconnect();
                window.activeSirenOsc = null;
            }
            if (window.activeSirenGain) {
                window.activeSirenGain.disconnect();
                window.activeSirenGain = null;
            }
        } catch (e) {
            console.error("Error stopping audio generator:", e);
        }
    }

    // Handle live stream frames from camera node
    function handleCameraFrame(message) {
        if (!liveStreamView || !livePlaceholder) return;
        
        // Show stream image and hide offline placeholder
        liveStreamView.src = message.frame;
        liveStreamView.style.display = "block";
        livePlaceholder.style.display = "none";
        
        // Update Live Status Indicators
        if (liveIndicatorDot && liveIndicatorText) {
            liveIndicatorDot.style.display = "inline-block";
            liveIndicatorDot.className = message.status === "alert" ? "status-dot red" : "status-dot green";
            
            liveIndicatorText.innerText = `Live: ${message.camera_name} (${message.status.toUpperCase()})`;
            liveIndicatorText.style.color = message.status === "alert" ? "var(--color-fire)" : "var(--color-safe)";
        }
        
        // Reset connection timeout (shows offline if no frames for 4 seconds)
        if (streamTimeout) clearTimeout(streamTimeout);
        streamTimeout = setTimeout(() => {
            liveStreamView.style.display = "none";
            livePlaceholder.style.display = "block";
            if (liveIndicatorDot && liveIndicatorText) {
                liveIndicatorDot.style.display = "none";
                liveIndicatorText.innerText = "Offline";
                liveIndicatorText.style.color = "var(--text-secondary)";
            }
        }, 4000);
    }

    // Handle incoming alerts
    function handleNewAlert(alert) {
        addAlertToFeed(alert, true); // Append to top and sound alarm
        updateStats();

        // Show Modal Popup with Snapshot
        modalCamName.innerText = alert.camera_name;
        modalLabel.innerText = alert.label;
        modalLabel.className = `log-badge ${alert.label.toLowerCase()}`;
        modalConfidence.innerText = `${(alert.confidence * 100).toFixed(1)}%`;
        
        if (alert.image_base64) {
            modalSnapshot.src = alert.image_base64;
            modalSnapshot.style.display = "block";
        } else {
            modalSnapshot.style.display = "none";
        }

        alertModal.className = "modal-visible";

        // Sound alarm and flash overlay
        if (!isMuted) {
            startSirenSound();
        }
        alertOverlay.className = "alert-overlay-active";
        
        // Update threat panel
        statusCard.className = "card card-status status-alert";
        statusIcon.className = "fa-solid fa-triangle-exclamation";
        threatLevelText.innerText = "WARNING: THREAT DETECTED";
        threatDesc.innerText = `Active ${alert.label} detected by ${alert.camera_name}. Take precautions.`;
    }

    function addAlertToFeed(alert, isNew = false) {
        if (noAlertsMsg) {
            noAlertsMsg.style.display = "none";
        }

        // Add sensor to set of unique active sensors
        activeSensors.add(alert.camera_name);

        const card = document.createElement("div");
        card.className = `card alert-log-card ${alert.label.toLowerCase()}`;
        card.dataset.id = alert.id;

        // Image thumbnail
        let imgHtml = `<div class="log-thumbnail-wrapper"><i class="fa-solid fa-image-portrait" style="font-size:24px; color:var(--text-secondary); margin: 20px;"></i></div>`;
        if (alert.image_base64) {
            imgHtml = `
                <div class="log-thumbnail-wrapper">
                    <img src="${alert.image_base64}" class="log-thumbnail" alt="Incident Snapshot" onclick="viewSnapshot('${alert.image_base64}', '${alert.camera_name}')"/>
                </div>
            `;
        }

        card.innerHTML = `
            ${imgHtml}
            <div class="log-info-box">
                <div class="log-title-row">
                    <span class="log-title">${alert.camera_name}</span>
                    <span class="log-badge ${alert.label.toLowerCase()}">${alert.label}</span>
                </div>
                <div class="log-meta-row">
                    <span><i class="fa-regular fa-clock"></i> ${alert.timestamp}</span>
                    <span><strong>Confidence:</strong> <span class="log-confidence">${(alert.confidence * 100).toFixed(1)}%</span></span>
                </div>
            </div>
            <div class="log-actions">
                <button class="btn btn-secondary btn-sm" onclick="viewSnapshot('${alert.image_base64}', '${alert.camera_name}')">
                    <i class="fa-solid fa-eye"></i> View
                </button>
            </div>
        `;

        if (isNew) {
            alertsFeed.insertBefore(card, alertsFeed.firstChild);
        } else {
            alertsFeed.appendChild(card);
        }
    }

    // Helper functions for updating stats on UI
    function updateStats() {
        const loggedCards = alertsFeed.querySelectorAll(".alert-log-card");
        statAlerts.innerText = loggedCards.length;
        logCount.innerText = `${loggedCards.length} Events`;
        statSensors.innerText = activeSensors.size;

        if (loggedCards.length === 0) {
            if (noAlertsMsg) noAlertsMsg.style.display = "flex";
            
            // Set secure status
            statusCard.className = "card card-status status-safe";
            statusIcon.className = "fa-solid fa-shield-halved";
            threatLevelText.innerText = "SYSTEM SECURE";
            threatDesc.innerText = "No active anomalies detected across connected camera sensors.";
        }
    }

    function clearFeedUI() {
        // Remove all log cards except the placeholder message
        const cards = alertsFeed.querySelectorAll(".alert-log-card");
        cards.forEach(card => card.remove());
        activeSensors.clear();
        updateStats();
    }

    function clearLocalHistory() {
        clearFeedUI();
        stopSirenSound();
        alertOverlay.className = "alert-overlay-hidden";
        alertModal.className = "modal-hidden";
    }

    // Clear backend database via API
    async function clearDatabase() {
        if (!confirm("Are you sure you want to clear all alerts and logs history?")) return;
        try {
            await fetch("/api/alerts", { method: "DELETE" });
            clearLocalHistory();
        } catch (e) {
            console.error("Failed to clear alert database:", e);
        }
    }

    // UI event listeners
    clearBtn.addEventListener("click", clearDatabase);

    sirenToggleBtn.addEventListener("click", () => {
        isMuted = !isMuted;
        if (isMuted) {
            stopSirenSound();
            sirenToggleBtn.innerHTML = `<i class="fa-solid fa-volume-xmark"></i> Unmute Alarm`;
            sirenToggleBtn.className = "btn btn-secondary";
        } else {
            sirenToggleBtn.innerHTML = `<i class="fa-solid fa-volume-high"></i> Mute Alarm`;
            sirenToggleBtn.className = "btn btn-secondary";
            // If threat is currently active, start playing
            const activeAlerts = alertsFeed.querySelectorAll(".alert-log-card").length;
            if (activeAlerts > 0 && alertOverlay.className === "alert-overlay-active") {
                startSirenSound();
            }
        }
    });

    modalDismissBtn.addEventListener("click", () => {
        alertModal.className = "modal-hidden";
        stopSirenSound();
        alertOverlay.className = "alert-overlay-hidden";
    });

    // Make viewSnapshot globally accessible for onclick events
    window.viewSnapshot = (base64Img, camName) => {
        if (!base64Img) return;
        modalCamName.innerText = camName;
        modalLabel.innerText = "Historical Log View";
        modalLabel.className = "badge";
        modalConfidence.innerText = "N/A";
        modalSnapshot.src = base64Img;
        modalSnapshot.style.display = "block";
        alertModal.className = "modal-visible";
    };

    // Load initial configuration
    loadAlertHistory();
    connectWebSocket();

    // User Session / Login Checks
    const savedUser = JSON.parse(localStorage.getItem("user"));
    if (!savedUser) {
        loginOverlay.className = ""; // Show modal overlay by removing hidden class
    } else {
        showUserProfile(savedUser);
    }

    function showUserProfile(user) {
        profileName.innerText = user.name;
        profileEmail.innerText = user.email;
        userProfile.style.display = "block";
    }

    // Login Form Submit Event
    if (loginForm) {
        loginForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const name = loginName.value.trim();
            const email = loginEmail.value.trim();

            if (!name || !email) return;

            try {
                const response = await fetch("/api/login", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ name, email })
                });
                const resData = await response.json();
                if (resData.status === "success") {
                    localStorage.setItem("user", JSON.stringify({ name, email }));
                    loginOverlay.className = "login-overlay-hidden";
                    showUserProfile({ name, email });
                    window.location.reload(); // Reload to refresh subscribers counter state
                }
            } catch (err) {
                console.error("Login endpoint failed:", err);
                alert("Subscription system offline. Detections will run locally, but email alerts are disabled.");
                loginOverlay.className = "login-overlay-hidden"; // allow access anyway
            }
        });
    }

    // Logout Click Event
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
            window.location.reload();
        });
    }

    // Ping WebSocket every 30 seconds to keep connection alive in browser
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send("ping");
        }
    }, 30000);
});

