import os
import json
import base64
import smtplib
import logging
import time
import asyncio
from typing import List, Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import cv2
import numpy as np
import onnxruntime as ort


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Smoke and Fire Detection Hub")

# In-memory database of alerts
alerts_db: List[Dict] = []
db_file_path = "alerts_history.json"

# Load history from local file if exists

if os.path.exists(db_file_path):
    try:
        with open(db_file_path, "r", encoding="utf-8") as f:
            alerts_db = json.load(f)
        logger.info(f"Loaded {len(alerts_db)} alerts from history.")
    except Exception as e:
        logger.error(f"Error loading alert history: {e}")

# Email configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")  # Gmail App Password
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", "")

# In-memory database of active subscribers (persistent)
users_db: List[Dict] = []
users_file_path = "users_history.json"

if os.path.exists(users_file_path):
    try:
        with open(users_file_path, "r", encoding="utf-8") as f:
            users_db = json.load(f)
        logger.info(f"Loaded {len(users_db)} registered email subscribers.")
    except Exception as e:
        logger.error(f"Error loading subscribers: {e}")


# Active WebSocket connections (Dashboard Nodes)
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Dashboard client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Dashboard client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message to client: {e}")

manager = ConnectionManager()

# Global tracking for camera detections
camera_consecutive_detections = {}
camera_last_alert_time = {}

# Load ONNX model session
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "best.onnx")
session = None
if os.path.exists(model_path):
    try:
        # Load ONNX session on CPU (suitable for Render free tier)
        session = ort.InferenceSession(model_path)
        logger.info(f"✅ ONNX model loaded successfully from: {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load ONNX model: {e}")
else:
    logger.warning(f"⚠️ ONNX model file not found at {model_path}. Detections will be skipped.")

# YOLOv8 ONNX Inference Helper
def run_onnx_inference(image_base64: str, threshold: float = 0.75):
    if session is None:
        return image_base64, "Neutral", 0.0, False

    try:
        # Decode base64 image data URI
        if "," in image_base64:
            header, encoded = image_base64.split(",", 1)
        else:
            encoded = image_base64
            
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_base64, "Neutral", 0.0, False
    except Exception as e:
        logger.error(f"Failed to decode base64 stream image: {e}")
        return image_base64, "Neutral", 0.0, False

    orig_h, orig_w = img.shape[:2]

    # Pre-processing for YOLOv8 (640x640 size, BCHW shape, normalized 0-1)
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = img_rgb.transpose(2, 0, 1)  # HWC to CHW
    img_expanded = np.expand_dims(img_transposed, axis=0)  # CHW to BCHW
    img_input = img_expanded.astype(np.float32) / 255.0  # Normalize to [0, 1]

    try:
        # Run inference session
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_input})
        
        # Outputs shape is typically [1, 6, 8400] for 2 classes
        predictions = outputs[0][0].T  # Transpose to shape [8400, 6]
        
        # Classes: 0 -> smoke, 1 -> fire
        classes_labels = {0: "Smoke", 1: "Fire"}
        
        boxes = []
        confidences = []
        class_ids = []
        
        x_scale = orig_w / 640.0
        y_scale = orig_h / 640.0
        
        for pred in predictions:
            smoke_score = float(pred[4])
            fire_score = float(pred[5])
            
            max_score = smoke_score
            class_id = 0
            if fire_score > smoke_score:
                max_score = fire_score
                class_id = 1
                
            if max_score >= threshold:
                cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
                
                # Convert center xywh to top-left xywh
                x = int((cx - w / 2) * x_scale)
                y = int((cy - h / 2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                
                boxes.append([x, y, width, height])
                confidences.append(max_score)
                class_ids.append(class_id)
                
        detected = False
        max_conf = 0.0
        detected_label = "Neutral"
        
        if len(boxes) > 0:
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.45)
            if len(indices) > 0:
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                    
                for idx in indices:
                    box = boxes[idx]
                    conf = confidences[idx]
                    cid = class_ids[idx]
                    label = classes_labels[cid]
                    
                    detected = True
                    if conf > max_conf:
                        max_conf = conf
                        detected_label = label
                        
                    # Draw bounding box (Calming terracotta for Fire, Sand-gold for Smoke)
                    color = (90, 107, 217) if cid == 1 else (107, 176, 224)
                    x, y, w, h = box
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                    
                    # Draw label text
                    text = f"{label}: {conf:.1%}"
                    cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
        if detected:
            # Encode image back to base64
            _, buffer = cv2.imencode('.jpg', img)
            encoded_img = base64.b64encode(buffer).decode('utf-8')
            annotated_base64 = f"data:image/jpeg;base64,{encoded_img}"
            return annotated_base64, detected_label, max_conf, True
            
    except Exception as e:
        logger.error(f"Inference processing failed: {e}")
        
    return image_base64, "Neutral", 0.0, False


# Input validation models
class AlertPayload(BaseModel):
    camera_name: str
    label: str  # "Fire" or "Smoke"
    confidence: float
    image_base64: Optional[str] = None  # Base64 image data URI

class UserLogin(BaseModel):
    name: str
    email: str

class UserLogout(BaseModel):
    email: str


# Email Sender Helper
def send_email_alert(camera_name: str, label: str, confidence: float, image_base64: Optional[str]):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        logger.warning("Email notifications are skipped because SMTP credentials (SENDER_EMAIL, SENDER_PASSWORD) are not set in .env")
        return

    # Determine recipient list
    recipients = [u for u in users_db if u.get("email")]
    if not recipients:
        if RECEIVER_EMAIL:
            recipients = [{"name": "System Admin", "email": RECEIVER_EMAIL}]
            logger.info("No registered subscribers. Falling back to default RECEIVER_EMAIL.")
        else:
            logger.info("No subscribers logged in and no RECEIVER_EMAIL configured. Skipping email alert.")
            return

    try:
        # Import platform for robust date/time representation
        import platform
        if platform.system() == 'Windows':
            timestamp = os.popen('date /t').read().strip() + " " + os.popen('time /t').read().strip()
        else:
            timestamp = os.popen('date').read().strip()

        logger.info(f"Connecting to SMTP server {SMTP_SERVER}:{SMTP_PORT}...")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        for user in recipients:
            recipient_email = user["email"]
            recipient_name = user["name"]
            
            subject = f"⚠️ CRITICAL ALERT: {label.upper()} Detected by {camera_name}!"
            
            # Create MIMEMultipart message
            msg = MIMEMultipart('related')
            msg['Subject'] = subject
            msg['From'] = SENDER_EMAIL
            msg['To'] = recipient_email

            # Plain text and HTML parts
            body_html = f"""
            <html>
            <head>
                <style>
                    .container {{ font-family: Arial, sans-serif; padding: 20px; border: 1px solid #ff4d4d; border-radius: 8px; max-width: 600px; }}
                    .header {{ background-color: #ff4d4d; color: white; padding: 10px; text-align: center; border-radius: 6px 6px 0 0; font-size: 20px; font-weight: bold; }}
                    .content {{ padding: 20px; line-height: 1.6; color: #333; }}
                    .details {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 5px solid #ff4d4d; }}
                    .footer {{ font-size: 12px; color: #777; text-align: center; margin-top: 20px; }}
                    .image-box {{ text-align: center; margin: 20px 0; }}
                    .image-box img {{ max-width: 100%; border: 3px solid #ff4d4d; border-radius: 6px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">🔥 {label.upper()} DETECTED 🔥</div>
                    <div class="content">
                        <p>Hello {recipient_name},</p>
                        <p>This is an automated safety alert from your <b>Smoke & Fire Detection App</b>. The system has detected a potential emergency in your monitored space.</p>
                        
                        <div class="details">
                            <b>Camera Sensor:</b> {camera_name}<br/>
                            <b>Event Type:</b> {label}<br/>
                            <b>Confidence Level:</b> {confidence:.2%}<br/>
                            <b>Timestamp:</b> {timestamp}
                        </div>
            """
            
            # Attach snapshot if present
            img_attached = False
            img_data = None
            if image_base64 and "," in image_base64:
                try:
                    # Extract base64 payload from data URI scheme
                    header, encoded = image_base64.split(",", 1)
                    img_data = base64.b64decode(encoded)
                    
                    body_html += """
                        <div class="image-box">
                            <p><b>Live Camera Snapshot:</b></p>
                            <img src="cid:snapshot" alt="Camera Snapshot"/>
                        </div>
                    """
                    img_attached = True
                except Exception as ex:
                    logger.error(f"Failed to decode base64 image for email attachment: {ex}")

            body_html += """
                        <p>Please check your central dashboard or investigate immediately.</p>
                    </div>
                    <div class="footer">
                        Smoke & Fire Detection System © 2026. This email was sent automatically to active dashboard subscribers.
                    </div>
                </div>
            </body>
            </html>
            """

            msg.attach(MIMEText(body_html, 'html'))

            # Attach image content-id matching cid:snapshot
            if img_attached and img_data:
                mime_img = MIMEImage(img_data)
                mime_img.add_header('Content-ID', '<snapshot>')
                mime_img.add_header('Content-Disposition', 'inline', filename="snapshot.jpg")
                msg.attach(mime_img)

            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
            logger.info(f"✅ Email alert sent successfully to {recipient_name} ({recipient_email})!")
            
        server.quit()
        logger.info("Finished sending alert email loop.")
    except Exception as e:
        logger.error(f"❌ Failed to send email alert loop: {e}")


# API Endpoints
@app.post("/api/alert")
async def receive_alert(payload: AlertPayload, background_tasks: BackgroundTasks):
    logger.info(f"Alert received: {payload.label} from {payload.camera_name} (Confidence: {payload.confidence})")
    
    import datetime
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    alert_event = {
        "id": len(alerts_db) + 1,
        "camera_name": payload.camera_name,
        "label": payload.label,
        "confidence": payload.confidence,
        "timestamp": timestamp_str,
        # Keep base64 photo for UI display, limit size if needed but in-memory is fine
        "image_base64": payload.image_base64
    }
    
    # Save to history db
    alerts_db.insert(0, alert_event)  # Store newest first
    
    # Keep only the last 100 alerts in history to save memory
    if len(alerts_db) > 100:
        alerts_db.pop()
        
    # Persist to local JSON file (exclude base64 images to avoid massive file sizes, or keep them. Let's keep them so users can load images on reload)
    try:
        with open(db_file_path, "w", encoding="utf-8") as f:
            json.dump(alerts_db, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to persist alert to file: {e}")

    # Broadcast to all active dashboards
    await manager.broadcast({
        "type": "NEW_ALERT",
        "data": alert_event
    })
    
    # Send email notification in background task so API responds immediately
    background_tasks.add_task(
        send_email_alert, 
        payload.camera_name, 
        payload.label, 
        payload.confidence, 
        payload.image_base64
    )
    
    return {"status": "success", "alert_id": alert_event["id"]}

@app.get("/api/alerts")
async def get_alerts():
    # Return alert history
    return alerts_db

@app.delete("/api/alerts")
async def clear_alerts():
    global alerts_db
    alerts_db = []
    if os.path.exists(db_file_path):
        os.remove(db_file_path)
    logger.info("Cleared all alerts history.")
    await manager.broadcast({"type": "ALERTS_CLEARED"})
    return {"status": "success"}

@app.post("/api/login")
async def login(user: UserLogin):
    email_lower = user.email.strip().lower()
    
    # Check if already registered
    exists = any(u["email"].lower() == email_lower for u in users_db)
    if not exists:
        users_db.append({
            "name": user.name.strip(),
            "email": email_lower
        })
        try:
            with open(users_file_path, "w", encoding="utf-8") as f:
                json.dump(users_db, f, indent=4)
            logger.info(f"Registered subscriber: {user.name} ({email_lower})")
        except Exception as e:
            logger.error(f"Failed to persist subscribers list: {e}")
            
    await manager.broadcast({
        "type": "SUBSCRIBERS_UPDATED",
        "data": len(users_db)
    })
    
    return {"status": "success", "user": {"name": user.name, "email": email_lower}}

@app.post("/api/logout")
async def logout(payload: UserLogout):
    global users_db
    email_lower = payload.email.strip().lower()
    
    initial_count = len(users_db)
    users_db = [u for u in users_db if u["email"].lower() != email_lower]
    
    if len(users_db) < initial_count:
        try:
            with open(users_file_path, "w", encoding="utf-8") as f:
                json.dump(users_db, f, indent=4)
            logger.info(f"Unregistered subscriber: {email_lower}")
        except Exception as e:
            logger.error(f"Failed to persist subscribers list: {e}")
            
    await manager.broadcast({
        "type": "SUBSCRIBERS_UPDATED",
        "data": len(users_db)
    })
    
    return {"status": "success"}

@app.get("/api/users")
async def get_users():
    return users_db


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial subscribers count upon connection
        await websocket.send_json({
            "type": "SUBSCRIBERS_UPDATED",
            "data": len(users_db)
        })
        
        while True:
            # Keep connection alive by waiting for client messages (if any)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
                continue
                
            try:
                message = json.loads(data)
                
                # Check if it is a live video frame from the camera node
                if message.get("type") == "CAMERA_FRAME":
                    camera_name = message.get("camera_name", "Remote Cam")
                    frame = message.get("frame")
                    threshold = float(message.get("threshold", 0.75))
                    
                    if frame:
                        # 1. Run ONNX object detection on server
                        annotated_frame, label, confidence, is_detected = run_onnx_inference(frame, threshold)
                        
                        # 2. Broadcast video frame to all connected dashboards
                        await manager.broadcast({
                            "type": "CAMERA_FRAME",
                            "camera_name": camera_name,
                            "frame": annotated_frame,
                            "status": "alert" if is_detected else "safe"
                        })
                        
                        # 3. Handle Alert Trigger logic
                        if is_detected:
                            camera_consecutive_detections[camera_name] = camera_consecutive_detections.get(camera_name, 0) + 1
                            if camera_consecutive_detections[camera_name] >= 2:
                                camera_consecutive_detections[camera_name] = 0
                                
                                # Throttle check (limit to 1 alert per 15 seconds)
                                now = time.time()
                                last_alert = camera_last_alert_time.get(camera_name, 0)
                                if now - last_alert >= 15:
                                    camera_last_alert_time[camera_name] = now
                                    
                                    # Create alert record
                                    import datetime
                                    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    
                                    alert_event = {
                                        "id": len(alerts_db) + 1,
                                        "camera_name": camera_name,
                                        "label": label,
                                        "confidence": confidence,
                                        "timestamp": timestamp_str,
                                        "image_base64": annotated_frame
                                    }
                                    
                                    alerts_db.insert(0, alert_event)
                                    if len(alerts_db) > 100:
                                        alerts_db.pop()
                                        
                                    try:
                                        with open(db_file_path, "w", encoding="utf-8") as f:
                                            json.dump(alerts_db, f, indent=4)
                                    except Exception as e:
                                        logger.error(f"Failed to save alert event: {e}")
                                        
                                    # Broadcast alert metadata to play siren and flash overlay
                                    await manager.broadcast({
                                        "type": "NEW_ALERT",
                                        "data": alert_event
                                    })
                                    
                                    # Send email alerts to subscribers in background thread
                                    asyncio.create_task(
                                        asyncio.to_thread(
                                            send_email_alert, 
                                            camera_name, 
                                            label, 
                                            confidence, 
                                            annotated_frame
                                        )
                                    )
                        else:
                            camera_consecutive_detections[camera_name] = 0
                            
            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.error(f"Error processing WS message: {e}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        manager.disconnect(websocket)

# Serve Frontend Pages
# Setup static files directory
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "static")
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found")

@app.get("/camera")
async def read_camera():
    camera_path = os.path.join(frontend_dir, "camera.html")
    if os.path.exists(camera_path):
        return FileResponse(camera_path)
    raise HTTPException(status_code=404, detail="camera.html not found")
