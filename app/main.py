# --- Pi5 Face Stream (green box) + Face Crop to RabbitMQ (no-freeze) ---
import time, json, base64, threading, zlib, queue
from datetime import datetime, timezone
from collections import deque
from typing import Optional, List

import logging
import onnxruntime as ort
import numpy as np
import cv2
import pika
from fastapi import FastAPI, WebSocket, WebSocketDisconnect 
from fastapi.responses import StreamingResponse
import asyncio  
from picamera2 import Picamera2

from app.config import *



# ---------------- RabbitMQ ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MQPublisher:
    def __init__(self, amqp_url: str, queue_name: str):
        self.amqp_url = amqp_url
        self.queue_name = queue_name
        self.connection = None
        self.channel = None
        self._connect()

    def _connect(self):
        """เชื่อมต่อ RabbitMQ และประกาศคิว durable"""
        params = pika.URLParameters(self.amqp_url)
        params.heartbeat = 30
        params.blocked_connection_timeout = 60

        while True:
            try:
                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()

                # ✅ ให้ queue durable จริง (ต้องตรงกับ backend)
                self.channel.queue_declare(queue=self.queue_name, durable=True)
                logging.info(f"✅ Connected to RabbitMQ (queue={self.queue_name}, durable=True)")
                break

            except Exception as e:
                logging.warning(f"❌ RabbitMQ connect failed: {e}, retrying...")
                time.sleep(3)

    def publish(self, data, content_type="application/json"):
        """ส่งข้อความไปที่คิวแบบ durable"""
        try:
            # ถ้าการเชื่อมต่อหลุด ให้ reconnect อัตโนมัติ
            if self.connection is None or self.connection.is_closed:
                self._connect()

            # ✅ ส่ง message แบบ durable (ไม่หายถ้า RabbitMQ รีสตาร์ต)
            self.channel.basic_publish(
                exchange='',
                routing_key=self.queue_name,
                body=data,
                properties=pika.BasicProperties(
                    content_type=content_type,
                    delivery_mode=2  # ✅ Durable message (เก็บบน disk)
                )
            )

            logging.info(f"📤 Published message to queue '{self.queue_name}' ({len(data)} bytes)")

        except Exception as e:
            logging.error(f"❌ MQ publish error: {e}")
            time.sleep(2)
            self._connect()


publisher = MQPublisher(AMQP_URL, QUEUE_NAME)
logging.info("🚀 MQPublisher initialized and ready")

# ------------- App & Shared State -------------
app = FastAPI(title="Pi5 Face Stream + Face Publisher")

FRAME_BUFFER = deque(maxlen=3)
_preview_jpeg = b""
_state_lock  = threading.Lock()
STOP = False
mq_outbox: "queue.Queue[bytes]" = queue.Queue(maxsize=200)

# ----------------- CV Utils -----------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def crop_with_margin(img: np.ndarray, x: int, y: int, w: int, h: int, margin: float = 0.2) -> np.ndarray:
    H, W = img.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    return img[y0:y1, x0:x1]

def encode_jpeg(img: np.ndarray, q: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return buf.tobytes() if ok else b""

def make_backend_message(camera_id: str, jpeg_bytes: bytes) -> bytes:
    comp = zlib.compress(jpeg_bytes)
    b64 = base64.b64encode(comp).decode("ascii")
    return json.dumps({"camera_id": camera_id, "image": b64}, ensure_ascii=False).encode("utf-8")

def is_face_quality_ok(face_crop, blur_thresh=100.0):
    """คืนค่า False ถ้าใบหน้าเบลอมากเกินไป"""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness >= blur_thresh

recent_faces = []  # เก็บ (x,y,w,h,timestamp)
DEDUP_TIME = 3.0   # วินาที
IOU_THRESHOLD = 0.5  # ซ้ำกันถ้า IOU > 0.5

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def is_duplicate_face(x1, y1, x2, y2):
    now = time.time()
    global recent_faces
    recent_faces = [f for f in recent_faces if now - f[4] < DEDUP_TIME]
    for fx1, fy1, fx2, fy2, ft in recent_faces:
        if iou((x1, y1, x2, y2), (fx1, fy1, fx2, fy2)) > IOU_THRESHOLD:
            return True
    recent_faces.append((x1, y1, x2, y2, now))
    return False

# ---------------- Threads ----------------
def camera_loop():
    global STOP
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"})
    picam2.configure(cfg)
    picam2.start()
    print("[camera] RGB888 started")

    try:
        while not STOP:
            frame = picam2.capture_array()
            with _state_lock:
                FRAME_BUFFER.append(frame)
            time.sleep(0.005)
    finally:
        picam2.stop()

print("🔍 Checking model output shape...")
for i, out in enumerate(session.get_outputs()):
    print(f"Output {i}: name={out.name}, shape={out.shape}")

face_buffer = []
WINDOW_TIME = 1.0
last_send_time = 0.0

def detect_loop():
    global _preview_jpeg, last_send_time
    interval = max(40, DETECT_INTERVAL_MS) / 1000.0
    conf_thresh = 0.4
    iou_thresh = 0.45

    print("[INFO] YOLO-face decode (Best-of-Window mode) started ✅")

    def nms(boxes, scores, iou_threshold=0.45):
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes)
        scores = np.array(scores)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    while not STOP:
        with _state_lock:
            frame = FRAME_BUFFER[-1].copy() if FRAME_BUFFER else None
        if frame is None:
            time.sleep(0.01)
            continue

        H, W = frame.shape[:2]
        img = cv2.resize(frame, (640, 640))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(np.transpose(img_rgb, (2, 0, 1)), 0).astype(np.float32) / 255.0

        pred = session.run(None, {"images": tensor})[0]
        pred = pred.squeeze(0).T  # (8400,20)

        x, y, w, h, conf = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]
        mask = conf > conf_thresh
        x, y, w, h, conf = x[mask], y[mask], w[mask], h[mask], conf[mask]

        boxes, scores = [], []
        for i in range(len(x)):
            x1 = int((x[i] - w[i] / 2) * W / 640)
            y1 = int((y[i] - h[i] / 2) * H / 640)
            x2 = int((x[i] + w[i] / 2) * W / 640)
            y2 = int((y[i] + h[i] / 2) * H / 640)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))
            scores.append(float(conf[i]))

        keep = nms(boxes, scores, iou_threshold=iou_thresh)
        frame_draw = frame.copy()

        if len(keep) > 0:
            print(f"[INFO] ✅ Detected {len(keep)} faces")
            for i in keep:
                x1, y1, x2, y2 = map(int, boxes[i])
                face_crop = crop_with_margin(frame, x1, y1, x2 - x1, y2 - y1, FACE_MARGIN_RATIO)
                if face_crop is None or face_crop.size == 0:
                    continue
                face_crop = cv2.resize(face_crop, (256, 256))

                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                jpeg_bytes = encode_jpeg(face_crop, JPEG_QUALITY)
                face_buffer.append((sharpness, time.time(), jpeg_bytes))

                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame_draw, f"{sharpness:.1f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        else:
            print("[INFO] ❌ No face detected")

        # --- ตรวจเวลาใน window ---
        now = time.time()
        if (now - last_send_time) >= WINDOW_TIME and len(face_buffer) > 0:
            best_face = max(face_buffer, key=lambda f: f[0])
            sharp, ts, jpeg = best_face

            if sharp < 20:
                print(f"[INFO] ⚠️ Skipped low-quality face (sharp={sharp:.1f})")
                face_buffer.clear()
                last_send_time = now   # ✅ ปรับตรงนี้
                continue

            payload = make_backend_message(DEVICE_ID, jpeg)
            try:
                mq_outbox.put_nowait(payload)
                print(f"[INFO] 🚀 Sent best face (sharp={sharp:.1f})")
            except queue.Full:
                pass

            face_buffer.clear()
            last_send_time = now

        with _state_lock:
            _preview_jpeg = encode_jpeg(frame_draw, JPEG_QUALITY)




def mq_worker_loop():
    global publisher  # ✅ บอก Python ว่าใช้ตัวแปร publisher ที่ประกาศไว้ข้างนอก
    while not STOP:
        try:
            payload = mq_outbox.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            publisher.publish(payload, content_type="application/json")
            logging.info("🚀 Face sent to RabbitMQ successfully")
        except Exception as e:
            logging.error(f"❌ MQ publish error: {e}")
            time.sleep(1)


# --------------- MJPEG streaming ---------------
def mjpeg_stream(get_jpeg_fn, boundary="frame", fps=15):
    min_interval = 1.0 / max(fps, 1)
    while True:
        start = time.time()
        jpg = get_jpeg_fn()
        if not jpg:
            black = np.zeros((200,200,3), dtype=np.uint8)
            jpg = encode_jpeg(black, 60)
        try:
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
            )
        except Exception:
            break
        elapsed = time.time() - start
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

def get_preview_jpeg() -> bytes:
    with _state_lock:
        data = _preview_jpeg
        fallback = FRAME_BUFFER[-1] if FRAME_BUFFER else None
    if data:
        return data
    if fallback is not None:
        return encode_jpeg(fallback, JPEG_QUALITY)
    return b""

# ----------------- API -----------------
@app.get("/")
def root():
    return {"ok": True, "device_id": DEVICE_ID, "stream": "/stream", "ts": now_iso()}

@app.get("/stream")
def stream():
    return StreamingResponse(
        mjpeg_stream(get_preview_jpeg, fps=15),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.websocket("/video_feed/{camera_id}")
async def video_feed(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    print(f"[WS] connected: {camera_id}")

    if camera_id != DEVICE_ID:
        await websocket.send_text("Invalid camera_id")
        await websocket.close()
        print(f"[WS] rejected: invalid camera_id {camera_id}")
        return

    try:
        while True:
            jpg = get_preview_jpeg()
            if not jpg:
                await asyncio.sleep(0.05)
                continue
            frame_base64 = base64.b64encode(jpg).decode("utf-8")
            await websocket.send_text(frame_base64)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        print(f"[WS] disconnected: {camera_id}")
    except Exception as e:
        print(f"[WS] error: {e}")

# -------------- Boot Threads --------------
def start_background():
    t1 = threading.Thread(target=camera_loop, daemon=True)
    t2 = threading.Thread(target=detect_loop, daemon=True)
    t3 = threading.Thread(target=mq_worker_loop, daemon=True)
    t1.start(); t2.start(); t3.start()

start_background()