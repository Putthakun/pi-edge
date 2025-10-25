# --- Pi5 Face Stream (green box) + Face Crop to RabbitMQ (no-freeze) ---
import os, time, json, base64, threading, zlib, queue
from datetime import datetime, timezone
from collections import deque
from typing import Optional, List

import onnxruntime as ort
import numpy as np
import cv2
import pika
from fastapi import FastAPI, WebSocket, WebSocketDisconnect 
from fastapi.responses import StreamingResponse
import asyncio  
from dotenv import load_dotenv
from pathlib import Path
from picamera2 import Picamera2

# ---------------- Config ----------------
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

SAVE_DIR = os.getenv("SAVE_DIR", "/tmp/pi_face_crops")
SAVE_EVERY_N = int(os.getenv("SAVE_EVERY_N", "1"))
os.makedirs(SAVE_DIR, exist_ok=True)
_save_counter = 0

CAM_WIDTH  = int(os.getenv("CAM_WIDTH", "1280"))
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", "720"))
DETECT_INTERVAL_MS = int(os.getenv("DETECT_INTERVAL_MS", "80"))
FACE_MARGIN_RATIO  = float(os.getenv("FACE_MARGIN_RATIO", "0.20"))
JPEG_QUALITY       = int(os.getenv("JPEG_QUALITY", "80"))
MAX_IMAGE_BYTES    = int(os.getenv("MAX_IMAGE_BYTES", str(512*1024)))
DEVICE_ID          = os.getenv("DEVICE_ID", "pi5-unknown")

AMQP_URL    = os.getenv("AMQP_URL")
EXCHANGE    = (os.getenv("EXCHANGE") or "").strip()
ROUTING_KEY = os.getenv("ROUTING_KEY", "face_images")
QUEUE_NAME  = os.getenv("QUEUE", "face_images")

HTTP_HOST   = os.getenv("HTTP_HOST", "0.0.0.0")
HTTP_PORT   = int(os.getenv("HTTP_PORT", "8000"))

session = ort.InferenceSession("face_final.onnx", providers=["CPUExecutionProvider"])

if not AMQP_URL:
    raise RuntimeError("AMQP_URL missing")

# ---------------- RabbitMQ ----------------
class MQPublisher:
    def __init__(self, url: str, exchange: str, routing_key: str, queue_name: Optional[str] = None):
        self._url=url; self._exchange=exchange; self._routing_key=routing_key; self._queue=queue_name
        self._conn=None; self._ch=None

    def _connect(self):
        p=pika.URLParameters(self._url); p.heartbeat=30; p.blocked_connection_timeout=60
        self._conn=pika.BlockingConnection(p); self._ch=self._conn.channel()
        if self._queue: self._ch.queue_declare(queue=self._queue, durable=True)
        if self._exchange:
            self._ch.exchange_declare(exchange=self._exchange, exchange_type='direct', durable=True)
            if self._queue: self._ch.queue_bind(queue=self._queue, exchange=self._exchange, routing_key=self._routing_key)

    def publish(self, body: bytes, content_type: str="application/json"):
        backoff=1.0
        while True:
            try:
                if (self._conn is None) or self._conn.is_closed or (self._ch is None) or self._ch.is_closed:
                    self._connect()
                props=pika.BasicProperties(content_type=content_type, delivery_mode=2)
                rk=self._routing_key if self._exchange else (self._queue or self._routing_key)
                self._ch.basic_publish(exchange=self._exchange if self._exchange else "", routing_key=rk, body=body, properties=props)
                return
            except Exception:
                try:
                    if self._conn: self._conn.close()
                except Exception: pass
                time.sleep(min(backoff,10)); backoff*=2

publisher = MQPublisher(AMQP_URL, EXCHANGE, ROUTING_KEY, QUEUE_NAME)

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

def detect_loop():
    global _preview_jpeg
    interval = max(40, DETECT_INTERVAL_MS) / 1000.0
    last = 0.0
    conf_thresh = 0.4
    iou_thresh = 0.45

    print("[INFO] YOLO-face decode (direct pixel output) started ✅")

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

        now = time.time()
        if (now - last) < interval:
            time.sleep(0.002)
            continue
        last = now

        H, W = frame.shape[:2]
        img = cv2.resize(frame, (640, 640))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(np.transpose(img_rgb, (2, 0, 1)), 0).astype(np.float32) / 255.0

        pred = session.run(None, {"images": tensor})[0]  # (1,20,8400)
        pred = pred.squeeze(0).T  # (8400,20)

        # ใช้ค่า x,y,w,h และ conf โดยตรง
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
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            print("[INFO] ❌ No face detected")

        with _state_lock:
            _preview_jpeg = encode_jpeg(frame_draw, JPEG_QUALITY)



def mq_worker_loop():
    while not STOP:
        try:
            payload = mq_outbox.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            publisher.publish(payload, content_type="application/json")
        except Exception:
            pass

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