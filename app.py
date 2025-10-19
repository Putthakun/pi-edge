# --- Pi5 Face Stream (green box) + Face Crop to RabbitMQ (no-freeze) ---
import os, time, json, base64, threading, zlib, queue
from datetime import datetime, timezone
from collections import deque
from typing import Optional, List

import numpy as np
import cv2
import pika
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from pathlib import Path
from picamera2 import Picamera2

# load_dotenv(override=True)---------------- Config ----------------
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

SAVE_DIR = os.getenv("SAVE_DIR", "/tmp/pi_face_crops")     # ??????????????? debug
SAVE_EVERY_N = int(os.getenv("SAVE_EVERY_N", "1"))         # ?????? N ?????? (1 = ?????????)
os.makedirs(SAVE_DIR, exist_ok=True)
_save_counter = 0

CAM_WIDTH  = int(os.getenv("CAM_WIDTH", "1280"))
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", "720"))
DETECT_INTERVAL_MS = int(os.getenv("DETECT_INTERVAL_MS", "80"))  # ????? 80ms ???????
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
        # ?? retry ??? block ??????????? (???????? detect/stream)
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

FRAME_BUFFER = deque(maxlen=3)     # ???????????????
_preview_jpeg = b""                # ?????? JPEG ?????? (???????? /stream)
_state_lock  = threading.Lock()    # ???????? ? ???????? reference

STOP = False

# ????????? "?????? MQ" ??????? (???????? detect)
mq_outbox: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
# ----------------- CV Utils -----------------
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
if CASCADE.empty():
    raise RuntimeError(f"Failed to load Haar cascade: {CASCADE_PATH}")

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
    cfg = picam2.create_video_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(cfg)
    picam2.start()
    print("[camera] RGB888 started")

    try:
        while not STOP:
            frame = picam2.capture_array()  # RGB ndarray
            # ?????????????? (??????? race ???? lock)
            with _state_lock:
                FRAME_BUFFER.append(frame)
            time.sleep(0.005)
    finally:
        picam2.stop()


def detect_loop():
    global _preview_jpeg, _save_counter
    interval = max(40, DETECT_INTERVAL_MS) / 1000.0
    last = 0.0

    while not STOP:
        # ?????????????
        with _state_lock:
            frame = FRAME_BUFFER[-1].copy() if FRAME_BUFFER else None
        if frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        if (now - last) >= interval:
            last = now

            # ???????: ??? (??? MQ) ???????????????? (???? /stream)
            frame_raw  = frame                      # ??? ? ???????/??? MQ
            frame_draw = frame.copy()               # ????????????????????????

            # ???????? + ??? ????? detect ????????
            small = cv2.resize(frame_raw, (0, 0), fx=0.5, fy=0.5)
            gray  = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

            # ?????????????
            faces = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                # scale ??????????????????
                X, Y, W, H = int(x * 2), int(y * 2), int(w * 2), int(h * 2)

                # ??????? "?????" ?? frame_draw ?????????????????
                cv2.rectangle(frame_draw, (X, Y), (X + W, Y + H), (0, 255, 0), 3)

                # ??????? "???????" (?????????) ??????? MQ
                face = crop_with_margin(frame_raw, X, Y, W, H, FACE_MARGIN_RATIO)
                if face is None or face.size == 0:
                    continue

                # ???????? JPEG ??????? (????????????? ??????????????)
                jpg = encode_jpeg(face, JPEG_QUALITY)
                if len(jpg) > MAX_IMAGE_BYTES:
                    jpg = encode_jpeg(face, max(50, JPEG_QUALITY - 20))
                    if len(jpg) > MAX_IMAGE_BYTES:
                        continue

                # (??????) ???????????????????????? MQ ????????
                _save_counter += 1
                if SAVE_EVERY_N > 0 and (_save_counter % SAVE_EVERY_N == 0):
                    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    fname = f"{DEVICE_ID}_{ts}_{X}-{Y}-{W}x{H}.jpg"
                    fpath = os.path.join(SAVE_DIR, fname)
                    try:
                        with open(fpath, "wb") as f:
                            f.write(jpg)  # ??????????????????????????? MQ 100%
                    except Exception as e:
                        print("[DEBUG SAVE ERROR]", e)

                # ????????? worker ??????? RabbitMQ (non-blocking)
                payload = make_backend_message(DEVICE_ID, jpg)
                try:
                    mq_outbox.put_nowait(payload)
                except queue.Full:
                    pass

            # ????????????????????? /stream (????????????????????)
            preview_jpeg = encode_jpeg(frame_draw, JPEG_QUALITY)
            with _state_lock:
                _preview_jpeg = preview_jpeg

        time.sleep(0.002)



def mq_worker_loop():
    # ??????? MQ ??? ? ??? broker ??????????????????/???????
    while not STOP:
        try:
            payload = mq_outbox.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            publisher.publish(payload, content_type="application/json")
        except Exception:
            # ?????????????? MQ ????? ? ??????????????????
            pass

# --------------- MJPEG streaming ---------------
def mjpeg_stream(get_jpeg_fn, boundary="frame", fps=15):
    min_interval = 1.0 / max(fps, 1)
    while True:
        start = time.time()
        jpg = get_jpeg_fn()
        # ????????????????? ????????????????? ?????????????????
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
            # ???????????/???????? ? ??????????
            break
        elapsed = time.time() - start
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

def get_preview_jpeg() -> bytes:
    # ???? reference ???????????? ??????????????
    with _state_lock:
        data = _preview_jpeg
        # ??????????? (???????) ?????????????????????????????????
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

# -------------- Boot Threads --------------
def start_background():
    t1 = threading.Thread(target=camera_loop, daemon=True)
    t2 = threading.Thread(target=detect_loop, daemon=True)
    t3 = threading.Thread(target=mq_worker_loop, daemon=True)
    t1.start(); t2.start(); t3.start()

start_background()