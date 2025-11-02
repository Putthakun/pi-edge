# --- Pi5 Face Stream (Production Architecture) ---
import threading, queue
from collections import deque
from fastapi import FastAPI

from app.logger import setup_logger
from app.config import *
from app.utils import *
from app.face_detect import detect_loop
from app.camera_manager import camera_loop
from app.mq_worker import mq_worker_loop
from app.api_routes import register_routes  # ✅ ใช้ router ใหม่

app = FastAPI(title="Pi5 Face Stream + Face Publisher")
setup_logger()

# Shared state
FRAME_BUFFER = deque(maxlen=3)
_state_lock = threading.Lock()
_preview_jpeg_ref = [b""]
mq_outbox: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
STOP = False

# ----- Function -----
def get_preview_jpeg() -> bytes:
    with _state_lock:
        data = _preview_jpeg_ref[0]
        fallback = FRAME_BUFFER[-1] if FRAME_BUFFER else None
    if data:
        return data
    if fallback is not None:
        return encode_jpeg(fallback, JPEG_QUALITY)
    return b""

# ----- Register API -----
register_routes(app, get_preview_jpeg, DEVICE_ID)

# ----- Threads -----
def start_background():
    t1 = threading.Thread(target=camera_loop, args=(FRAME_BUFFER, _state_lock, lambda: STOP), daemon=True)
    t2 = threading.Thread(target=detect_loop, args=(FRAME_BUFFER, _state_lock, mq_outbox, _preview_jpeg_ref, lambda: STOP), daemon=True)
    t3 = threading.Thread(target=mq_worker_loop, args=(mq_outbox, lambda: STOP), daemon=True)
    t1.start(); t2.start(); t3.start()

start_background()
