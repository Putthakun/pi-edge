# app/camera_manager.py
import time, threading
from picamera2 import Picamera2
from app.config import CAM_WIDTH, CAM_HEIGHT

def camera_loop(FRAME_BUFFER, _state_lock, STOP_ref):
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(cfg)
    picam2.start()
    print("[camera] RGB888 started")

    try:
        while not STOP_ref():
            frame = picam2.capture_array()
            with _state_lock:
                FRAME_BUFFER.append(frame)
            time.sleep(0.005)
    finally:
        picam2.stop()
