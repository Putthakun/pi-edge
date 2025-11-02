import cv2, numpy as np
from picamera2 import Picamera2
picam2 = Picamera2()
cfg = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    controls={
        "ExposureTime": 25000,      # microseconds (แสงในอาคาร)
        "AnalogueGain": 4.0,        # เพิ่มความสว่างโดยไม่แตก
        "Brightness": 0.05,         # นิดเดียวพอ
        "Contrast": 1.1,            # เพิ่ม contrast นิดเดียวให้คมขึ้น
        "Saturation": 1.1,          # เพิ่มความอิ่มสี
        "Sharpness": 1.2,           # ช่วยขับขอบภาพโดยไม่แตก
    }
)

picam2.configure(cfg)
picam2.start()

for i in range(100):
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Frame {i+1} sharpness={sharp:.2f}")
