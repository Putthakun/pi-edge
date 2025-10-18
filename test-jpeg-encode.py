#!/usr/bin/env python3
"""
Test JPEG encoding with different color conversions
"""
import cv2
import numpy as np
from picamera2 import Picamera2
import time

picam2 = Picamera2()
cfg = picam2.create_video_configuration(main={"size":(640,480), "format":"RGB888"})
picam2.configure(cfg)
picam2.start()
time.sleep(0.5)

# Capture RGB frame
frame_rgb = picam2.capture_array()
print(f"Captured frame shape: {frame_rgb.shape}")

# Test 1: Save RGB directly (will be wrong)
cv2.imwrite("output_1_rgb_direct.jpg", frame_rgb)
print("Saved: output_1_rgb_direct.jpg (RGB saved directly - WRONG)")

# Test 2: Convert RGB to BGR then save
frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_2_bgr_converted.jpg", frame_bgr)
print("Saved: output_2_bgr_converted.jpg (RGB->BGR converted - SHOULD BE CORRECT)")

# Test 3: Use imencode with BGR
ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
if ok:
    with open("output_3_imencode_bgr.jpg", "wb") as f:
        f.write(buf.tobytes())
    print("Saved: output_3_imencode_bgr.jpg (imencode with BGR - SHOULD BE CORRECT)")

# Test 4: Use imencode with RGB directly (wrong)
ok, buf = cv2.imencode(".jpg", frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
if ok:
    with open("output_4_imencode_rgb.jpg", "wb") as f:
        f.write(buf.tobytes())
    print("Saved: output_4_imencode_rgb.jpg (imencode with RGB - WRONG)")

picam2.stop()

print("\n" + "="*60)
print("Check all output_*.jpg files")
print("output_2 and output_3 should have CORRECT colors")
print("="*60)