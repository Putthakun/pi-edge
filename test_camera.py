#!/usr/bin/env python3
"""
Test script to debug camera color issue
Run this separately to test camera formats
"""
import cv2
import numpy as np
from picamera2 import Picamera2
import time

def test_camera_formats():
    print("Testing camera formats...")
    picam2 = Picamera2()
    
    # Test 1: Try different formats
    formats_to_test = ["BGR888", "RGB888", "XBGR8888", "XRGB8888"]
    
    for fmt in formats_to_test:
        try:
            print(f"\n--- Testing {fmt} ---")
            cfg = picam2.create_video_configuration(main={"size":(640,480), "format":fmt})
            picam2.configure(cfg)
            picam2.start()
            time.sleep(0.5)
            
            # Capture a frame
            frame = picam2.capture_array()
            print(f"Shape: {frame.shape}, dtype: {frame.dtype}")
            
            # Save test images
            cv2.imwrite(f"test_{fmt}_raw.jpg", frame)
            print(f"Saved test_{fmt}_raw.jpg")
            
            # Try with color conversion
            if "RGB" in fmt:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"test_{fmt}_converted.jpg", frame_bgr)
                print(f"Saved test_{fmt}_converted.jpg (RGB->BGR converted)")
            
            picam2.stop()
            print(f"? {fmt} works!")
            
        except Exception as e:
            print(f"? {fmt} failed: {e}")
            try:
                picam2.stop()
            except:
                pass
    
    print("\n" + "="*50)
    print("Check the test_*.jpg files to see which has correct colors")
    print("="*50)

if __name__ == "__main__":
    test_camera_formats()