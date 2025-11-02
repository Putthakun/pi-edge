import threading, time
import queue
from collections import deque
from app.load_model import session
import cv2
import numpy as np
from app.config import *
from app.utils import *


STOP = False
face_buffer = []
WINDOW_TIME = 1.0
last_send_time = 0.0

def detect_loop(FRAME_BUFFER, _state_lock, mq_outbox, _preview_jpeg_ref, STOP_ref):
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
            _preview_jpeg_ref[0] = encode_jpeg(frame_draw, JPEG_QUALITY)

