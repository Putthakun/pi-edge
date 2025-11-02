import numpy as np
import cv2, zlib, base64
import json

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
