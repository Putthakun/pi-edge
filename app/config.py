import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

# ---------------- Config ----------------
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


if not AMQP_URL:
    raise RuntimeError("AMQP_URL missing")