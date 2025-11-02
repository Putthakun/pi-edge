# app/api_routes.py
import asyncio, base64, numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from app.utils import encode_jpeg, now_iso

router = APIRouter()

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

def register_routes(app, get_preview_jpeg, DEVICE_ID):
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
