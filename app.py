# --- Block 1: imports & config ---
import os, time, json, base64, threading, zlib
from datetime import datetime, timezone
from collections import deque
from typing import List, Optional

import numpy as np
import cv2
import pika
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from picamera2 import Picamera2

load_dotenv()
CAM_WIDTH=int(os.getenv("CAM_WIDTH","1280")); CAM_HEIGHT=int(os.getenv("CAM_HEIGHT","720"))
DETECT_INTERVAL_MS=int(os.getenv("DETECT_INTERVAL_MS","100"))
FACE_MARGIN_RATIO=float(os.getenv("FACE_MARGIN_RATIO","0.2"))
JPEG_QUALITY=int(os.getenv("JPEG_QUALITY","80"))
MAX_IMAGE_BYTES=int(os.getenv("MAX_IMAGE_BYTES",str(512*1024)))
DEVICE_ID=os.getenv("DEVICE_ID","pi5-unknown")
AMQP_URL=os.getenv("AMQP_URL")
EXCHANGE=(os.getenv("EXCHANGE") or "").strip()
ROUTING_KEY=os.getenv("ROUTING_KEY","face_images")
QUEUE=os.getenv("QUEUE","face_images")
HTTP_HOST=os.getenv("HTTP_HOST","0.0.0.0"); HTTP_PORT=int(os.getenv("HTTP_PORT","8000"))
if not AMQP_URL: raise RuntimeError("AMQP_URL missing")
# --- Block 2: MQ & utils ---
class MQPublisher:
    def __init__(self, url:str, exchange:str, routing_key:str, queue:Optional[str]=None):
        self._url=url; self._exchange=exchange; self._routing_key=routing_key; self._queue=queue
        self._conn=None; self._ch=None; self._lock=threading.Lock()
    def _connect(self):
        p=pika.URLParameters(self._url); p.heartbeat=30; p.blocked_connection_timeout=60
        self._conn=pika.BlockingConnection(p); self._ch=self._conn.channel()
        if self._queue: self._ch.queue_declare(queue=self._queue, durable=True)
        if self._exchange:
            self._ch.exchange_declare(exchange=self._exchange, exchange_type='direct', durable=True)
            if self._queue: self._ch.queue_bind(queue=self._queue, exchange=self._exchange, routing_key=self._routing_key)
    def _ensure(self):
        if (self._conn is None) or self._conn.is_closed or (self._ch is None) or self._ch.is_closed: self._connect()
    def publish(self, body:bytes, content_type:str="application/json"):
        with self._lock:
            backoff=1.0
            while True:
                try:
                    self._ensure()
                    props=pika.BasicProperties(content_type=content_type, delivery_mode=2)
                    rk=self._routing_key if self._exchange else (self._queue or self._routing_key)
                    self._ch.basic_publish(exchange=self._exchange if self._exchange else "", routing_key=rk, body=body, properties=props)
                    return
                except Exception:
                    try:
                        if self._conn: self._conn.close()
                    except Exception: pass
                    time.sleep(min(backoff,10)); backoff*=2
    def close(self):
        with self._lock:
            try:
                if self._conn and self._conn.is_open: self._conn.close()
            except Exception: pass

publisher=MQPublisher(AMQP_URL, EXCHANGE, ROUTING_KEY, QUEUE)
app=FastAPI(title="Pi5 HQ Realtime Stream + Face Publisher")
FRAME_BUFFER=deque(maxlen=3); LAST_FACES_GRID=None; LOCK=threading.Lock(); STOP=False

CASCADE=cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/"+"haarcascade_frontalface_default.xml")
if CASCADE.empty():
    raise RuntimeError("Failed to load Haar cascade: /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
if CASCADE.empty():
    raise RuntimeError("Failed to load Haar cascade from /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
if CASCADE.empty(): raise RuntimeError("Cannot load haarcascade_frontalface_default.xml")

def now_iso(): return datetime.now(timezone.utc).isoformat()
def crop_with_margin(img,x,y,w,h,margin=0.2):
    H,W=img.shape[:2]; mx=int(w*margin); my=int(h*margin)
    x0=max(0,x-mx); y0=max(0,y-my); x1=min(W,x+w+mx); y1=min(H,y+h+my); return img[y0:y1,x0:x1]
def encode_jpeg(img,q=80):
    ok,buf=cv2.imencode(".jpg",img,[int(cv2.IMWRITE_JPEG_QUALITY),q]); return buf.tobytes() if ok else b""
def faces_grid(crops:List[np.ndarray],cols=3,cell=200,bg=(30,30,30)):
    if not crops: return np.zeros((cell,cell,3),dtype=np.uint8)
    import math
    rows=int(math.ceil(len(crops)/cols)); canvas=np.full((rows*cell,cols*cell,3),bg,dtype=np.uint8)
    for i,face in enumerate(crops):
        if face is None or face.size==0: continue
        h,w=face.shape[:2]; s=min(cell/max(w,1), cell/max(h,1))
        nw,nh=max(int(w*s),1), max(int(h*s),1); rsz=cv2.resize(face,(nw,nh))
        r=i//cols; c=i%cols; x=c*cell+(cell-nw)//2; y=r*cell+(cell-nh)//2; canvas[y:y+nh,x:x+nw]=rsz
    return canvas
def make_backend_message(camera_id:str,jpeg_bytes:bytes)->bytes:
    comp=zlib.compress(jpeg_bytes); b64=base64.b64encode(comp).decode("ascii")
    return json.dumps({"camera_id":camera_id,"image":b64}, ensure_ascii=False).encode("utf-8")
# --- Block 3: camera, detect, stream, endpoints ---
def camera_loop():
    global STOP
    picam2=Picamera2()
    cfg=picam2.create_video_configuration(main={"size":(CAM_WIDTH,CAM_HEIGHT),"format":"RGB888"})
    picam2.configure(cfg); picam2.start()
    try:
        while not STOP:
            frame=picam2.capture_array()  # RGB
            bgr=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            with LOCK: FRAME_BUFFER.append(bgr)
            time.sleep(0.005)
    finally:
        picam2.stop()

def detection_and_publish_loop():
    global LAST_FACES_GRID
    interval=max(20, DETECT_INTERVAL_MS)/1000.0; last=0.0
    while not STOP:
        with LOCK: frame=FRAME_BUFFER[-1].copy() if FRAME_BUFFER else None
        if frame is None: time.sleep(0.01); continue
        now=time.time()
        if (now-last)>=interval:
            last=now
            small=cv2.resize(frame,(0,0),fx=0.5,fy=0.5); gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
            faces=CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))
            crops=[]
            for (x,y,w,h) in faces:
                X,Y,W,H=int(x*2),int(y*2),int(w*2),int(h*2)
                cv2.rectangle(frame,(X,Y),(X+W,Y+H),(0,255,0),2)
                face=crop_with_margin(frame,X,Y,W,H,FACE_MARGIN_RATIO)
                if face is None or face.size==0: continue
                crops.append(face)
                jpg=encode_jpeg(face, JPEG_QUALITY)
                if len(jpg)>MAX_IMAGE_BYTES:
                    jpg=encode_jpeg(face, max(50, JPEG_QUALITY-20))
                    if len(jpg)>MAX_IMAGE_BYTES: continue
                publisher.publish(make_backend_message(DEVICE_ID, jpg), content_type="application/json")
            with LOCK: LAST_FACES_GRID=faces_grid(crops or [], cols=3, cell=200)
        with LOCK:
            if FRAME_BUFFER: FRAME_BUFFER[-1]=frame
        time.sleep(0.005)

def mjpeg_stream(get_fn, boundary="frame", fps=20):
    mi=1.0/max(fps,1)
    while True:
        start=time.time(); jpg=get_fn()
        if jpg:
            yield (b"--"+boundary.encode()+b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: "+str(len(jpg)).encode()+b"\r\n\r\n"+jpg+b"\r\n")
        el=time.time()-start
        if el<mi: time.sleep(mi-el)

def get_preview_jpeg():
    with LOCK:
        if not FRAME_BUFFER: return b""
        return encode_jpeg(FRAME_BUFFER[-1], JPEG_QUALITY)

def get_faces_grid_jpeg():
    with LOCK:
        if LAST_FACES_GRID is None:
            return encode_jpeg(np.zeros((200,200,3),dtype=np.uint8), JPEG_QUALITY)
        return encode_jpeg(LAST_FACES_GRID, JPEG_QUALITY)

@app.get("/")
def root():
    return {"ok":True,"device_id":DEVICE_ID,"stream":"/stream","faces":"/faces","ts":datetime.now(timezone.utc).isoformat()}

@app.get("/stream")
def stream():
    return StreamingResponse(mjpeg_stream(get_preview_jpeg, fps=20),
        media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/faces")
def faces():
    return StreamingResponse(mjpeg_stream(get_faces_grid_jpeg, fps=8),
        media_type='multipart/x-mixed-replace; boundary=frame')

def start_background():
    t1=threading.Thread(target=camera_loop, daemon=True)
    t2=threading.Thread(target=detection_and_publish_loop, daemon=True)
    t1.start(); t2.start()
start_background()
