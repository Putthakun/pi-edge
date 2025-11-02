# app/mq_worker.py
import time, logging, queue
from app.mq_publisher import MQPublisher
from app.config import AMQP_URL, QUEUE_NAME

publisher = MQPublisher(AMQP_URL, QUEUE_NAME)
mq_outbox: "queue.Queue[bytes]" = queue.Queue(maxsize=200)

def mq_worker_loop(mq_outbox, STOP_ref):
    global publisher
    while not STOP_ref():
        try:
            payload = mq_outbox.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            publisher.publish(payload, content_type="application/json")
            logging.info("🚀 Face sent to RabbitMQ successfully")
        except Exception as e:
            logging.error(f"❌ MQ publish error: {e}")
            time.sleep(1)
