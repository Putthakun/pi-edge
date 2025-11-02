# app/mq_publisher.py
import time
import logging
import pika

class MQPublisher:
    def __init__(self, amqp_url: str, queue_name: str):
        self.amqp_url = amqp_url
        self.queue_name = queue_name
        self.connection = None
        self.channel = None
        self._connect()

    def _connect(self):
        """เชื่อมต่อ RabbitMQ และประกาศคิว durable"""
        params = pika.URLParameters(self.amqp_url)
        params.heartbeat = 30
        params.blocked_connection_timeout = 60

        while True:
            try:
                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()

                # ✅ ให้ queue durable จริง (ต้องตรงกับ backend)
                self.channel.queue_declare(queue=self.queue_name, durable=True)
                logging.info(f"✅ Connected to RabbitMQ (queue={self.queue_name}, durable=True)")
                break

            except Exception as e:
                logging.warning(f"❌ RabbitMQ connect failed: {e}, retrying...")
                time.sleep(3)

    def publish(self, data, content_type="application/json"):
        """ส่งข้อความไปที่คิวแบบ durable"""
        try:
            if self.connection is None or self.connection.is_closed:
                self._connect()

            # ✅ ส่ง message แบบ durable (ไม่หายถ้า RabbitMQ รีสตาร์ต)
            self.channel.basic_publish(
                exchange='',
                routing_key=self.queue_name,
                body=data,
                properties=pika.BasicProperties(
                    content_type=content_type,
                    delivery_mode=2  # Durable message
                )
            )
            logging.info(f"📤 Published message to queue '{self.queue_name}' ({len(data)} bytes)")

        except Exception as e:
            logging.error(f"❌ MQ publish error: {e}")
            time.sleep(2)
            self._connect()
