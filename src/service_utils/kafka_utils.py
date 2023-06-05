import json
import uuid
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
import logging

logger = logging.getLogger("root")


class KafkaEventListener:
    def __init__(self, config, message_handler):
        self.cmd_topic = config['model_cmd_topic'].as_str()
        self.consumer = KafkaConsumer(
            self.cmd_topic,
            group_id=config['consumer_group_id'].as_str(),
            bootstrap_servers=config['brokers'].as_str(),
            auto_offset_reset='earliest',
            client_id=config['client_id'].as_str(),
            api_version=(2, 4)
        )
        self.message_handler = message_handler

    def consume(self):
        logger.info("Listening to kafka commands")
        for message in self.consumer:
            self.message_handler(message)

    def __del__(self):
        self.consumer.close()
        logger.info('Kafka consumer closed')


class KafkaMonitor:
    output_files = []

    # runMsg is provided in server mode, and will be None in cron mode 
    def __init__(self, config, runMsg=None):
        self.status_topic = config['model_run_topic'].as_str()
        self.dlq_topic = config['model_cmd_dlq_topic'].as_str()

        self.producer = KafkaProducer(
            bootstrap_servers=config['brokers'].as_str(),
            client_id=config['client_id'].as_str(),
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retries=10,
            api_version=(2, 4)
        )

        runId = runMsg.get("runId") if isinstance(runMsg, dict) else None
        self.runID = runId if runId is not None else str(uuid.uuid4())
        self.runMsg = runMsg

    def __enter__(self):
        self.send_run_msg("start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and exc_val is None and exc_tb is None:
            self.send_run_msg("done")
        else:
            self.send_run_msg("error")
            if self.runMsg is not None:
                self.post_run_msg_dlq()

        self.producer.flush()
        self.producer.close()

    def send_run_msg(self, status):
        self.producer.send(self.status_topic, key=b'status', value={
            'status': status, 'runId': self.runID, 'timestamp': datetime.now().isoformat(), 'files': self.output_files
        })

    def post_run_msg_dlq(self):
        self.producer.send(self.dlq_topic, value=self.runMsg)

    def set_output_files(self, files):
        self.output_files = files
