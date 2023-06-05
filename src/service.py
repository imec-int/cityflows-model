import json
import logging
import threading

from .model.Main import execute_model
from .service_utils import config
from .service_utils import http_server
from .service_utils import kafka_utils
from .service_utils.azure_helpers import blob_storage_input
from .service_utils.files_utils import clean_all
from .service_utils.logger import setupLogger
from .service_utils.upload_data import upload_data

# Setup logger
setupLogger()
logger = logging.getLogger("root")

# Load configuration
current_config = config.load()

# Load and store azure blob storage key
account_key = config.get_secret("AZURE_CITYFLOWS_STORAGE_KEY")
current_config['azure']['output']['blob_account_key'].set(account_key)
current_config['azure']['input']['blob_account_key'].set(account_key)


def get_files_from_payload(payload) -> dict:
    custom_files = payload.get("files")
    if custom_files == None:
        logger.warn(
            "Empty payload files, falling back on last data on azure blob storage")
    return custom_files


def message_handler(message):
    message_key = message.key.decode('utf-8')
    message_value = message.value.decode('utf-8')
    if message_key == "start":
        logger.info('Start received')
        runMsg = json.loads(message_value)
        # Check if custom files are passed
        # If none are given the model will download and use the latest data on azure blob storage similar to cronjob mode
        custom_files = get_files_from_payload(runMsg)
        # TODO: run in a thread to support aborting the model
        # TODO: store the runId to support aborting the model
        start_model(custom_files, runMsg)
    # TODO: implement stop message


def start_server():
    logger.info("Starting Cityflows Data Model server")

    kafka_event_listener = kafka_utils.KafkaEventListener(
        config=current_config['kafka'], message_handler=message_handler)

    consumer = threading.Thread(
        name="consumer", target=kafka_event_listener.consume)
    # Set as a daemon so it will be killed once the main thread is dead.
    consumer.setDaemon(True)
    consumer.start()

    server = threading.Thread(name="http_server", target=http_server.start, kwargs={
        "port": current_config['server']['http_port'].as_number()})
    # Set as a daemon so it will be killed once the main thread is dead.
    server.setDaemon(True)
    server.start()

    server.join()
    consumer.join()


def start_model(files=None, runMsg=None):
    logger.info("Starting Cityflows Data Model job")
    try:
        with kafka_utils.KafkaMonitor(current_config['kafka'], runMsg) as producer:

            runId = runMsg.get("runId")
            if runId is None:
                # this triggers the message to be put on the dead-letter queue
                raise Exception('runId is missing')

            if not config.is_cache_enabled(current_config):
                logger.info("Cleaning model data")
                clean_all(current_config)

            if config.is_download_model_input_data_enabled(current_config):
                logger.info("Downloading model input data")
                blob_storage_input.download_model_input_data(
                    current_config, files)

            if config.is_model_enabled(current_config):
                logger.info("Starting model")

                input_folder = current_config["data"]["input"].get()
                output_folder = current_config["data"]["output"].get()
                modality_mapping_path = current_config['model']['modality_mapping'].get(
                )
                modality_mixing_iteration_step = current_config['model']['modality_mixing_iteration_step'].as_number(
                )
                upsampling_frequency = current_config['model']['upsampling_frequency'].get(
                )

                execute_model(input_folder, output_folder, modality_mapping_path,
                              modality_mixing_iteration_step, upsampling_frequency)

            if config.is_publish_model_output_data_enabled(current_config):
                logger.info("Publishing model output data")
                uploaded_files = upload_data(current_config, blob_folder=runId)

                producer.set_output_files(uploaded_files)

        logger.info("Run completed")

    except Exception as error:
        logger.error(error)


if current_config['server']['enabled'].get():
    start_server()
else:
    start_model()
