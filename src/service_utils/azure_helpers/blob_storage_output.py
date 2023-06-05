from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from .common import connect_to_blob_service
import os
import typing
from confuse import Configuration
import logging

logger = logging.getLogger("root")


def upload_file_to_output(config: Configuration, output_file_path: str, blob_path: str) -> bool:
    if not os.path.isfile(output_file_path):
        logger.warn('File path does not point to a file ')
    if not os.path.exists(output_file_path):
        logger.error("Filepath does not exist")
        return False
    try:
        # Connect to azure and fetch the blob client
        blob_service_client = connect_to_blob_service(config["azure"]["output"])
        container_name = config["azure"]["output"]["container_name"].get()
        blob_client = blob_service_client.get_blob_client(container_name, blob_path)

        with open(output_file_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)

        logger.info('Successfully uploaded {0} to azure container {1}'.format(
            output_file_path, container_name))
    except Exception as ex:
        logger.error('Uploading {0} failed, exception occured: {1}'.format(
            output_file_path, ex))
        return False

    return True
