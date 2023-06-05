from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import typing
from confuse import Configuration
import logging

logger = logging.getLogger("root")


def connect_to_blob_service(config: Configuration) -> BlobServiceClient:
    """
        Attempts to connect to a blob service for the specified config
        @param config: a pyConfuse config object
        @param config_view: a string that specifies which view in the azure config to use
    """
    try:
        # Connect to azure and fetch the container client
        account_key: str = config["blob_account_key"].get(
        )
        conn_string: str = config["blob_connection_string"].get(
        ).format(account_key)

        blob_service_client = BlobServiceClient.from_connection_string(
            conn_string)

        return blob_service_client
    except Exception as ex:
        logger.error('Exception:' + ex)


def download_file(container_client, blob: str) -> bytes:
    """
    Attempts to download a blob from azure blob storage and returns the content as a byte array
    @param container_client: connection to azure blob storage
    @param blob: an Azure blob name
    """
    logger.debug("Downloading blob %s" % blob)
    return container_client.download_blob(blob).content_as_bytes()
