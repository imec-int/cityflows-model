import os
import logging

from .azure_helpers import blob_storage_output

logger = logging.getLogger("root")


def upload_data(config, blob_folder = ''):
    total_count = 0
    failed_count = 0
    uploaded_files = []
    for root, dirs, files in os.walk(config['data']['output'].get()):
        for name in files:
            file_path: str = os.path.join(root, name)
            blob_path: str = f'{blob_folder}/{name}'.lstrip('/')
            result: bool = blob_storage_output.upload_file_to_output(config, file_path, blob_path)
            total_count += 1
            if not result:
                failed_count += 1
            else:
                uploaded_files.append(blob_path)
    logger.info(
        "Succesfully uploaded {0}/{1} files".format(total_count-failed_count, total_count))

    return uploaded_files
