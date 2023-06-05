import os
import logging
from confuse import Configuration

logger = logging.getLogger("root")


def clean_all(current_config: Configuration):
    """
    Cleans all input and output files.
    Files are cleaned by deleting the contents of the input and output folders.
    """
    total_count = clean_folder(current_config['data']['input'].get())
    logger.info(
        "Cleaned {0} input files.".format(total_count))
    total_count = clean_folder(current_config['data']['output'].get())
    logger.info(
        "Cleaned {0} output files.".format(total_count))


def clean_folder(folder: str) -> int:
    """
    Deletes all files from the specified folder path.
    The operation will try to delete all files even if some of them fail.
    """
    logger.info(
        "Cleaning files from path {0}.".format(folder))
    total_count = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            try:
                os.remove(os.path.join(root, file))
                total_count += 1
            except Exception as ex:
                logger.error('Error cleaning file {0}: {1}'.format(
                    file, ex))
                raise ex
    return total_count
