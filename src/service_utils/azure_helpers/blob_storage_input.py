from confuse import Configuration
from datetime import datetime
from dateutil.tz import tzutc
from typing import List, Tuple
import dateutil.parser
import logging
import os
import re

from .common import connect_to_blob_service, download_file

logger = logging.getLogger("root")


def download_merged_blobs(container_client,  local_file_path: str, blobs: List[str]):
    """
    Downloads all contents from the blobs array and writes sequentially them into the local file
    The first line is removed from every file expect the first to take .csv column headers into account
    @param container_client: connection to azure blob storage
    @param local_file_path: location to download to
    @param blobs: an array of Azure blob names
    """
    byteNewLine = str.encode('\n')
    with open(local_file_path, "wb") as local_file:
        for index, blob in enumerate(blobs):
            file = download_file(container_client, blob)

            if index == 0:
                local_file.write(file)
            else:
                end_first_line = file.find(byteNewLine) + 1
                local_file.write(file[end_first_line:])
            local_file.write(byteNewLine)
        logger.info("Downloaded and merged %s blobs to %s" %
                    (len(blobs), local_file_path))


def dissect_blob_name(name: str) -> Tuple[str, datetime]:
    """
    Attempts to parse a blob name with format <ISO8601timestamp>_<filename>
    and returns a Tuple containing a filename and timestamp object
    """
    file_parts = name.split('_', 1)
    file_timestamp = file_parts[0]
    file_category = file_parts[1]  # e.g. all_data.csv
    # Attempt to get timestamp
    timestamp = dateutil.parser.isoparse(file_timestamp)
    return file_category, timestamp


def filter_blobs(blobs_to_download_per_category: dict) -> dict:
    """
    Some categories only require a single file timestamp
    Here we hardcoded replace these category files with only the latest version
    """
    for blob_category, associated_blobs in blobs_to_download_per_category.items():
        # TODO: make this less hardcoded
        if blob_category in ["intersections.csv", "street_segments.csv"]:
            blobs_to_download_per_category[blob_category] = [
                blobs_to_download_per_category[blob_category][1]]
    return blobs_to_download_per_category


def find_latest_model_input_data(container_client, minimum_age: int) -> dict:
    """
    Fetches all the available blobs metadata and checks if they match our name formatting
    When they match we check if the files are older then 4 hours and take the 2 most recent files of each category
    @param container_client: connection to azure blob storage
    @return: files as a dictionary [<categoryName>]: [<blobNamesArr>]
    """
    # Get all the files in the container and filter
    blobs_list = container_client.list_blobs()
    # Will contain an array [most recent, second most recent] per category
    blobs_to_download_per_category = {}

    # The utc timezone is required for date math
    time_now = datetime.now().astimezone(tz=tzutc())
    pattern = re.compile(r"^([0-9]\S*)_(\S*)\.csv")
    timestamp_to_beat = {}
    # Iterate over all blobs in the container
    # NOTE: If there are files in the blob storage that do not follow the naming convention, we currently just ignore them
    logger.info("Finding model input data")
    for index, blob in enumerate(blobs_list):
        if index % 10000 == 0 and index != 0:
            logger.info("Files checked %s" % index)

        if not pattern.match(blob.name):
            continue
        try:
            file_category, timestamp = dissect_blob_name(blob.name)
        except:
            logger.warn('Failed to parse name for blob: {} \n\t Does it follow the naming convention: <ISO8601string>_<filename>.csv?'.format(
                blob.name))
            continue
        time_diff = time_now-timestamp
        # If younger than minimum age, ignore
        if time_diff.total_seconds() < minimum_age:
            continue
        # Get the blob we currently store in the dictionary
        # Compare its timestamp with the current blob's timestamp
        blobs_to_beat = blobs_to_download_per_category.get(
            file_category, [None, None])

        if blobs_to_beat[0] == None:
            timestamp_to_beat[file_category] = timestamp
            blobs_to_download_per_category[file_category] = [
                blob.name, blob.name]
            continue

        time_diff_to_beat = time_now-timestamp_to_beat[file_category]
        if time_diff.total_seconds() < time_diff_to_beat.total_seconds():
            timestamp_to_beat[file_category] = timestamp
            # Save the previous blob as the second most recent one
            blobs_to_download_per_category[file_category][0] = blobs_to_download_per_category[file_category][1]
            blobs_to_download_per_category[file_category][1] = blob.name

    blobs_to_download_per_category = filter_blobs(
        blobs_to_download_per_category)
    return blobs_to_download_per_category


def download_model_input_data(config: Configuration, files=None) -> List[str]:
    """
    Attempts to download files from the blob storage specified in the config, to the input folder specified in the config
    @param config: a Configuration object
    @param files: an optional custom list of files to download from blob storage if left empty it will get the last available files
    @return: a list of paths to all the downloaded files
    """
    # Set up the download directory
    input_path: str = config["data"]["input"].get()
    if not os.path.exists(input_path):
        logger.debug("[fetch input data] - % s does not exist, creating directory" %
                     (input_path))
        os.makedirs(input_path)
    # Set up the return value
    downloaded_file_paths = []
    # Try to connect to blob storage and get the relevant files
    try:
        blob_service_client = connect_to_blob_service(config["azure"]["input"])
        container_client = blob_service_client.get_container_client(
            config["azure"]["input"]["container_name"].get())

        if files == None:
            blobs_to_download_per_category = find_latest_model_input_data(
                container_client, config["data"]["minimum_date_age_s"].get())
        else:
            blobs_to_download_per_category = files
        logger.info(blobs_to_download_per_category)

        # Download the most recent file that is older than 4 hours, per "category"
        for blob_category, associated_blobs in blobs_to_download_per_category.items():
            local_file_path = os.path.join(input_path, blob_category)
            download_merged_blobs(
                container_client, local_file_path, associated_blobs)

    except Exception as ex:
        logger.error('Exception caught: ', ex)

    return downloaded_file_paths
