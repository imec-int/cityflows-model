import argparse
import os
import shutil
import sys
import textwrap

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobClient, ContainerClient
from progress.bar import Bar
from src.config import config

CONFIG_LOCAL_DIR = config['data_files']['local_dir'].get()
CONNECTION_STRING = config['data_files']['blob_storage']['connection_string'].get()
CONTAINER_NAME = config['data_files']['blob_storage']['container_name'].get()

THIS_DIR = os.path.dirname(__file__)
REPO_ROOT_DIR = os.path.join(THIS_DIR, os.pardir, os.pardir)
LOCAL_DIR = CONFIG_LOCAL_DIR if os.path.isabs(CONFIG_LOCAL_DIR) else os.path.join(REPO_ROOT_DIR, CONFIG_LOCAL_DIR)

def get_container_client():
    try:
        container_client = ContainerClient.from_connection_string(conn_str=CONNECTION_STRING, container_name=CONTAINER_NAME)
        return container_client
    except Exception as e:
        print('Unexpected error occurred while getting the container client, aborting execution. Error: %s' % e)
        raise SystemExit

def get_blob_client(blob_path):
    try:
        blob_client = BlobClient.from_connection_string(conn_str=CONNECTION_STRING, container_name=CONTAINER_NAME, blob_name=blob_path)
        return blob_client
    except Exception as e:
        print('Data file "%s": unexpected error occurred while getting the blob client, aborting execution. Error: %s' % (blob_path, e))
        raise SystemExit

def use_data(blob_path, overwrite=False):
    '''
    Download (if necessary) a file referenced from a Blob storage container containing data files to be shared between developers

    blob_path (string): Path of the file of interest in a Blob storage container
    overwrite (boolean): Whether to overwrite the file if it already exists locally

    Returns (string): the local path to the file of interest
    '''

    log_prefix = 'Data file "%s" - download:' % blob_path

    blob_client = get_blob_client(blob_path)
    if not blob_client.exists():
        print('%s does not exist in the blob container, aborting execution' % log_prefix)
        raise SystemExit

    local_path = os.path.join(LOCAL_DIR, blob_path)
    if os.path.exists(local_path):
        if overwrite:
            print('%s already exists locally, overwriting' % log_prefix)
        else:
            print('%s already exists locally, skipping' % log_prefix)
            return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    with open(local_path, 'wb') as f:
        total_size = blob_client.get_blob_properties()['size']
        progress_bar = Bar(log_prefix, max=total_size, suffix='%(percent)d%%')
        
        storageStreamDownloader = blob_client.download_blob()
        chunks_iterator = storageStreamDownloader.chunks()
        for chunk in chunks_iterator:
            progress_bar.next(len(chunk))
            f.write(chunk)
        progress_bar.finish()

    return local_path

def upload_data(file_path, blob_path, overwrite=False, copy_to_managed_data_dir=False):
    '''
    Uploads (if necessary) a local file to a Blob storage container containing data files to be shared between developers

    file_path (string): Local file path to be uploaded
    blob_path (string): Path of the file in the Blob storage container
    overwrite (boolean): Whether to overwrite the file if it already exists on the Blob storage container
    '''
    
    log_prefix = 'Data file "%s" - upload:' % blob_path

    if not os.path.exists(file_path):
        print('%s does not exist locally, aborting execution' % log_prefix)
        raise SystemExit

    with open(file_path, 'rb') as f:
        try:
            blob_client = get_blob_client(blob_path)
            if blob_client.exists() and overwrite:
                print('%s already exists in the blob container, overwriting' % log_prefix)
            
            blob_client.upload_blob(data=f, overwrite=overwrite)
            print('%s in progress...' % log_prefix)
        except ResourceExistsError:
            print('%s already exists in the blob container, skipping' % log_prefix)
            return
        except Exception as e:
            print('%s unexpected error occurred while uploading, aborting execution. Error: %s' % (log_prefix, e))
            raise SystemExit

    print('%s done !' % log_prefix)

    # copy the file to the managed files folder (LOCAL_DIR)
    if copy_to_managed_data_dir:
        local_path = os.path.join(LOCAL_DIR, blob_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(local_path):
            print(f'Overriding managed file locally: {blob_path}')
            os.remove(local_path)
        shutil.copy(file_path, local_path)

    return

def download_blobs(prefixes, overwrite=False):
    if len(prefixes) == 0:
        print('Provided an empty prefixes array, nothing to download')

    container_client = get_container_client()
    for prefix in prefixes:
        blobs = container_client.list_blobs(prefix)
        for blob in blobs:
            use_data(blob.name, overwrite)

def upload_files(files, overwrite=False):
    for file in files:
        blob_path = os.path.relpath(file, LOCAL_DIR)
        upload_data(file, blob_path, overwrite=overwrite, copy_to_managed_data_dir=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='./manage_data.sh',
        description=textwrap.dedent(
            '''
            Download/upload data files from/to the blob container.
            Specify a subcommand to see specific usage, e.g.
                %(prog)s download --help
            or
                %(prog)s upload --help
            '''
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(title='subcommand', dest='subcommand', required=True, help='Action to perform')
    
    parser_download = subparsers.add_parser(
        'download',
        description='Download data files from the blob container',
        epilog=textwrap.dedent(
            '''
            Examples:
            ---------
            Download a single file:
                %(prog)s <path_to_file>
            Download multiple files:
                %(prog)s <path_to_file_1> <path_to_file_2> <path_to_file_3>
            Download all files in a folder:
                %(prog)s <path_to_folder>
            Download all files in multiple folders:
                %(prog)s <path_to_folder_1> <path_to_folder_2> <path_to_folder_3>
            '''
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser_download.add_argument('prefix', nargs='*', help='All remote files starting with these prefixes will be downloaded from blob container')
    parser_download.add_argument('-a', '--all', action='store_true', help='Download all remote files')
    parser_download.add_argument('-o', '--overwrite', action='store_true', help='Provide this flag to overwrite the local files')
    
    # TODO: implement this documentation when upload functionality has been added
    parser_upload = subparsers.add_parser('upload')
    parser_upload.add_argument('prefix', nargs='*', help='All local files starting with these prefixes will be uploaded to blob container')
    parser_upload.add_argument('-a', '--all', action='store_true', help='Upload all local files')
    parser_upload.add_argument('-o', '--overwrite', action='store_true', help='Provide this flag to overwrite the remote files')

    args = vars(parser.parse_args())

    subcommand = args['subcommand']
    prefixes = [''] if args['all'] else args['prefix']
    overwrite = args['overwrite']

    if subcommand == "download":
        download_blobs(prefixes, overwrite)
        sys.exit()

    if subcommand == "upload":
        upload_files(prefixes, overwrite)
        sys.exit()
