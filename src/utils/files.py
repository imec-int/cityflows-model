import json
import os


def write_json_file(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)


def ensure_file_not_exists(filepath):
    if filepath is None:
        return

    if os.path.exists(filepath):
        os.remove(filepath)

    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
