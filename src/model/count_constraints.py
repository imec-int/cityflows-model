import json
def get_bounds(bounds_path):
    with open(bounds_path) as file:
        bounds = json.load(file)
    return bounds