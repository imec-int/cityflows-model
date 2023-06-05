import json
import os

directory = 'data/scraping/signco_serv/VehicleDetails'

if __name__ == '__main__':
    counts = {}
    for file in os.listdir(directory):
        basename = os.path.splitext(file)
        guid = basename[0].split("_")[1]
        if guid not in counts:
            counts[guid] = 0

        filepath = os.path.join(directory, file)
        with open(filepath, 'r') as f:
            content = json.loads(f.read())

        counts[guid] += content['itemsCount']

    print(counts)
