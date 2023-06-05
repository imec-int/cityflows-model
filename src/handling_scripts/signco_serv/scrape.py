import json
import os
import requests
from base64 import b64encode
from datetime import date
from shapely.geometry import Point

BASE_URL = 'https://api.signcoserv.be'
from config import CLIENT_ID, CLIENT_SECRET # These you should get from signco_serv website
OUTPUT_DIRECTORY = 'data/scraping/signco_serv'
ACCESS_TOKEN = None


def set_access_token():
    res = requests.get(BASE_URL + '/api/v1/AuthenticationAuthority')
    if res.status_code != 200:
        raise Exception('Issue when calling Authentication Authority')

    auth_url = res.text + '/connect/token'

    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    credentials_encoded = b64encode(
        credentials.encode("ascii")).decode("ascii")

    headers = {'Authorization': f'Basic {credentials_encoded}'}
    body = {'grant_type': 'client_credentials', 'scope': 'signco-api'}

    token_res = requests.post(
        auth_url, data=body, headers=headers)
    if token_res.status_code != 200:
        raise Exception('Issue when retrieving token')

    global ACCESS_TOKEN
    ACCESS_TOKEN = token_res.json()['access_token']


def get_measuring_points():
    url = BASE_URL + "/api/integrations/generic/v1/MeasuringPoints"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        raise Exception("Could not retrieve MeasuringPoints")

    data = res.json()
    directory = os.path.join(OUTPUT_DIRECTORY)
    filename = 'MeasuringPoints.json'
    filepath = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(json.dumps(data))

    # format the list of measuring points as guid/location
    measuring_points = []
    for el in data['data']:
        location = el['coordinate']
        for mp in el['measuringPoints']:
            measuring_points.append({"guid": mp["guid"], "location": location})
    return measuring_points


def filter(measuring_points, zone):
    # Spatially filtering the measuring points (checking if they are in zone)
    return [mp for mp in measuring_points if Point(mp['location']['longitude'], mp['location']['latitude']).within(zone)]


def scrape_measuring_points(measuring_points, start_date, end_date):
    start_date_ordinal = start_date.toordinal()
    end_date_ordinal = end_date.toordinal()
    ordinal_dates = range(start_date_ordinal, end_date_ordinal + 1)
    dates = [date.fromordinal(v) for v in ordinal_dates]
    for iter_date in dates:
        print(f"Scraping date: {iter_date.isoformat()}")
        for mp in measuring_points:
            id = mp['guid']

            url = BASE_URL + \
                f'/api/integrations/generic/v1//MeasuringPoints/{id}/VehicleDetails'
            params = {
                "date": iter_date.isoformat()
            }
            headers = {
                "Authorization": f"Bearer {ACCESS_TOKEN}"
            }
            res = requests.get(url, params=params, headers=headers)
            if res.status_code != 200:
                raise Exception("Could not retrieve VehicleDetails")

            data = res.json()
            directory = os.path.join(OUTPUT_DIRECTORY, 'VehicleDetails')
            filename = f'{iter_date.isoformat()}_{id}.json'
            filepath = os.path.join(directory, filename)
            os.makedirs(directory, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json.dumps(data))


if __name__ == '__main__':
    set_access_token()
    measuring_points = get_measuring_points()

    import geopandas as gpd
    # geojson with a border that is the zone of interest
    zone_file = 'data/managed_data_files/shapefiles/modelling_zones/antwerp/zone.geojson'
    zone = gpd.GeoDataFrame.from_file(zone_file).unary_union
    antwerp_measuring_points = filter(measuring_points, zone)

    # both dates are included
    start_date = date.fromisoformat('2020-01-01')
    end_date = date.fromisoformat('2020-01-02')
    scrape_measuring_points(antwerp_measuring_points, start_date, end_date)
