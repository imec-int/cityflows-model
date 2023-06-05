import datetime
import json
import os
import requests
import time
import geopandas as gpd

from src.utils.files import write_json_file
from src.utils.retry import make_execute_with_retry

# Get api key from https://telraam.net/
from config import TELRAAM_API_KEY

# rate limiting, unit is: number / second, or Hertz
TELRAAM_API_RATE = 1

# the time increment between reports, unit is seconds
REPORTS_TIME_INCREMENT = 3600

URL = "https://telraam-api.net/v1/reports/traffic_snapshot"
HEADERS = {
    'Content-Type': 'application/json',
    'X-Api-Key': TELRAAM_API_KEY,
}


def fetch_traffic_report(time, area):
    body = {
        "time": f"{time.isoformat()}",
        "contents": "minimal",
        "area": f"{area['min_lon']},{area['min_lat']},{area['max_lon']},{area['max_lat']}"
    }

    response = requests.request(
        "POST", URL, headers=HEADERS, data=json.dumps(body))

    if response.status_code != 200:
        raise Exception(
            f"Error fetching traffic report: status code {response.status_code}")

    return response.json()


def download_traffic_reports(period, area, download_folder):
    os.makedirs(download_folder, exist_ok=True)

    inbetween_requests_wait_time = 1 / TELRAAM_API_RATE
    execute_with_retry = make_execute_with_retry(
        n_tries=5, wait_time=inbetween_requests_wait_time)

    iter_datetime = period['start']
    while (iter_datetime <= period['end']):
        print(f'Fetching traffic report for {iter_datetime.isoformat()}')

        report = execute_with_retry(fetch_traffic_report, iter_datetime, area)
        filename = f'traffic_snapshot_{iter_datetime.isoformat()}.json'
        filepath = os.path.join(download_folder, filename)
        write_json_file(filepath, report)

        # wait for the rate limit
        time.sleep(inbetween_requests_wait_time)

        iter_datetime += datetime.timedelta(seconds=REPORTS_TIME_INCREMENT)


if __name__ == '__main__':
    # run parameters
    period_start = '2020-05-11T00:00:00+02:00'
    period_end = '2020-05-18T00:00:00+02:00'
    area_filepath = 'data/managed_data_files/shapefiles/modelling_zones/antwerp/zone.geojson'
    download_folder = 'data/scraping/telraam'

    # format function arguments from run parameters
    period = {
        'start': datetime.datetime.fromisoformat(period_start),
        'end': datetime.datetime.fromisoformat(period_end)
    }

    area_gdf = gpd.read_file(area_filepath)
    min_lon, min_lat, max_lon, max_lat = area_gdf.total_bounds
    area = {
        'min_lon': min_lon,
        'min_lat': min_lat,
        'max_lon': max_lon,
        'max_lat': max_lat,
    }

    download_traffic_reports(period, area, download_folder)
