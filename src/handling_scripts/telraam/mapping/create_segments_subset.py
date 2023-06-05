import dask.dataframe as dd
import geopandas as gpd
import requests
from src.handling_scripts.telraam.scraping import HEADERS


def get_all_segments():
    return requests.get('https://telraam-api.net/v1/segments/all', headers=HEADERS).json()


if __name__ == '__main__':
    # retrieve all the segments from Telraam API
    all_segments = get_all_segments()
    all_segments_gdf = gpd.GeoDataFrame.from_features(
        all_segments['features']).set_crs(crs=31370).to_crs(4326)

    # create the subset of oidns that we scraped
    oidns = set([])
    scraped_files_df = dd.read_json('data/scraping/telraam/*').compute()
    for features in scraped_files_df['features']:
        for feature in features:
            oidns.add(feature['properties']['segment_id'])

    # create the subset of segments that we need to manually map
    scraped_segments = all_segments_gdf[all_segments_gdf['oidn'].isin(oidns)]
    scraped_segments.to_file(
        'data/managed_data_files/mobiele_stad/learning_cycle_2/data_preparation/telraam/antwerp_scraped_segments.geojson', driver='GeoJSON')
