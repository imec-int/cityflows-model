import os
import geopandas as gpd
import pandas as pd

import json
import datetime

from src.utils.geometry import convert_to_linestring


def project(filepath, street_mapping):
    raw_folder = filepath

    time_window_start = pd.to_datetime('2020-12-01T00:00:00+00:00')
    time_window_end = pd.to_datetime('2021-01-01T00:01:00+00:00')

    
    meta = pd.read_csv ('data/managed_data_files/mobiele_stad/datasources/telraam_crawled/_meta.csv')
    segment_ids = meta['oidn'].to_list()
    gmeta = gpd.GeoDataFrame(meta.loc[:, [c for c in meta.columns if c != "geometry"]],
        geometry=gpd.GeoSeries.from_wkt(meta["geometry"]),
        crs="epsg:31370", #4326",
    )
    gmeta=gmeta.to_crs(epsg = 4326)

    iter_datetime = time_window_start
    reports = []
    while (iter_datetime < time_window_end):
        next_iter_datetime = iter_datetime + datetime.timedelta(hours=1)
        start = iter_datetime.isoformat()
        end = next_iter_datetime.isoformat()

        for segment_id in segment_ids:
            fp = raw_folder + f'{segment_id}_{start}_{end}.json'.replace(':','_')
            if os.path.exists(fp):
                jsonContents = open(fp, 'r').read()
                parsed = json.loads(jsonContents)
                reports.extend(parsed['report'])

        iter_datetime = next_iter_datetime
    
    df = pd.DataFrame(reports)
    gdf = gpd.GeoDataFrame(df)

    if len(gdf) == 0:
        return None

    gdf = gdf.merge(gmeta, left_on='segment_id', right_on='oidn')
    gdf['data_source'] = 'telraam'
    gdf['measurement_type'] = 'point_measurement'
    gdf['speed'] = None
    gdf['locationrange'] = gdf['geometry'].apply(convert_to_linestring)

    # timestamp is the beginning of the measurement interval
    gdf.rename(columns={'date': 'timestamp'}, inplace=True)

    # apply street mapping
    gdf.set_index('segment_id', inplace=True)
    gdf['refRoadSegment'] = street_mapping['wegenregister_street_id']
    if gdf['refRoadSegment'].isnull().any():
        raise Exception('Missing mapping for some segments')
    gdf.reset_index(inplace=True)

    value_vars = ['car', 'bike', 'pedestrian', 'heavy']
    melted = gdf.melt(id_vars=gdf.columns.difference(value_vars),
                      value_vars=value_vars,
                      var_name='modality',
                      value_name='count'
                      )

    melted['index'] = melted['modality'] + \
        '_' + melted['segment_id'].astype(str)
    return melted[[
        'data_source',
        'index',
        'timestamp',
        'modality',
        'count',
        'speed',
        'measurement_type',
        'refRoadSegment',
        'locationrange'
    ]]


def merge_all_files(data_folder, street_mapping):
    chunks = []
    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)
        chunk = project(filepath, street_mapping)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


if __name__ == '__main__':
    """
    adapt the folders and filepaths to the correct location on your machine to make it work
    """
    # folder with the scraped files
    scan_folder = 'data/managed_data_files/mobiele_stad/datasources/telraam_crawled/telraam/raw/'
    # file for the results
    merged_filepath = 'data/managed_data_files/mobiele_stad/2020_analysis/data_preparation/telraam/counts_telraam_dec.csv'
    # csv with columns "oidn", "WS_OIDN" with the oidn's from the telraam sources and the corresponding WS_OIDN (roadregister id) of the streetsegment it's on
    street_mapping_filepath = 'data/managed_data_files/mobiele_stad/datasources/telraam_crawled/mapping.csv'

    street_mapping = pd.read_csv(street_mapping_filepath) \
        .rename(columns={'oidn': 'telraam_segment_id', 'WS_OIDN': 'wegenregister_street_id'}) \
        .set_index('telraam_segment_id')


    # df = merge_all_files(scan_folder, street_mapping)
    df = project(scan_folder, street_mapping)
    df.to_csv(merged_filepath, index=False)
