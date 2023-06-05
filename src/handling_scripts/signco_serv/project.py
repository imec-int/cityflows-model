import os
import geopandas as gpd
import pandas as pd
import json


def extract_observations(filepath):

    f = open(filepath)
    content = json.load(f)

    # parse data into a dataframe / JSON file

    n_observations = len(content['data']) if 'data' in content else 0

    result = {}
    result['device'] = [0] * n_observations
    result['timestamp'] = [0] * n_observations
    result['modality'] = [0] * n_observations
    result['speed'] = [0] * n_observations
    for i in range(0, n_observations):
        result['device'][i] = filepath.split('_')[-1][:-5]
        result['timestamp'][i] = content['data'][i]['timestamp']
        result['modality'][i] = content['data'][i]['type']
        result['speed'][i] = content['data'][i]['speed']

    df = pd.DataFrame.from_dict(data=result)
    return df


def aggregate_observations(df, period):
    df['timestamp_rounded'] = pd.to_datetime(df['timestamp']).dt.floor(period)

    intensities = df.groupby(by=['device', 'modality', 'timestamp_rounded']).aggregate(
        {'speed': 'mean', 'timestamp': 'count'}).reset_index().rename(columns={'timestamp': 'count', 'timestamp_rounded': 'timestamp'})
    return intensities


def final_transformation(df, street_mapping, street_geometry_mapping):
    df['data_source'] = 'signco_fietstellus'
    df['measurement_type'] = 'point_measurement'
    df['speed'] = df['speed'] / 3.6
    df['index'] = df['device']

    # apply street mapping
    df['refRoadSegment'] = df['device'].map(street_mapping)
    print(len(df['refRoadSegment'].isnull()))
    print(len(df))
    df=df.dropna(subset=['refRoadSegment'])
    if df['refRoadSegment'].isnull().any():
        raise Exception('Missing street mapping for some devices')

    # apply geometry mapping
    df['locationrange'] = df['refRoadSegment'].map(street_geometry_mapping)

    if df['locationrange'].isnull().any():
        raise Exception('Missing geometry mapping for some segments')

    return df[[
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


def project_all_files(data_folder, street_mapping, street_geometry_mapping):
    all_observations = []
    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)
        observations = extract_observations(filepath)
        all_observations.append(observations)

    all_observations_df = pd.concat(all_observations, ignore_index=True)

    # map moped to bikes
    all_observations_df['modality'] = 'bike'

    intensities = aggregate_observations(all_observations_df, '5min')

    df = final_transformation(
        intensities, street_mapping, street_geometry_mapping)
    return df


if __name__ == '__main__':
    from shapely.wkt import loads

    """
    adapt the folders and filepaths to correct versions to use them
    """
    # folder with the scraped files
    scan_folder = 'data/scraping/signco_serv/vehicleDetails'
    # folder for results
    result_filepath = 'data/managed_data_files/mobiele_stad/2020_analysis/data_preparation/signco/counts_signco.csv'
    # street_mapping: a csv file containing the following columns 'index' & 'refRoadSegment'. These are respectively the index of the signco device and the segment_id (WS_OIDN from the roadregister) of the street it's in.
    street_mapping_filepath = 'data/managed_data_files/mobiele_stad/datasources/singco/signco_new_mapping.csv'
    # road registry geojson containing info about the area of interest:
    streets_filepath = 'data/managed_data_files/shapefiles/Wegenregister_geojson/wegenregister_antwerp.geojson'

    street_mapping = pd.read_csv(
        street_mapping_filepath).set_index('index')['refRoadSegment']

    streets = gpd.read_file(streets_filepath)
    street_geometry_mapping = streets.set_index('WS_OIDN')['geometry']

    df = project_all_files(scan_folder, street_mapping,
                           street_geometry_mapping)
    df.to_csv(result_filepath, index=False)

    print(df.head())
    print(len(df))
