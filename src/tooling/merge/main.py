import argparse
import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, GeometryCollection
from shapely.wkt import loads as loads_wkt


"""
creates an outputfile ( usually named all_data.csv) by combining the count (or already all_data) files in the CityFlows dataformat (from one or more different data sources).
"""

def load_input_file(filepath):
    all_cols = set(['data_source', 'index', 'timestamp', 'modality', 'count',
                   'speed', 'measurement_type', 'refRoadSegment', 'locationrange'])
    mandatory_cols = all_cols - set(['refRoadSegment'])

    dtypes = {
        'data_source': str,
        'index': str,
        'refRoadSegment': pd.Int64Dtype()
    }
    try:
        return pd.read_csv(filepath, usecols=all_cols, dtype=dtypes)
    except:
        return pd.read_csv(filepath, usecols=mandatory_cols, dtype=dtypes)


def get_filtering_ids(filter_ids):
    if filter_ids is None:
        return None

    filters = {}
    for filter_id in filter_ids:
        items = filter_id.split('=')
        if len(items) != 2:
            raise Exception('Invalid ids filter: {}'.format(filter_id))

        key, values = items
        values = values.split(',')
        values = map(lambda value: value.strip(), values)
        values = [value for value in values if len(value) > 0]
        if len(values) == 0:
            raise Exception(
                'No valid values found for filter: {}'.format(filter_id))

        filters[key] = values
    return filters


def get_filtering_time_bounds(min_timestamp, max_timestamp):
    return {
        'min': pd.to_datetime(min_timestamp, utc=True) if min_timestamp is not None else None,
        'max': pd.to_datetime(max_timestamp, utc=True) if max_timestamp is not None else None
    }


def get_filtering_area(area_file):
    if area_file is None:
        return None

    with open(area_file) as f:
        try:
            features = json.load(f)["features"]

            # buffer(0) is a trick for fixing scenarios where polygons have overlapping coordinates
            area = GeometryCollection(
                [shape(feature["geometry"]).buffer(0) for feature in features])
            return area
        except:
            raise Exception(
                'Invalid area file format: {}, expected a GeoJSON feature collection'.format(area_file))


def apply_filters(data, telco_data_source, filtering_ids, filtering_time_bounds, filtering_area):
    counts = data

    if filtering_time_bounds['min'] is not None:
        counts = counts[counts['timestamp'] >= filtering_time_bounds['min']]
    if filtering_time_bounds['max'] is not None:
        counts = counts[counts['timestamp'] <= filtering_time_bounds['max']]

    if filtering_ids is not None:
        keep = [
            tuple.data_source in filtering_ids and
            tuple.index in filtering_ids[tuple.data_source]
            for tuple in counts.itertuples()
        ]
        counts = counts[keep]

    if filtering_area is not None:
        counts['locationrange'] = counts['locationrange'].apply(loads_wkt)
        gdf = gpd.GeoDataFrame(counts, geometry='locationrange')

        telco_gdf = gdf[gdf['data_source'] == telco_data_source]
        other_gdf = gdf[gdf['data_source'] != telco_data_source]

        telco_data = telco_gdf[telco_gdf.intersects(filtering_area)]
        telco_boundary = telco_data.unary_union
        other_data = other_gdf[other_gdf.within(telco_boundary)]

        counts = pd.concat([telco_data, other_data])

    return counts


def merge_data_files(input_files, output_file, telco_data_source, filter_ids, min_timestamp, max_timestamp, area_file):
    dfs = map(load_input_file, input_files)
    input = pd.concat(dfs)
    input['timestamp'] = pd.to_datetime(input['timestamp'], utc=True)

    filtering_ids = get_filtering_ids(filter_ids)
    filtering_time = get_filtering_time_bounds(min_timestamp, max_timestamp)
    filtering_area = get_filtering_area(area_file)
    output = apply_filters(input, telco_data_source, filtering_ids,
                           filtering_time, filtering_area)

    output.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python -m src.tooling.merge.main',
        description='Merge two all_data data files'
    )
    parser.add_argument('--input_files', help='Input files',
                        nargs='+', required=True)
    parser.add_argument(
        '--output_file', help='Output file path', required=True)
    parser.add_argument('--telco_data_source',
                        help='Name of the datasource providing the telco data', required=True)
    parser.add_argument('--filter_ids', help='Ids for cells filtering. Multiple entries can be provided, separated by a space. Each entry should be formatted as <data source name>=<comma separated list of ids>. For example: --filter_ids cropland=123,456,789 velo=789', nargs='+')
    parser.add_argument(
        '--min_ts', help='Temporal filtering by providing min datetime, included in output')
    parser.add_argument(
        '--max_ts', help='Temporal filtering by providing max datetime, included in output')
    parser.add_argument('--area', help='Area for spatial filtering')
    args = vars(parser.parse_args())

    merge_data_files(args['input_files'], args['output_file'], args['telco_data_source'],
                     args['filter_ids'], args['min_ts'], args['max_ts'], args['area'])
