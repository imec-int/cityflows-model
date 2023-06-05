import pandas as pd
import geopandas as gpd
import os
import shapely

THIS_DIR = os.path.dirname(__file__)

def load_street_segments(input_path):
    """Reads in street segments files that have been cut from disk.

    Args:
        input path: location on disk where csv file is located

    Returns:
        street geo dataframe containing only relevant information

    Raises:
        --
    """

    columns = [
        'street_segment_id',
        'street_object_id',
        'street_segment_length',
        'street_object_length',
        'data_source',
        'data_source_index',
        'is_edge',
        'street_segment_geometry',
        'street_object_geometry'
    ]

    df_streets = pd.read_csv(input_path, usecols=columns, dtype={'data_source_index': str})
    df_streets['street_segment_geometry'] = df_streets['street_segment_geometry'].apply(shapely.wkt.loads)
    df_streets['street_object_geometry'] = df_streets['street_object_geometry'].apply(shapely.wkt.loads)
    gdf_streets = gpd.GeoDataFrame(df_streets, geometry='street_segment_geometry')
    return gdf_streets

def get_streets_metadata(street_segments):
    return street_segments.drop_duplicates(['street_object_id'])[['street_object_id', 'street_object_length', 'street_object_geometry']]

def add_edge_intersections(street_segments, intersections):
    #not necessary yet
    pass

if __name__ == "__main__":
    ##
    repo_path = os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir) #3 levels up
    input_segments_path = os.path.join(repo_path, 'data/managed_data_files/mobiele_stad/learning_cycle_1/road_cutter_output_refactored/')
    input_segments_file = 'street_segments.csv'
    input_segments_full_path = os.path.join(input_segments_path, input_segments_file)

    df = load_street_segments(input_segments_full_path)
    print(df.head())