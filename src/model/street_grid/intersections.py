import pandas as pd
import geopandas as gpd
import os
import shapely

THIS_DIR = os.path.dirname(__file__)

def load_intersections(input_path):
    """Reads in intersections files from disk.

    Args:
        input path: location on disk where csv file is located

    Returns:
        intersections geo dataframe containing only relevant information

    Raises:
        --
    """

    columns = [
        'intersection_id',
        'street_object_id',
        'street_segment_id',
        'end_type',
        'intersection_type',
        'is_edge',
        'geometry'
    ]

    df_intersections = pd.read_csv(input_path, usecols=columns)
    df_intersections['geometry'] = df_intersections['geometry'].apply(shapely.wkt.loads)
    gdf_intersections = gpd.GeoDataFrame(df_intersections)
    return gdf_intersections

if __name__ == "__main__":
    repo_path = os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir) #3 levels up
    input_segments_path = os.path.join(repo_path, 'data/managed_data_files/mobiele_stad/learning_cycle_1/road_cutter_output_refactored/')
    input_segments_file = 'intersections.csv'
    input_segments_full_path = os.path.join(input_segments_path, input_segments_file)

    df = load_intersections(input_segments_full_path)
    print(df.head())