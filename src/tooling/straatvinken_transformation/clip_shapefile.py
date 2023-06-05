import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

def compute_straatvinken_data_bbox(input_filepath):
    df = pd.read_csv(input_filepath)
    min_lat = df['Lat'].min()
    max_lat = df['Lat'].max()
    min_lon = df['Long'].min()
    max_lon = df['Long'].max()
    return {
        'min_lon': min_lon,
        'max_lon': max_lon,
        'min_lat': min_lat,
        'max_lat': max_lat,
    }

def bbox_to_polygon(bbox):
    p1 = [bbox['min_lon'], bbox['min_lat']]
    p2 = [bbox['min_lon'], bbox['max_lat']]
    p3 = [bbox['max_lon'], bbox['max_lat']]
    p4 = [bbox['max_lon'], bbox['min_lat']]
    return Polygon([p1, p2, p3, p4, p1])

def clip_shapefile(shapefile_filepath, output_shapefile_filepath, bbox):
    polygon = bbox_to_polygon(bbox)

    df = gpd.read_file(shapefile_filepath).to_crs("EPSG:4326")
    res = df[df.intersects(polygon)]
    res.to_file(output_shapefile_filepath)

if __name__ == '__main__':
    straatvinken_filepath = 'data/mobiele_stad/validation_straatvinken/SV2020_DataVVR-Antwerp_20210422.csv'

    input_shapefile_filepath = 'data/shapefiles/Wegenregister_SHAPE_20210617/Shapefile/Wegsegment.shp'
    output_shapefile_filepath = 'data/shapefiles/Wegenregister_straatvinken/street_segments.shp'

    bbox = compute_straatvinken_data_bbox(straatvinken_filepath)
    clip_shapefile(input_shapefile_filepath, output_shapefile_filepath, bbox)