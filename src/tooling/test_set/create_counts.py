import argparse
import geopandas as gpd
import os
import pandas as pd
import shapely
import sys

"""
Writes a subset of a data csv based on the data sources using the given coordinates or on the given indices
"""

# arguments handling
parser = argparse.ArgumentParser()
parser.add_argument('--latitudes', nargs=2, type=float)
parser.add_argument('--longitudes', nargs=2, type=float)
parser.add_argument('--indices', nargs='+', type=int)

args = parser.parse_args()
latitudes = args.latitudes
longitudes = args.longitudes
indices = args.indices

# determine the mode of execution, priority is given to indices mode
INDICES_MODE = 'indices'
BOUNDS_MODE = 'bounds'

mode = None
if indices is not None:
    mode = INDICES_MODE
elif latitudes is not None and longitudes is not None:
    mode = BOUNDS_MODE

if mode is None:
    print('Forgot to specify either indices or latitudes and longitudes')
    sys.exit(1)

# actual computation
df = pd.read_csv(
    'data/managed_data_files/AAA/input/counts/cropland_2020_01_01.csv')
df['geometry'] = df['locationrange'].apply(lambda x: shapely.wkt.loads(x))
gdf = gpd.GeoDataFrame(df)


if mode == INDICES_MODE:
    gdf = gdf[gdf['index'].isin(indices)]
elif mode == BOUNDS_MODE:
    min_x = min(*longitudes)
    max_x = max(*longitudes)
    min_y = min(*latitudes)
    max_y = max(*latitudes)
    target_polygon = shapely.geometry.box(min_x, min_y, max_x, max_y)
    gdf = gdf[gdf.intersects(target_polygon)]

output_filepath = 'data/managed_data_files/test_set_bigger/input/counts/data.csv'
os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
gdf.drop(['geometry'], axis='columns').to_csv(output_filepath)
