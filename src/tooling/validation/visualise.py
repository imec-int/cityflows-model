import os
from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
import folium

# use this to load and analyse data from validation.cityflows.io !


def load_data(path):
    # getting in the coordinates of the streets: #street_segments file carries this

    df_app = pd.read_csv(path, sep=';')
    df_app['method'] = 'app'
    # minor transformation of epoch towards datetime
    df_app['start'] = pd.to_datetime(df_app['start'] * 1000000)
    df_app['end'] = pd.to_datetime(df_app['end'] * 1000000)

    df_app['start'] = df_app['start'].dt.tz_localize('UTC').dt.round('S')
    df_app['end'] = df_app['end'].dt.tz_localize('UTC').dt.round('S')

    return df_app


def aggregate(df, group_by_columns):
    cols = group_by_columns + ['bike', 'velo', 'car',
                               'publicTransport', 'truck', 'pedestrian']
    selector = df[cols]
    df_agg = selector.groupby(group_by_columns).sum()
    df_agg.reset_index(inplace=True)

    return df_agg


def load_wegenregister_mapping(path):
    # data is an export from teh qgis wegenregister as csv. Columns required are WS_OIDN and geometry as Well Known Text WKT
    locations = pd.read_csv(path)
    locations['geometry'] = locations['WKT'].apply(
        lambda x: shapely.wkt.loads(x).centroid)
    locations.drop(columns=['WKT'], inplace=True)
    locations.rename(columns={'WS_OIDN': 'objectId'}, inplace=True)
    return locations


def attach_wegenregister_location(df, wegenregister_mapping):
    df_all = df.merge(wegenregister_mapping, how='left')
    df_all.dropna(subset=['geometry'], inplace=True)

    # dealing with projections:
    LAMBERT_CRS_ID = 31370
    gdf = gpd.GeoDataFrame(df_all, geometry='geometry')
    gdf.set_crs(epsg=LAMBERT_CRS_ID, inplace=True)

    GPS_CRS_ID = 4326
    gdf.to_crs(epsg=GPS_CRS_ID, inplace=True)

    # get x, y coordinates
    gdf['lat'] = gdf['geometry'].apply(lambda p: p.x)
    gdf['lon'] = gdf['geometry'].apply(lambda p: p.y)

    return gdf


def apply_coloring(gdf, column_name):
    lower = gdf[column_name].min()
    upper = gdf[column_name].max()
    cmap = sns.color_palette('flare', as_cmap=True)

    def rgb_2_hex(float_colors):
        int_colors = [int(256 * c) for c in float_colors]
        hexes = [hex(c)[-2:] for c in int_colors]
        return '#' + ''.join(hexes)

    def assign_color(row):
        if row[column_name] < lower:
            return {'fill_color': 'white', 'fill_opacity': 0, 'stroke_color': 'black', 'stroke_opacity': 1}

        f = (row[column_name] - lower) / (upper - lower)

        r, g, b, _ = cmap.__call__(f)
        color = rgb_2_hex([r, g, b])
        return {'fill_color': color, 'fill_opacity': 0.7, 'stroke_color': color, 'stroke_opacity': 1}

    coloring = gdf.apply(func=assign_color, axis=1, result_type='expand')
    data = gdf.join(coloring)
    return data


def get_folium_map(data):
    map_bounds = np.flip(np.reshape(
        data.total_bounds, (2, 2)), axis=1).tolist()

    m = folium.Map()
    m.fit_bounds(map_bounds)
    for row in data.itertuples():
        folium.Circle(location=[row.lon, row.lat], radius=100, color=row.stroke_color, fill_color=row.fill_color,
                      opacity=row.stroke_opacity, fill_opacity=row.fill_opacity, popup=f'velo_pct: {row.velo_pct}').add_to(m)
    return m


if __name__ == "__main__":
    location = 'src/tooling/visualise_validation_data/input'
    filename_result = 'cf_data_validator_antwerp_production_export.csv'
    path = os.path.join(location, filename_result)

    df = load_data(path)
    df_agg = aggregate(df, ['objectId'])

    # specific test case for this data, in general anything can be done
    def calculate_velo_pct(row):
        try:
            result = row.velo / row.bike
            #print('calculated', result)
            return result
        except:
            return np.nan
    df_agg['velo_pct'] = pd.Series(calculate_velo_pct(
        tuple) for tuple in df_agg.itertuples())

    location_mapping = 'src/tooling/visualise_validation_data/input'
    filename_mapping = 'groot_antwerpen.csv'
    path = os.path.join(location_mapping, filename_mapping)

    wegenregister_mapping = load_wegenregister_mapping(path)
    gdf = attach_wegenregister_location(df_agg, wegenregister_mapping)
    data = apply_coloring(gdf, 'velo_pct')
    m = get_folium_map(data)
