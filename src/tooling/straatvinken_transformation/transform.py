from datetime import datetime
import os
from uuid import uuid4 as uuid

from fuzzywuzzy import fuzz
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# used to log progress
progress = 0

# in meters
NEARBY_DISTANCE_THRESHOLD = 5

# between 0 (lowest confidence) and 100 (highest confidence)
HIGH_CONFIDENCE_THRESHOLD = 80

def get_nearby_streets(straatvinken_row, streets):
    distances = streets.distance(straatvinken_row['geometry'])
    selection = distances <= NEARBY_DISTANCE_THRESHOLD
    nearby_streets = streets[selection]
    nearby_streets['distance'] = distances[selection]
    return nearby_streets

def get_confident_streets(straatvinken_row, streets):
    confidences = streets.apply(
        func=lambda street: max(
            fuzz.token_set_ratio(street['LSTRNM'], straatvinken_row['Streetname']),
            fuzz.token_set_ratio(street['RSTRNM'], straatvinken_row['Streetname']),
        ),
        axis=1
    )    
    selection = confidences >= HIGH_CONFIDENCE_THRESHOLD
    streets_high_confidence = streets[selection]
    streets_high_confidence['confidence'] = confidences[selection]
    return streets_high_confidence


def transform(input_filepath, output_filepath, streets_filepath, window_start_time, window_end_time, include_debug_outputs=False):
    df_straatvinken = pd.read_csv(input_filepath)
    geometries = df_straatvinken.apply(
        func=lambda row: Point(row['Long'], row['Lat']),
        axis=1
    )
    straatvinken = gpd.GeoDataFrame(df_straatvinken, geometry=geometries, crs=4326)

    # the execution takes around 25-30 minutes when using the full straatvinken dataframe
    # you can use the following subsets to gain development speed

    # straatvinken = straatvinken.head(10)
    # straatvinken = straatvinken[straatvinken['Municipality'] == 'Aartselaar']

    streets = gpd.read_file(streets_filepath)

    # set crs to EPSG:3310 so that distance are computed in meters
    # as advised here https://gis.stackexchange.com/questions/293310/how-to-use-geoseries-distance-to-get-the-right-answer
    straatvinken.to_crs(crs=3310, inplace=True)
    streets.to_crs(crs=3310, inplace=True)

    def map_match(row):
        global progress
        progress = progress + 1

        if progress % 10 == 0:
            print('%.1f %% done' % (100 * progress / straatvinken.shape[0]))

        res = row.copy()
        
        # only keep nearby streets, this is an optimization to reduce the amount
        # of streets processed in the following steps
        nearby_streets = get_nearby_streets(row, streets)
        n_nearby_streets = nearby_streets.shape[0]

        if n_nearby_streets == 0:
            res['drop'] = True
            res['drop_reason'] = 'No streets within %d meters' % NEARBY_DISTANCE_THRESHOLD
            return res

        # only keep nearby streets for which the street name in the shapefile is confidently 
        # close enough to the street name referenced in the straatvinken row
        streets_high_confidence = get_confident_streets(row, nearby_streets)
        n_streets_high_confidence = streets_high_confidence.shape[0]
        
        if n_streets_high_confidence == 0:
            res['drop'] = True
            res['drop_reason'] = 'Inconfident name matching'
            return res

        # exclude ambiguous locations
        # (we could try disambiguating a bit more but that's not worth the effort since we are
        # not dropping that many straatvinken rows)
        if n_streets_high_confidence > 1:
            res['drop'] = True
            res['drop_reason'] = 'Ambiguous location'
            return res

        # pick the closest street in the remaining streets
        arg_min = streets_high_confidence['distance'].argmin()
        street = streets_high_confidence.iloc[arg_min, :]
    
        res['refRoadSegment'] = street['WS_OIDN']
        res['distance'] = street['distance']
        res['confidence'] = street['confidence']
        res['drop'] = False

        return res

    straatvinken_map_matched = straatvinken.apply(func=map_match, axis=1)
    straatvinken_dropped = straatvinken_map_matched[straatvinken_map_matched['drop'] == True]
    straatvinken_enriched = straatvinken_map_matched[straatvinken_map_matched['drop'] == False]
    print('Kept %.1f%% of straatvinken locations' % (100 * straatvinken_enriched.shape[0] / straatvinken.shape[0]))

    sessionId = uuid()
    df_res = pd.DataFrame()

    df_res['car'] = straatvinken_enriched['Car'] + straatvinken_enriched['Van']
    df_res['pedestrian'] = straatvinken_enriched['Walk']
    df_res['bike'] = straatvinken_enriched['Bike']
    df_res['truck'] = straatvinken_enriched['Truck']
    df_res['publicTransport'] = straatvinken_enriched['Bus']

    df_res['start'] = window_start_time
    df_res['end'] = window_end_time

    df_res['sessionId'] = sessionId
    
    df_res['objectId'] = straatvinken_enriched['refRoadSegment']
    
    df_res['trafficPressure'] = ''
    df_res['feedback'] = ''

    df_res.to_csv(output_filepath, index=False)
    
    if include_debug_outputs:
        [common_filepath, file_extension] = os.path.splitext(output_filepath)
        
        dropped_filepath = common_filepath + '_dropped' + file_extension
        enriched_filepath = common_filepath + '_enriched' + file_extension
        
        straatvinken_dropped.to_csv(dropped_filepath)
        straatvinken_enriched.to_csv(enriched_filepath)


if __name__ == '__main__':
    folder = 'data/mobiele_stad/validation_straatvinken'
    input_file = 'SV2020_DataVVR-Antwerp_20210422.csv'
    output_file = 'SV2020_DataVVR-Antwerp_20210422_transformed.csv'
    
    street_segments_filepath = 'data/shapefiles/Wegenregister_straatvinken/street_segments.shp'

    window_start_time = int(datetime.fromisoformat("2020-05-14T17:00:00+02:00").timestamp() * 1000)
    window_end_time = int(datetime.fromisoformat("2020-05-14T18:00:00+02:00").timestamp() * 1000)
    
    input_filepath = os.path.join(folder, input_file)
    output_filepath = os.path.join(folder, output_file)
    transform(input_filepath, output_filepath, street_segments_filepath, window_start_time, window_end_time, include_debug_outputs=True)
