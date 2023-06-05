from datetime import timedelta

import numpy as np
import pandas as pd


def convert_kmh_speed_to_ms(value):
    return value / 3.6


# values extracted from 202012_Validation_report.docx
MINIMAL_ACCEPTED_SPEED_FACTOR = 0.66
MAXIMAL_ACCEPTED_SPEED_FACTOR = 1.5

# values extracted from 202012_Validation_report.docx
# in meters per second
VEHICLE_TYPE_REFERENCE_SPEED_MS = {
    'pedestrian': convert_kmh_speed_to_ms(5.4),
    'bike': convert_kmh_speed_to_ms(11.88),
    'car': convert_kmh_speed_to_ms(13.7),
    'truck': convert_kmh_speed_to_ms(13.7),
    'publicTransport': convert_kmh_speed_to_ms(13.7)
}


def get_accepted_speed_interval(vehicle_type):
    '''
    Returns the accepted speed interval for given vehicle type

    Parameters:
    vehicle_type (string): The vehicle type

    Returns:
    res (dict): The accepted speed interval, in meters per second
    res['min'] (float): The minimal accepted speed
    res['max'] (float): The maximal accepted speed
    '''

    reference_speed = VEHICLE_TYPE_REFERENCE_SPEED_MS[vehicle_type]
    return {
        'min': reference_speed * MINIMAL_ACCEPTED_SPEED_FACTOR,
        'max': reference_speed * MAXIMAL_ACCEPTED_SPEED_FACTOR,
    }


def read_validation_data(file_location, upsampling_frequency='5min', modality_mapping=None):
    '''
    Loads, upsamples and formats validation reference data.

    Arguments:
    - file_location (str): path to the validation data
    - upsampling_frequency (str): frequency of the upsampling
    - modality_mapping (dict | None): if provided, the modality is inferred from the modality_mapping dict, otherwise the modality will be the vehicle_type

    Returns a DataFrame with the following columns:
    - timestamp: datetime of the validation point
    - street_object_id: id of the road
    - modality: modality of the measurement
    - density_lower_bound: lower bound of the density acceptance interval
    - density_upper_bound: upper bound of the density acceptance interval
    '''

    df = pd.read_csv(
        file_location, dtype={'objectId': int})
    df.rename(columns={'objectId': 'street_object_id'}, inplace=True)

    # handle counting window time bounds
    df['start'] = pd.to_datetime(
        df['start'], origin='unix', unit='ms').dt.tz_localize(
        'utc').dt.tz_convert('Europe/Brussels')
    df['end'] = pd.to_datetime(
        df['end'], origin='unix', unit='ms').dt.tz_localize(
        'utc').dt.tz_convert('Europe/Brussels')
    df['duration'] = df['end'] - df['start']

    # filter out validation data points where the time window is smaller than a threshold
    keep = df['duration'] > timedelta(minutes=4, seconds=55)
    df = df[keep]

    # unpivot counts columns from wide to long
    df = pd.melt(df, id_vars=['start', 'end', 'duration', 'street_object_id'], value_vars=['car', 'pedestrian', 'bike', 'truck', 'publicTransport'],
                 var_name='vehicle_type', value_name='count')

    # transform counts into intensities
    df['intensity'] = df['count'] / df['duration'].dt.seconds

    # compute the acceptable densities bounds
    # the theory tells that:
    #   density = intensity / speed
    # for more information, read https://imec.atlassian.net/wiki/spaces/CF/pages/4485971981/Validation+concerns#The-functional-issue-with-the-validation-procedure
    acceptable_speeds = df['vehicle_type'].apply(
        lambda vehicle_type: pd.Series(data=get_accepted_speed_interval(vehicle_type)))
    df['density_lower_bound'] = df['intensity'] / acceptable_speeds['max']
    df['density_upper_bound'] = df['intensity'] / acceptable_speeds['min']

    # change of point of view, from a count measurement time interval to a point in time density
    tmp = df.melt(id_vars=['street_object_id', 'vehicle_type', 'density_lower_bound', 'density_upper_bound'], value_vars=[
                  'start', 'end'], var_name='timestamp_type', value_name='timestamp')

    tmp['timestamp'] = tmp['timestamp'].dt.floor(upsampling_frequency)

    # upsample to provided frequency
    def resampler(df):
        # timestamp_type is either 'start' or 'end', hence the following line eliminates the rows with duplicated timestamps by keeping the row
        # associated to timestamp_type == 'start'
        tmp = df.sort_values(by=['timestamp', 'timestamp_type']).drop_duplicates(
            subset='timestamp', keep='last')
        tmp = tmp.set_index('timestamp').resample(
            upsampling_frequency).ffill().reset_index()
        return tmp[tmp['timestamp_type'] == 'start'].drop(columns='timestamp_type')

    resampled = []
    index = tmp[['street_object_id', 'vehicle_type']].drop_duplicates()
    for tuple in index.itertuples():
        resampled_subset = resampler(
            tmp[(tmp['street_object_id'] == tuple.street_object_id) & (tmp['vehicle_type'] == tuple.vehicle_type)])
        resampled.append(resampled_subset)
    df = pd.concat(resampled)

    # map the modalities from car, pedestrian, bike, lorry onto aggregations
    if (modality_mapping is not None):
        lookup = {}
        for modality, modality_config in modality_mapping.items():
            if 'raw_modalities' in modality_config:
                for vehicle_type in modality_config['raw_modalities']:
                    lookup[vehicle_type] = modality

        df['modality'] = df['vehicle_type'].apply(
            lambda vehicle_type: lookup[vehicle_type])

        # grouping by on the introduced categories
        df = df.groupby(by=['timestamp',
                            'modality', 'street_object_id']).sum()
        df.reset_index(inplace=True)

    else:
        df['modality'] = df['vehicle_type']

    return df[['timestamp', 'street_object_id', 'modality', 'density_lower_bound', 'density_upper_bound']]


def read_densities(filepath, time_bounds=None):
    '''
    Returns a DataFrame with the following columns:
    - timestamp: datetime of the measurement
    - street_object_id: id of the road
    - modality: modality of the measurement
    - density: density of the measurement for the modality
    '''

    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if time_bounds is not None:
        df = df[df['timestamp'].between(
            time_bounds["min"], time_bounds["max"])]
    id_cols = ['timestamp', 'street_object_id']
    densities_columns = df.columns.difference(id_cols)
    data = df.melt(id_vars=id_cols, value_vars=densities_columns,
                   var_name='modality', value_name='density')
    return data


def read_streets(file_location):
    '''
    Returns a DataFrame with the following columns:
    - street_object_id: id of the road
    - street_object_length: length of the road
    '''

    streets = pd.read_csv(file_location)
    df = streets[['street_object_id', 'street_object_length']].copy()
    df.drop_duplicates(inplace=True)
    return df


def add_street_length(dataframe, streets):
    '''
    Returns a dataframe with an aadded street length column, based on street lengths found in the street ref file.

    Parameters:
    dataframe: dataframe with a column 'street_object_id' that we want to enrich with street length
    streets: a reference dataframe carrying information on the street lengths

    Returns:
    dataframe: the dataframe given as input appended with extra street_object_length and relative_street_length columns
    '''
    df = dataframe.set_index('street_object_id')
    streets = streets.set_index('street_object_id')
    df['street_object_length'] = streets['street_object_length']
    df.reset_index(inplace=True)
    streets.reset_index(inplace=True)

    if df['street_object_length'].isna().any():
        raise Exception(
            'Some street lengths were not found in the streets dataframe.')

    validation_street_ids = dataframe['street_object_id'].unique()
    validation_streets = streets.loc[streets['street_object_id'].isin(
        validation_street_ids)]
    total_street_length = validation_streets['street_object_length'].sum()

    df['relative_street_length'] = df['street_object_length'] / total_street_length
    return df


def validate(validation, densities, streets, street_object_id=None, modality=None):
    '''
    validation: dataframe with validation data
    densities: dataframe with densities coming out of the CityFlows model
    streets: dataframe with street length information
    street_object_id (None | int): The id of the road segment to perform the validation for. If None is provided, validation on all road segments
    modality (None | string): The modality to perform the validation for. If None is provided, validation on all modalities

    Returns:
    (float, dict): tuple containing an overall validation score and a dict of floats between 0 and 1 for each modality. When a modality is provided as argument, then overall score is the modality score, otherwise it's the average of the modality scores
    '''

    if street_object_id is not None:
        densities = densities[densities['street_object_id']
                              == street_object_id]
    if modality is not None:
        densities = densities[densities['modality'] == modality]

    df = densities.merge(
        validation, on=['timestamp', 'street_object_id', 'modality'])

    if len(df) == 0:
        return None

    # REMARK this function can, if necessary, be improved. Currently it is a simple binary measure (within bounds, out bounds)
    def withinBounds(row):
        # helper function to apply to a dataframe, checking whether datapoint is within bounds
        if (row.density <= row.density_upper_bound) and (row.density >= row.density_lower_bound):
            return 1.0
        else:
            return 0.0

    df['withinBounds'] = df.apply(func=withinBounds, axis=1)

    # group by logic: per street, per modality
    grouped_by_street_and_modality = df.groupby(
        by=['street_object_id', 'modality']).agg({'withinBounds': np.average})
    grouped_by_street_and_modality.reset_index(inplace=True)

    enriched = add_street_length(grouped_by_street_and_modality, streets)

    enriched['validation_score'] = enriched['withinBounds'] * \
        enriched['relative_street_length']

    # group all streets together for the full validation region:
    grouped_by_modality = enriched.groupby(
        by=['modality']).agg({'validation_score': sum})

    # compute the overall score, if a modality is provided as argument then this overall score is the modality score
    overall_score = grouped_by_modality['validation_score'].mean()

    # putting result in a dict object
    result = grouped_by_modality['validation_score'].to_dict()

    return overall_score, result


def test_validate():
    # write some unit tests here
    pass


if __name__ == '__main__':

    import os
    from src.model.modality import get_modality_objects

    # define input file paths

    validation_filepath = 'data/managed_data_files/mobiele_stad/validation_straatvinken/transformed/SV2020_DataVVR-Antwerp_20210422_transformed.csv'

    modality_mapping_filepath = 'src/model/modality_mapping.json'
    model_output_folder = 'data/managed_data_files/mobiele_stad/learning_cycle_3/output'
    densities_filepath = os.path.join(model_output_folder, 'densities.csv')
    streets_filepath = os.path.join(model_output_folder, 'streets.csv')

    # load files content

    modality_mapping = get_modality_objects(modality_mapping_filepath)
    validation = read_validation_data(
        validation_filepath, modality_mapping=modality_mapping)

    ts_min = validation['timestamp'].min()
    ts_max = validation['timestamp'].max()
    time_bounds = {'min': ts_min, 'max': ts_max}

    streets = read_streets(streets_filepath)
    densities = read_densities(densities_filepath, time_bounds)

    # compute validation score

    print(validate(densities, validation, streets))
