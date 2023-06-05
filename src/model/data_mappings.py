import math
import os
import pickle

import pandas as pd
from src.model.street_grid.street_segments import load_street_segments
from src.utils.time import transformTimestamp

from .modality import get_modality_objects

THIS_DIR = os.path.dirname(__file__)
temp_dir = os.path.join(THIS_DIR, 'temp')


def interpolate_missing_counts(counts):
    data_source_cells = counts.drop_duplicates(
        subset=['data_source', 'data_source_index', 'modality'])
    chunks = []
    for data_source_cell in data_source_cells.itertuples():
        counts_df = counts[(counts['data_source'] == data_source_cell.data_source) & (
            counts['data_source_index'] == data_source_cell.data_source_index) & (
            counts['modality'] == data_source_cell.modality)]
        counts_series = counts_df.set_index('timestamp')['count'].sort_index()
        counts_series.interpolate(
            method='time', inplace=True, limit_area='inside')

        df = counts_series.to_frame().reset_index()
        df['data_source'] = data_source_cell.data_source
        df['data_source_index'] = data_source_cell.data_source_index
        df['modality'] = data_source_cell.modality

        chunks.append(df)

    res = pd.concat(chunks)
    return res


def add_modality_mapping_to_data(data, modality_mapping):
    """Reads in data, transforms raw modalities into mapped modalities according to a given mapping

    Args:
        data: input dataframe, with a modality column
        modality_mapping: contains information on how to transform tha raw modality to the mapped modality

    Returns:
        adapted_data, a dataframe with a changed modality column
    Raises:
        --
    """

    print('Mapping modalities given to this set of final modalities: ',
          list(modality_mapping.keys()))
    data['modality'] = data['modality'].apply(
        lambda x: mapping_from_dict(x, modality_mapping))
    data['measurement_type'].fillna('count', inplace=True)

    # Making modality mapping independent from columns
    columns = data.columns.difference(['count', 'speed']).tolist()
    # the mean is mathematically not perfect, but it will do for now
    all_data_agg = data.groupby(by=columns).agg(
        {'count': 'sum', 'speed': 'mean'})
    all_data_agg.reset_index(inplace=True)

    return all_data_agg


def mapping_from_dict(argument, modality_mapping):
    if type(argument) == float and math.isnan(argument):
        # case for no modality
        return 'all'
    else:
        try:
            # search in dict for the raw modality
            for key in modality_mapping.keys():
                try:
                    if argument in modality_mapping[key]['raw_modalities']:
                        return key
                    else:
                        pass
                except:
                    pass
            return 'all'  # catch-all for
        except:
            print('Warning: a modality could not be correctly mapped.')
            return None


def add_modality_to_street_grid():
    # not yet needed
    return None

# preprocessing of data


def load_counts_data(input_path):
    """Reads in data, applies minor cleansing actions (drop duplicates, columns, timestamp snapping)

    Args:
        input_path: where data is located, with following input columns:
            count, data_source, modality, timestamp, index, locationrange (not necessarily in that order)

    Returns:
        loaded pandas dataframe in memory
    Raises:
        --
    """

    columns = [
        'count',
        'data_source',
        'index',
        'modality',
        'timestamp',
        'measurement_type',
        'speed'
    ]
    # explicitly force index and modality to string value
    all_data = pd.read_csv(
        input_path, dtype={'index': str, 'modality': str,'count':float}, usecols=columns)
    all_data.rename(columns={'index': 'data_source_index'}, inplace=True)

    # minor changes
    all_data.drop_duplicates(inplace=True)
    all_data['timestamp'] = all_data['timestamp'].apply(transformTimestamp)
    all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])
    return all_data


def add_missing_speeds_to_data(counts_data, modality_mapping):
    """fill in the missing speeds with the average speeds known from the modality mapping file.
        Assumptions: traffic from a modality is moving as a stream with average speed during the measurement interval

    Args:
        counts: dataframe with all counts, modalities must already be mapped!
        modality_mapping: for filling in of missing speeds

    Returns:
        counts: adapted columns based on measurement_type column
    Raises:
        Warning is modalities are not properly mapped
    """
    def apply_average_speed(modality):
        try:
            return modality_mapping[modality]['average_speed']
        except:
            pass
        return None

    # test if modalities are mapped, otherwise throw error
    modalities_in_data = set(counts_data['modality'].unique())
    modalities_in_config = set(modality_mapping.keys()).union(
        {'all'})  # all the keys from the mapping dictionary plus 'all'
    if not modalities_in_data.issubset(modalities_in_config):
        raise Warning(
            "Data given is not properly matched to modalities given in config file. Results might suffer.")

    # get missing speed from dictionary
    fallback_speeds = counts_data['modality'].apply(
        lambda x: apply_average_speed(x))
    # filling missing speeds with the fallback speed
    counts_data['speed'].fillna(fallback_speeds, inplace=True)

    return counts_data


def add_frequency_of_measurement(counts_data, measurement_type='point_measurement'):
    """Add an extra column indicating the time between previous measurement in seconds

    Args:
        counts: dataframe with all counts, modalities must already be mapped!

    Returns:
        counts: added frequency column (expressed in seconds)
    Raises:
        -- Warning: if some data sources only contain 1 data point, the frequency cannot be accurately computed
    """

    counts_data['frequency'] = None
    frequency_change_needed = counts_data.loc[counts_data['measurement_type']
                                              == measurement_type]
    no_change_needed = counts_data.loc[counts_data['measurement_type']
                                       != measurement_type]

    # assuming a constant frequency of measurements, we measure for each data source the minimal difference between timesteps.
    # This value is taken to be the update frequency. This way, outages of data sources do not influence the calculation.
    # if nothing can be computed, a hardcode 1-HOUR = 3600 seconds frequency will be assumed.

    def measure_frequency(slice):
        ordered_slice = slice.sort_values(by='timestamp')
        try:
            local_freq = int(
                ordered_slice['timestamp'].diff().min().total_seconds())
            # print('LOCAL FREQ:', local_freq)
            slice['frequency'] = local_freq
            return slice
        except:
            # cannot be computed, we take the hourly standard
            slice['frequency'] = 3600
            raise Warning(
                'Some data has less than 2 timesteps, making it impossible to calculate the frequency of updates.')
            return slice

    # TODO the code below can probably be somehow vectorized
    results = []  # will contain all slices

    data_sources = list(frequency_change_needed['data_source'].unique())
    for ds in data_sources:
        selected_df = frequency_change_needed.loc[frequency_change_needed['data_source'] == ds]
        indices = selected_df['data_source_index'].unique()
        for index in indices:
            selected_index_df = selected_df.loc[selected_df['data_source_index'] == index]
            modalities = selected_index_df['modality'].unique()
            for modality in modalities:
                slice = selected_index_df.loc[selected_index_df['modality'] == modality]
                # now we can apply the measure frequency function on this subset
                results.append(measure_frequency(slice))

    results.append(no_change_needed)
    final = pd.concat(results)

    return final


def add_street_length(counts_data, street_segments=None):
    """add the total street length that a datapoint is applicable to, to the counts dataset

    Args:
        counts_data: dataframe with all counts, modalities must already be mapped!
        street segments: street segments dataframe, including lengts of cut segments on a column named 'street_segment_length'
            This is optional, if not provided it will be loaded here.

    Returns:
        counts: counts_data with an added column total_street_length
    Raises:
        -
    """

    # column names
    identifiers = ['data_source', 'data_source_index']
    column_name_length = 'street_segment_length'
    filter_columns = identifiers.copy()
    filter_columns.append(column_name_length)

    if street_segments is None:
        raise('Street segments dataframe is Null object, streets lengths cannot be added.')

    # lengths aggregation
    lengths = street_segments[filter_columns]
    lengths_agg = lengths.groupby(identifiers).sum().reset_index()

    # type of merge must be left, inner join loses data!
    counts_data2 = pd.merge(counts_data, lengths_agg,
                            on=identifiers, how='left')
    columns_to_keep = list(counts_data.columns)
    columns_to_keep.append(column_name_length)
    counts_data2 = counts_data2[columns_to_keep]

    counts_data2.rename(
        columns={'street_segment_length': 'total_street_length'}, inplace=True)
    return counts_data2


def transform_data_types(counts):
    """Transformation according to type of data source. Frequency, total street length and mapped modalities should all be present!
    Explanation of calculation:

    Args:
        counts: dataframe with all counts, modalities must already be mapped. These columns should be present: measurement_type, frequency, speed, total_street_length.
            # these can be obtained by calling these functions:
                counts_data = add_modality_mapping_to_data(
                    counts_data, modality_mapping)
                counts_data = add_missing_speeds_to_data(
                    counts_data, modality_mapping)
                counts_data = add_frequency_of_measurement(counts_data)
                counts_data = add_street_length(counts_data, street_segments)

    Returns:
        counts: adapted columns based on measurement_type column
    Raises:
        -- error if measurement_column not present
    """

    ###############################
    # handling point measurements
    ###############################

    # filter on measurement_type = 'interval_measurement', 'point measurement'
    # naming of variable can be changed here - this is the measurement type we want to change
    intensity = 'point_measurement'

    # for the all modality, no such transformation is possible: throw warning if records of this kind exist
    if not counts[(counts['modality'] == 'all') & (counts['measurement_type'] == intensity)].empty:
        print(counts[(counts['modality'] == 'all') & (counts['measurement_type'] == intensity)])
        raise Warning(
            'There are point measurements without dedicated modality, this currently cannot be handled correctly.')

    # creating some auxiliary columns
    # in number of people per meter, aka density. Needs to be multiplied by street length to get real count
    counts['density'] = counts['count'] / \
        (counts['frequency'] * counts['speed'])
    # now the units is plain number of people at time t
    counts['new_count'] = counts['density'] * counts['total_street_length']
    counts['new_count'] = pd.to_numeric(counts['new_count'], errors='coerce')

    # Filter on measurement_type
    counts.loc[counts['measurement_type'] ==
               intensity, 'count'] = counts['new_count']

    # dropping auxiliary columns
    counts.drop(columns=['density', 'new_count'], inplace=True)

    ###############################
    # handling other types
    ###############################
    # TODO later if necessary

    return counts


def upsampling_data(df, f='60S'):
    """upsampling of data frame df, according to frequency f. Method: loop over data sources and indexes, within those controlled sliced dataframes, we have a timeseries
        where time interpolation can be done.

    Args:
        df: dataframe
        f: upsampling frequency, formatted as a string according to these rules. Example: '60S' gives minute interpolation
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    Returns:
        upsampled data frame
    Raises:
        --
    """

    def controlled_upsample(slice_cell, f):
        # print('DEBUGGING', slice_cell.dtypes)
        # this dataframe contains a time series from a specific data source. The code rests on the reindixing command from the pandas library.
        gdf_trans_reindexed = slice_cell.reindex(pd.date_range(
            start=slice_cell.index.min(), end=slice_cell.index.max(), freq=f))
        # gdf_trans_reindexed.index = pd.to_datetime(gdf_trans_reindexed.index)
        gdf_trans_reindexed.index.name = 'timestamp'
        # linear interpolation between available timesteps
        gdf_trans_reindexed.interpolate(method='time', inplace=True, )
        gdf_trans_reindexed["modality"].fillna(method='ffill', inplace=True)
        # gdf_trans_reindexed["measurement_type"].fillna(method ='ffill', inplace = True)
        gdf_trans_reindexed["data_source"].fillna(method='ffill', inplace=True)
        gdf_trans_reindexed["data_source_index"].fillna(
            method='ffill', inplace=True)
        gdf_slice = gdf_trans_reindexed.reset_index(level=['timestamp'])

        return gdf_slice

    print('Upsampling data...')
    # print(df.head(5))
    print('Datasources to be upsampled: ', df['data_source'].unique())
    # df.drop(df.columns.difference(['count', 'data_source', 'data_source_index', 'modality', 'timestamp', 'locationrange', 'speed', 'measurement_type']), 1, inplace=True)
    # a clean dataframe will be handed down

    df.dropna(subset=['data_source'], inplace=True)
    slices = [df]
    for data_src in df['data_source'].unique():
        print('Upsampling ' + data_src + '...')
        slice_frame_temp = df[(df['data_source'] == data_src)]
        for data_src_idx in slice_frame_temp['data_source_index'].unique():
            slice_cell_temp = slice_frame_temp[(
                slice_frame_temp['data_source_index'] == data_src_idx)]
            for cell in slice_cell_temp['modality'].unique():
                slice_cell = slice_cell_temp[(
                    slice_cell_temp['modality'] == cell)]
                if slice_cell.empty:
                    print('Warning empty dataframe')
                else:
                    slice_cell.set_index('timestamp', inplace=True)
                    slice_cell.sort_values(
                        by='timestamp', axis='index', ascending=True)
                    test = slice_cell[slice_cell.index.duplicated()]
                    if (not(test.empty)):
                        # TODO: This warning appears several times -? Still to be investigated what is happening
                        print('***Warning*** Duplicated index for ' +
                              data_src + ' - ' + cell)
                        slice_cell = slice_cell.reset_index().drop_duplicates(
                            subset=['timestamp'], keep='first').set_index('timestamp')

                    gdf_slice = controlled_upsample(slice_cell, f)
                    slices.append(gdf_slice)

    gdf_upsampled_data = pd.concat(slices, sort=True)
    gdf_upsampled_data.drop_duplicates(
        subset=['data_source', 'data_source_index', 'modality', 'timestamp'], keep='first', inplace=True)
    gdf_upsampled_data.sort_values(
        by=['data_source_index', 'timestamp'], inplace=True)
    # print(gdf_upsampled_data.loc[gdf_upsampled_data['data_source'] == 'citymesh'].head())
    print('Done upsampling!')
    return gdf_upsampled_data


def load_and_preprocess_counts_data(counts_data_path, street_segments, modality_mapping, upsampling_frequency='60S', create_new_pickle_files=False, data_source_cells=None):
    """full data preprocessing pipeline

    Args:
        counts_data_path: path to the counts data file
        street_segments: dataframe containing the street network
        modality_mapping: dictionary of modality mapping
        create_new_pickle_files: if new pickles have to be created

    Returns:
        loaded pandas dataframe in memory
        Might dump pickle in temp folder along the way
    Raises:
        --
    """
    # try read pickle file
    pickle_filename = 'preprocessed_data.p'
    pickle_path = os.path.join(temp_dir, pickle_filename)

    if os.path.isfile(pickle_path) and not create_new_pickle_files:
        print('    Found data pickle file!')
        with open(pickle_path, 'rb') as handle:
            counts_data = pickle.loads(handle.read())
    else:
        print('No pickle file found, loading data and applying modality transformations...')

        counts_data = load_counts_data(counts_data_path)
        counts_data = add_modality_mapping_to_data(
            counts_data, modality_mapping)

        if data_source_cells is not None:
            counts_data = counts_data.merge(
                data_source_cells, on=['data_source', 'data_source_index', 'modality'])
        
        counts_data = add_missing_speeds_to_data(counts_data, modality_mapping)
        counts_data = add_frequency_of_measurement(counts_data)
        counts_data = add_street_length(counts_data, street_segments)
        counts_data = transform_data_types(counts_data)
        # maybe do this in the tranform data types ? or make interpolate missing counts able to handle it
        counts_data = counts_data[counts_data.columns.difference(
            ['measurement_type', 'speed'])]

        # print('after transform measurement types: ', counts_data.tail())

        counts_data = interpolate_missing_counts(counts_data)
        # counts_data = add_modality_mapping_to_data(counts_data, modality_mapping)
        counts_data = upsampling_data(counts_data, upsampling_frequency)

        if create_new_pickle_files:
            # pickle dump
            print('Writing pickle files...')
            with open(pickle_path, 'wb') as handle:
                pickle.dump(counts_data, handle)
            print('Done writing pickle files.')

    return counts_data


def load_and_preprocess_data_and_grid():
    # to define!
    pass


def filter_counts(counts, street_segments):
    # only where there is also telco counts data
    # time filter
    # geofilter ?

    # filter out data points of datasource cells that don't intersect any street segments
    columns = ['data_source', 'data_source_index']
    tmp = street_segments[columns].drop_duplicates()
    filtered_data = counts.merge(tmp, left_on=columns, right_on=columns)
    return filtered_data


def get_input_counts_for_data_source_cells(counts, data_source_cells):
    selection = counts.merge(data_source_cells, on=[
                             'data_source', 'data_source_index', 'modality'])
    return selection[['data_source', 'data_source_index', 'modality', 'timestamp', 'count']]


# for testing purposes
if __name__ == "__main__":
    modality_mapping_filename = 'modality_mapping.json'
    input_modality_path = os.path.join(THIS_DIR, modality_mapping_filename)
    modality_mapping = get_modality_objects(input_modality_path)

    input_data_path = os.path.join(THIS_DIR, os.pardir, os.pardir,
                                   'data/managed_data_files/mobiele_stad/ready_for_main/')  # normally
    # for testing measurement type
    input_data_path = os.path.join(
        THIS_DIR, os.pardir, os.pardir, 'data/transformation_measurement_type/')
    input_data_file = 'all_data_measurement_type.csv'
    input_data_full_path = os.path.join(input_data_path, input_data_file)

    street_segments_file = 'street_segments.csv'
    street_segments_full_path = os.path.join(
        input_data_path, street_segments_file)
    street_segments = load_street_segments(street_segments_full_path)

    modality_mapping = get_modality_objects(input_modality_path)
    data_adapted = load_and_preprocess_counts_data(
        input_data_full_path, street_segments, modality_mapping, create_new_pickle_files=True)
    data_filtered = filter_counts(data_adapted, street_segments)
    print(data_adapted.shape)
    print(data_filtered.shape)
    print(data_filtered.dtypes)
