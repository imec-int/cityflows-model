# import dask.dataframe as dd
import pandas as pd


def load_results(results_path, use_dask=False):
    module = pd#dd if use_dask else pd
    results = module.read_csv(
        results_path, dtype={'street_object_id': int})
    results['timestamp'] = module.to_datetime(results['timestamp'])
    return results


def unpivot_results(results):
    return results.melt(id_vars=['street_object_id', 'timestamp'], var_name='modality', value_name='density')


def get_computed_counts_for_data_source_cells(results, data_source_cells, street_segments,modality):
    # let's compute a dataframe holding the street intersection lengths for every (street, data source cell) combination
    # its columns are:
    # - data_source
    # - data_source_index
    # - modality
    # - street_object_id
    # - intersection_length

    ss = street_segments[['data_source', 'data_source_index',
                          'street_object_id', 'street_segment_length']]
    ss = ss.merge(data_source_cells, on=['data_source', 'data_source_index'])
    intersection_lengths = ss.groupby(['data_source', 'data_source_index', 'modality', 'street_object_id']) \
        .sum() \
        .rename(columns={'street_segment_length': 'intersection_length'}) \
        .reset_index()

    extended_results = results.merge(
        intersection_lengths, on='street_object_id', suffixes=['_output', '_input'])
    extended_results = extended_results[(extended_results['modality_input'] == 'all') | (
        extended_results['modality_output'].apply(lambda x: x.split('_')[0]) == extended_results['modality_input'].apply(lambda x: x.split('_')[0]))]
    extended_results['count'] = extended_results['density'] * \
        extended_results['intersection_length']
    counts = extended_results.groupby(
        ['data_source', 'data_source_index', 'modality_input', 'timestamp'])['count'].sum().reset_index().rename(columns={"modality_input": "modality"})
    return counts
