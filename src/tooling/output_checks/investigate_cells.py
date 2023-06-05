import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

from src.model.data_mappings import \
    get_input_counts_for_data_source_cells, load_and_preprocess_counts_data
from src.model.modality import get_modality_objects
from src.model.results import get_computed_counts_for_data_source_cells, load_results, unpivot_results
from src.model.street_grid.street_segments import load_street_segments

"""
This script will display the input and  the output of one or more data_source_cells specified for the specified timewindow.
The necesary files are on lines 43-51.
"""

modality_mapping_path = 'src/model/modality_mapping.json'
modality_mapping = get_modality_objects(modality_mapping_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', required=True)
    parser.add_argument('--data_source_cells', nargs='+', required=True)
    parser.add_argument('--modality', required=True,
                        choices=["all"] + list(modality_mapping.keys()))
    parser.add_argument('--min_time', required=False,default=None)
    parser.add_argument('--max_time', required=False,default=None)

    args = parser.parse_args()
    data_source = args.data_source
    data_source_cells_list = args.data_source_cells
    modality = args.modality

    # create the data source cells dataframe (including modality)
    data_source_cells_obj = [{
        "data_source": data_source,
        "data_source_index": c,
        "modality": modality
    } for c in data_source_cells_list]
    data_source_cells = pd.DataFrame(data_source_cells_obj)

    data_path = 'data'
    main_path = os.path.join(
        data_path, 'managed_data_files/mobiele_stad/learning_cycle_3')
    input_path = os.path.join(main_path, 'input')
    output_path = os.path.join(main_path, 'output')

    street_segments_path = os.path.join(input_path, 'street_segments.csv')
    counts_path = os.path.join(input_path, 'all_data.csv')
    results_path = os.path.join(output_path, 'densities.csv')

    street_segments = load_street_segments(street_segments_path)
    counts_data = load_and_preprocess_counts_data(
        counts_path, street_segments, modality_mapping, data_source_cells=data_source_cells)
    
    if args.min_time is not None:
        min_ts = args.min_time
    else:
        min_ts = counts_data['timestamp'].min()
    if args.max_time is not None:
        max_ts = args.max_time
    else:
        max_ts = counts_data['timestamp'].max()

    # compute the computed counts in the provided cells
    print("Filtering input counts for provided data source cells...", end=" ")
    input_counts = get_input_counts_for_data_source_cells(
        counts_data, data_source_cells)
    input_counts = input_counts[input_counts['timestamp'].between(min_ts, max_ts)]
    print("Done")

    print("Loading and unpivoting results...", end=" ")
    results_wide = load_results(results_path, use_dask=True)
    results_wide = results_wide[results_wide['timestamp'].between(min_ts, max_ts)]
    results = unpivot_results(results_wide)#.compute()
    print("Done")

    print("Compute counts based on model output for provided data source cells...", end=" ")
    computed_counts = get_computed_counts_for_data_source_cells(
        results, data_source_cells, street_segments, modality)
    print("Done")

    # plot computed against input
    input_counts['method'] = 'input'
    computed_counts['method'] = 'computed'
    counts = pd.concat([input_counts, computed_counts])
    sns.relplot(data=counts, kind='line', x='timestamp', y='count',
                hue='method', row='data_source_index')
    plt.ylim(bottom=0)
    plt.show()

    difference_df = computed_counts.copy()
    input_counts2 = input_counts.rename(columns={'count':'input_count'})
    difference_df=difference_df.merge(input_counts2,how='left',on=['data_source','data_source_index','timestamp'])


    difference_df['diff'] = difference_df['input_count']-difference_df['count']
    sns.relplot(data=difference_df, kind='line', x='timestamp', y='diff',
                row='data_source_index')

    # plt.plot(difference_df['timestamp'].to_list(),difference_df['diff'].to_list())
    plt.show()
    