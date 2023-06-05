import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
This script will determine the data_source_cells with the largest and the smallest alpha-values in the ouptu of the model.
It will also produce a histogram of all the alpha-values sorted by their respective data_source
"""

# tolerance on negative values
TOL = 1e-9

if __name__ == '__main__':
    # alphas_path = sys.argv[1]
    alphas_path = 'data/managed_data_files/mobiele_stad/2020_analysis/sample_of_the_final_computation/1711/alphas.csv'
    alphas = pd.read_csv(alphas_path)

    # sns.relplot(data=alphas[alphas['data_source']== 'cropland'], kind='line', x='timestamp', y='solution',
    #             col='data_source_index')
    # plt.show()

    # detect negative alphas
    # (those should not happen in theory but because of the nature of the OSQP solver
    # some slightly negative values can occur)
    problematic_alphas = alphas[alphas['solution'] < -TOL]
    if len(problematic_alphas) > 0:
        n_issues = len(problematic_alphas)
        print(
            f'Warning: There are negative alphas ({n_issues}), smaller than accepted tolerance of {TOL}\n')
        print(problematic_alphas['data_source'].unique())

    # descriptive statistics for each datasource
    print('Descriptive statistics')
    print(alphas.groupby(['data_source'])['solution'].describe(), '\n')
    alphas['abs'] = alphas['solution'].apply(abs)

    # plot histograms of alphas values for each datasource
    alphas.hist(column='abs', by='data_source')
    plt.show()

    # display what datasource cells have larger alphas
    head_count = 3
    print(f'Top {head_count} cells with largest alpha values for each datasource')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    print(
        alphas
        .groupby(['data_source', 'data_source_index'])['abs']
        .max()
        .to_frame()
        .sort_values(['data_source', 'abs'], ascending=[True, False])
        .groupby('data_source')
        .head(head_count)
    )

    print(f'Top {head_count} cells with smallest alpha values for each datasource')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    print(
        alphas
        .groupby(['data_source', 'data_source_index'])['abs']
        .max()
        .to_frame()
        .sort_values(['data_source', 'abs'], ascending=[True, True])
        .groupby('data_source')
        .head(head_count)
    )
