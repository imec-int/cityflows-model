import os
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import pandas as pd
from src.model.modality import get_modality_list
from src.model.street_grid.ml_weights import get_default_weights

Solution = namedtuple('Solution', ['x', 'alphas', 'obj_val'])

# abstract class


class Solver(metaclass=ABCMeta):
    # public methods
    def __init__(self, use_previous_solution, modality_mixing=False, silent_output=True, debugging=False, debugging_directory=None):
        self.use_previous_solution = use_previous_solution
        self.modality_mixing = modality_mixing

        solver_name = 'CityFlows'
        if not use_previous_solution:
            solver_name += '_first_timestamp'
        elif modality_mixing:
            solver_name += '_mixed_modalities'
        else:
            solver_name += '_unmixed_modalities'
        self.solver_name = solver_name

        self.silent_output = silent_output
        self.debugging = debugging
        self.debugging_directory = debugging_directory

    # concrete methods: loading data is common regardless of the core solver used
    def load_data(self, counts, street_segments, intersections, modality_mapping, weights, bounds):
        self.counts = counts 
        #should check whether or not there are only correct modalities..
        # All the telraam data is entered separatly 
        self.street_segments = street_segments
        self.intersections = intersections
        self.modality_mapping = modality_mapping
        self.modalities = get_modality_list(modality_mapping)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = get_default_weights(
                street_segments=street_segments, modalities_list=self.modalities)
        self.bounds = bounds

    @abstractmethod
    def prepare(self):
        self.set_street_cell_length()
        self.set_datasource_cell_modalities()

    def set_previous_solution(self, previous_solution):
        self.previous_solution = previous_solution

    @abstractmethod
    def update(self, timestamp):
        self.timestamp = timestamp
        self.timestamp_counts = self.get_counts_for_timestamp()

    @abstractmethod
    def solve_iteration(self):
        pass

    # private methods
    @property
    def debugging_file_basename(self):
        filename = f'{self.timestamp}_{self.solver_name}'
        basename = os.path.join(self.debugging_directory, filename)
        return basename

    def get_data_sources_info(self):
        data_sources_dataframe = self.counts[[
            'data_source', 'data_source_index', 'modality']].drop_duplicates()
        return data_sources_dataframe

    def get_counts_for_timestamp(self):
        return self.counts.loc[self.counts['timestamp'] == self.timestamp]

    def set_street_cell_length(self):
        ss = self.street_segments.filter(
            ['data_source', 'data_source_index', 'street_object_id', 'street_segment_length'], axis='columns')
        street_cell_length = ss.groupby(
            ['data_source', 'data_source_index', 'street_object_id']).sum()['street_segment_length']
        self.street_cell_length = pd.Series(
            street_cell_length, name='street_cell_length')

    def set_datasource_cell_modalities(self):
        '''
        This function will create and store (as a solver instance property) a dataframe that holds all modalities
        for all datasource cells
        '''
        # first, we isolate the datasource cells for which measurements apply to all modalities together
        datasource_cells_handling_all_modalities = self.counts.loc[self.counts['modality'] == 'all'].drop_duplicates(
            ['data_source', 'data_source_index'])[['data_source', 'data_source_index']]

        # second, we isolate the datasource cells for which measurements apply to a single modality
        datasource_cells_handling_single_modality = self.counts.loc[self.counts['modality'] != 'all'].drop_duplicates(
            ['data_source', 'data_source_index'])[['data_source', 'data_source_index', 'modality']]

        modalities = pd.DataFrame(self.modalities, columns=['modality'])

        # time to put everything together
        self.datasource_cell_modalities = pd.concat([
            datasource_cells_handling_single_modality,
            datasource_cells_handling_all_modalities.merge(
                modalities, how='cross')
        ])
