import os
from .data_mappings import load_and_preprocess_counts_data, filter_counts
from .modality import get_modality_objects
from .count_constraints import get_bounds
from .solve.osqp_solver import OSQP_Solver
from .solve.smart_solver import SmartSolver
from .street_grid.intersections import load_intersections
from .street_grid.street_segments import get_streets_metadata, load_street_segments
from .street_grid.ml_weights import load_ml_weights


def execute_model(input_folder, output_folder, modality_mapping_path, bounds_path, modality_mixing_iteration_step, upsampling_frequency):
    """Execute the fusion model on some local files using OSQP solver.

    Parameters:
    input_folder (str): path to the folder containing the input data files
    output_folder (str): path to the folder where the output files will be written
    modality_mapping_path (str): path to the modality mapping file
    modality_mixing_iteration_step (int): number of iterations to perform before applying the modality mixing variant
    upsampling_frequency (str): frequency of the upsampling (e.g. '1h', '5min', '10sec')    

    """

    # run inputs
    street_segments_path = f'{input_folder}/street_segments.csv'
    intersections_path = f'{input_folder}/intersections.csv'
    counts_path = f'{input_folder}/all_data.csv'
    ml_weights_path = f'{input_folder}/ml_weights.csv'

    # run outputs
    streets_path = f'{output_folder}/streets.csv'
    output_path = f'{output_folder}/densities.csv'
    alphas_path = f'{output_folder}/alphas.csv'

    # load input data
    street_segments = load_street_segments(street_segments_path)
    intersections = load_intersections(intersections_path)
    modality_mapping = get_modality_objects(modality_mapping_path)
    ml_weights = load_ml_weights(ml_weights_path, modality_mapping)
    bounds = get_bounds(bounds_path)

    counts = load_and_preprocess_counts_data(
        counts_path,
        street_segments,
        modality_mapping,
        upsampling_frequency=upsampling_frequency,
    )
    counts = filter_counts(counts, street_segments)

    os.makedirs(output_folder, exist_ok=True)

    # store streets information
    get_streets_metadata(street_segments).to_csv(streets_path, index=False)

    # compute and store the densities
    solver = SmartSolver(
        solver_class=OSQP_Solver,
        modality_mixing_iteration_step=modality_mixing_iteration_step,
        output_path=output_path,
        alphas_path=alphas_path
    )
    solver.load_data(counts, street_segments, intersections,
                     modality_mapping, ml_weights, bounds)
    solver.prepare()
    solver.solve()


if __name__ == '__main__':
    execute_model(
        input_folder='data/managed_data_files/mobiele_stad/test_set_LC3_b/input',
        output_folder='data/managed_data_files/mobiele_stad/test_set_LC3_b/output',
        modality_mapping_path='src/model/modality_mapping.json',
        bounds_path='src/model/count_constraints_bounds.json',
        modality_mixing_iteration_step=4,
        upsampling_frequency='5min',
    )
