import argparse
import os

from src.model.data_mappings import (
    filter_counts, load_and_preprocess_counts_data)
from src.model.solve.gurobi_solver import Gurobi_Solver
from src.model.solve.osqp_solver import OSQP_Solver
from src.model.solve.smart_solver import SmartSolver
from src.model.modality import get_modality_objects
from src.model.street_grid.intersections import load_intersections
from src.model.street_grid.street_segments import \
    get_streets_metadata, load_street_segments
from src.utils.manage_data_files import download_blobs, upload_data
from src.utils.timer import PausableTimer, Timer, UseTimer

THIS_DIR = os.path.dirname(__file__)

if __name__ == "__main__":

    # command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', choices=['gurobi', 'osqp'], required=True)
    parser.add_argument('--blobs_dir', required=True)
    parser.add_argument('--counts_prefixes', nargs='+', required=True)
    parser.add_argument('--overwrite_blobs', action='store_true')
    parser.add_argument('--compute_on_local_files', action='store_true')
    parser.add_argument('--remove_output_files', action='store_true')
    parser.add_argument('--densities_folder_suffix', default='')

    args = parser.parse_args()
    solver_id = args.solver
    blobs_dir = args.blobs_dir
    counts_prefixes = args.counts_prefixes
    overwrite_blobs = args.overwrite_blobs
    compute_on_local_files = args.compute_on_local_files
    remove_output_files = args.remove_output_files
    densities_folder_suffix = args.densities_folder_suffix

    download_timer = Timer('Download')
    data_preparation_timer = Timer('Data preparation')
    counts_preparation_timer = PausableTimer('Counts preparation')
    solver_timer = PausableTimer('Solver')
    upload_timer = PausableTimer('Upload')
    solver_durations = {
        'prepare': 0,
        'update': 0,
        'solve': 0,
        'io': 0
    }

    # download all files following the given blob storage patterns given in the command,
    # unless --compute_on_local_files is set, then the files on disk are used
    with UseTimer(download_timer):
        if not compute_on_local_files:
            download_blobs_prefixes = [
                f'{blobs_dir}/input/street_segments.csv',
                f'{blobs_dir}/input/intersections.csv'
            ] + [f'{blobs_dir}/input/counts/{prefix}' for prefix in counts_prefixes]

            download_blobs(download_blobs_prefixes, overwrite=overwrite_blobs)

    with UseTimer(data_preparation_timer):
        # load and transform data
        run_data_files_dir = os.path.join(
            THIS_DIR, os.pardir, os.pardir, 'data', 'managed_data_files', blobs_dir)
        input_data_files_dir = os.path.join(run_data_files_dir, 'input')
        output_data_files_dir = os.path.join(run_data_files_dir, 'output')
        counts_files_dir = os.path.join(input_data_files_dir, 'counts')

        os.makedirs(output_data_files_dir, exist_ok=True)

        street_segments_path = os.path.join(
            input_data_files_dir, 'street_segments.csv')
        intersections_path = os.path.join(
            input_data_files_dir, 'intersections.csv')
        street_segments = load_street_segments(street_segments_path)
        intersections = load_intersections(intersections_path)

        # compute and upload the street geometries file
        blob_path = f'{blobs_dir}/output/streets.csv'
        file_path = os.path.join(output_data_files_dir, 'streets.csv')
        get_streets_metadata(street_segments).to_csv(file_path, index=False)
        upload_data(file_path, blob_path)

        filename_modality = 'modality_mapping.json'
        modalities_path = os.path.join(THIS_DIR, filename_modality)
        modality_mapping = get_modality_objects(modalities_path)

        # this still means modality mixing is applied every 20 minutes
        upsampling_frequency = '5min'
        modality_mixing_iteration_step = 4

        files_to_solve = counts_prefixes if compute_on_local_files else sorted(
            os.listdir(counts_files_dir))

    for file in files_to_solve:
        with UseTimer(counts_preparation_timer):
            counts_path = os.path.join(counts_files_dir, file)
            counts = load_and_preprocess_counts_data(
                counts_path, street_segments, modality_mapping, upsampling_frequency=upsampling_frequency)
            filtered_counts = filter_counts(counts, street_segments)

            # keep in mind that the batch files contain the counts for midnight of the day after for proper upsampling
            # since the upsampling is now done, let's remove those so that we don't compute for this timestep as it will
            # be taken care of in the next batch
            filtered_counts['day'] = filtered_counts['timestamp'].dt.day
            reference_day = filtered_counts.iloc[0]['day']
            filtered_counts = filtered_counts[filtered_counts['day']
                                              == reference_day]

        with UseTimer(solver_timer):
            if solver_id == 'gurobi':
                solver_class = Gurobi_Solver
            elif solver_id == 'osqp':
                solver_class = OSQP_Solver
            else:
                raise Exception('Unknown solver')

            densities_foldername = f'densities_{densities_folder_suffix}'.rstrip(
                '_')
            alphas_foldername = 'alphas'

            output_path = os.path.join(
                output_data_files_dir, densities_foldername, file)
            alphas_path = os.path.join(
                output_data_files_dir, alphas_foldername, file)
            solver = SmartSolver(
                solver_class, modality_mixing_iteration_step, output_path=output_path, alphas_path=alphas_path)
            solver.load_data(filtered_counts, street_segments,
                             intersections, modality_mapping)
            solver.prepare()
            solver.solve()

            timestep_durations = solver.time_stats()
            solver_durations['prepare'] += timestep_durations['prepare']
            solver_durations['update'] += timestep_durations['update']
            solver_durations['solve'] += timestep_durations['solve']
            solver_durations['io'] += timestep_durations['io']

        with UseTimer(upload_timer):
            densities_blob_path = f'{blobs_dir}/output/{densities_foldername}/{file}'
            alphas_blob_path = f'{blobs_dir}/output/{alphas_foldername}/{file}'
            upload_data(output_path, densities_blob_path,
                        overwrite=overwrite_blobs)
            upload_data(alphas_path, alphas_blob_path,
                        overwrite=overwrite_blobs)
            if remove_output_files:
                os.remove(output_path)
                os.remove(alphas_path)

    print("requests_cpu, limits_cpu, download_time, data_preparation_time, counts_preparation_time, solver_prepare_time, solver_update_time, solver_solve_time, solver_io_time, upload_time")
    print(
        f", , {download_timer.duration}, {data_preparation_timer.duration}, {counts_preparation_timer.duration}, {solver_durations['prepare']}, {solver_durations['update']}, {solver_durations['solve']}, {solver_durations['io']}, {upload_timer.duration}")
