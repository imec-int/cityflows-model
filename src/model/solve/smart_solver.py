from datetime import datetime
from uuid import uuid4
from src.model.street_grid.ml_weights import load_ml_weights
from src.utils.files import ensure_file_not_exists
from src.utils.bcolors import bcolors
from src.utils.timer import PausableTimer, Timer, UseTimer

from .osqp_solver import OSQP_Solver
from .solver import Solver


class SmartSolver():
    def __init__(self, solver_class, modality_mixing_iteration_step, output_path=None, alphas_path=None, silent_output=True, debugging=False):
        if not issubclass(solver_class, Solver):
            raise Exception(
                'Provided solver_class is not a subclass of Solver')

        if debugging:
            debugging_session_id = str(uuid4())
            debugging_directory = os.path.join('debug', debugging_session_id)
            os.makedirs(debugging_directory, exist_ok=True)
            print(f"Debugging session output: {debugging_directory}")
        else:
            debugging_directory=None

        self.solver_class_name = solver_class.__qualname__
        self.first_timestamp_solver = solver_class(
            use_previous_solution=False, silent_output=silent_output, debugging=debugging, debugging_directory=debugging_directory)
        self.mixed_modality_solver = solver_class(
            use_previous_solution=True, modality_mixing=True, silent_output=silent_output, debugging=debugging, debugging_directory=debugging_directory)
        self.unmixed_modality_solver = solver_class(
            use_previous_solution=True, modality_mixing=False, silent_output=silent_output, debugging=debugging, debugging_directory=debugging_directory)

        self.modality_mixing_iteration_step = modality_mixing_iteration_step
        self.output_path = output_path
        self.alphas_path = alphas_path

        self.prepare_timer = Timer('Prepare')
        self.update_timer = PausableTimer('Update')
        self.solve_timer = PausableTimer('Solver')
        self.io_timer = PausableTimer('IO')

    def load_data(self, counts, street_segments, intersections, modality_mapping, ml_weights, bounds):
        unique_timestamps = [ts for ts in counts['timestamp'].unique()]
        unique_timestamps.sort()
        self.timestamps = unique_timestamps

        self.first_timestamp_solver.load_data(
            counts, street_segments, intersections, modality_mapping, ml_weights, bounds)
        self.mixed_modality_solver.load_data(
            counts, street_segments, intersections, modality_mapping, ml_weights, bounds)
        self.unmixed_modality_solver.load_data(
            counts, street_segments, intersections, modality_mapping, ml_weights, bounds)

    def prepare(self):
        with UseTimer(self.prepare_timer):
            self.first_timestamp_solver.prepare()
            self.mixed_modality_solver.prepare()
            self.unmixed_modality_solver.prepare()

    def solve(self, n_steps=None):
        fails = []
        ensure_file_not_exists(self.output_path)
        ensure_file_not_exists(self.alphas_path)

        if n_steps is None:
            n_steps = len(self.timestamps)

        previous_solution = None
        for i in range(n_steps):
            if i == 0:
                solver = self.first_timestamp_solver
            elif i % self.modality_mixing_iteration_step == 0:
                solver = self.mixed_modality_solver
            else:
                solver = self.unmixed_modality_solver

            timestamp = self.timestamps[i]
            print(
                f'\nSolving {timestamp} with {self.solver_class_name}_{solver.solver_name}')

            solver.set_previous_solution(previous_solution)

            with UseTimer(self.update_timer):
                solver.update(timestamp)

            try:
                with UseTimer(self.solve_timer):
                    iteration_solution = solver.solve_iteration()
            except Exception as e:
                fails.append(timestamp)
                if solver == self.first_timestamp_solver:
                    # in this case there is no point in retrying, it would lead to the same error
                    raise e
                else:
                    print(
                        f"{bcolors.WARNING}Warning: couldn't solve model with appropriate solver, using the relaxed model instead{bcolors.ENDC}")
                    solver = self.first_timestamp_solver

                    with UseTimer(self.update_timer):
                        solver.update(timestamp)

                    with UseTimer(self.solve_timer):
                        iteration_solution = solver.solve_iteration()

            previous_solution = iteration_solution.x
            alphas = iteration_solution.alphas

            with UseTimer(self.io_timer):
                write_header = True if i == 0 else False
                mode = 'w' if i == 0 else 'a'
                if self.output_path is not None:
                    data = previous_solution.reset_index()
                    data['timestamp'] = timestamp
                    output = data.pivot(
                        index=['street_object_id', 'timestamp'], columns='modality', values='solution')
                    output.to_csv(self.output_path, mode=mode,
                                  header=write_header)

                if self.alphas_path is not None:
                    alphas['timestamp'] = timestamp
                    alphas.to_csv(self.alphas_path, mode=mode,
                                  header=write_header)

        if len(fails) > 0:
            print('\n\nThe model used a relaxed model on the following timestamps:\n',fails)

    def time_stats(self):
        return {
            'prepare': self.prepare_timer.duration,
            'update': self.update_timer.duration,
            'solve': self.solve_timer.duration,
            'io': self.io_timer.duration
        }


if __name__ == "__main__":
    import os

    from src.model.data_mappings import (
        filter_counts, load_and_preprocess_counts_data)
    from src.model.modality import get_modality_objects
    from src.model.street_grid.intersections import load_intersections
    from src.model.street_grid.street_segments import \
        load_street_segments

    # load and transform data
    filename_streets = 'street_segments_IE.csv'
    filename_intersections = 'intersections.csv'
    filename_data = 'all_data.csv'
    filename_modality = 'modality_mapping.json'

    THIS_DIR = os.path.dirname(__file__)
    data_files_path = os.path.join(
        THIS_DIR, os.pardir, os.pardir, os.pardir, 'data/managed_data_files/mobiele_stad/learning_cycle_3/input')
    street_path = os.path.join(data_files_path, filename_streets)
    intersections_path = os.path.join(data_files_path, filename_intersections)
    data_path = os.path.join(data_files_path, filename_data)
    modalities_path = os.path.join(THIS_DIR, os.pardir, filename_modality)
    weights_path = os.path.join(
        THIS_DIR, os.pardir, os.pardir, os.pardir, 'data/managed_data_files/mobiele_stad/learning_cycle_3/input/ml_weights.csv')

    street_segments = load_street_segments(street_path)
    intersections = load_intersections(intersections_path)
    modality_mapping = get_modality_objects(modalities_path)
    weights = load_ml_weights(weights_path, modality_mapping)

    # define run parameters
    frequency = 5  # minutes
    duration = 1  # hours
    output_path = os.path.join(
        'data/output/learning_cycle_3', f'duration_{duration}_hours_frequency_{frequency}_mins.csv')

    frequency_str = f'{frequency}min'
    n_steps = int(duration * 60 / frequency + 1)
    modality_mixing_iteration_step = 20 / frequency

    counts = load_and_preprocess_counts_data(
        data_path, street_segments, modality_mapping, upsampling_frequency=frequency_str)
    filtered_counts = filter_counts(counts, street_segments)

    # filter to start at a later time
    # filtered_counts = filtered_counts[filtered_counts['timestamp'] >= '2020-01-01 01:00:00']

    # solving process starts here
    now = datetime.now()
    solver = SmartSolver(OSQP_Solver, modality_mixing_iteration_step,
                         output_path=output_path, debugging=False)
    solver.load_data(filtered_counts, street_segments,
                     intersections, modality_mapping, weights)
    solver.prepare()
    solver.solve(n_steps)
