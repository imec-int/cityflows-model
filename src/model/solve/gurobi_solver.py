# IMPORTANT
#
# This file is not supported anymore
# In case you want to implement a Gurobi solver version, definitely don't use this as is
# but you can use this for inspiration

import os

import pandas as pd
from gurobipy import GRB, LinExpr, Model, quicksum
from src.utils.bcolors import bcolors
from src.utils.timer import timer_func

from .solver import Solution, Solver


class Gurobi_Solver(Solver):
    # public methods
    def __init__(self, use_previous_solution, modality_mixing=False, silent_output=True, debugging=False):
        super().__init__(use_previous_solution, modality_mixing, silent_output, debugging)

        self.model = Model(self.solver_name)
        if (self.silent_output):
            self.model.setParam("OutputFlag", 0)

        # this holds the dynamic constraints that have been added at the previous timestep
        # this allows for easy cleanup
        self.dynamic_constraints = []

    @timer_func
    def prepare(self):
        super().prepare()
        self.set_variables()
        self.set_static_constraints()
        self.compute_static_lin_expr()
        self.set_cost_function_quadratic_term()

    @timer_func
    def update(self, timestamp):
        super().update(timestamp)
        self.set_cost_function()
        self.update_constraints()

    @timer_func
    def update_constraints(self):
        self.clean_dynamic_constraints()

        # TODO: what about global modal split constraints ?
        self.set_counts_constraints()
        if self.use_previous_solution:
            self.set_density_update_constraints()

    @timer_func
    def solve_iteration(self):
        self.model.optimize()

        if self.debugging:
            self.write_debugging_files()

        if self.model.status != 2:
            # TODO: consider retrying with simplex, and if it still fails, then raise error
            print(
                f"{bcolors.FAIL}Gurobi model failed with status: {self.model.status}{bcolors.ENDC}")
            raise Exception('Model could not be solved')

        index_columns = ['street_object_id', 'modality']
        solution = self.densities[index_columns].copy()
        solution['solution'] = self.densities['variable'].apply(
            lambda v: v.getAttr("x"))

        return Solution(solution.set_index(index_columns), self.model.getAttr("ObjVal"))

    def write_debugging_files(self):
        basename = super().debugging_file_basename
        self.model.write(f'{basename}.lp')

    # private methods

    #################
    # set variables #
    #################
    def set_variables(self):
        self.create_density_variables()
        self.create_alpha_variables()
        if (self.use_previous_solution):
            self.create_delta_variables()

    # we create a density variable for each (street, modality) combination
    def create_density_variables(self):
        model = self.model
        street_segments = self.street_segments
        modalities = self.modalities

        modalities_series = pd.Series(modalities, name="modality")
        street_ids = street_segments[['street_object_id']].copy()
        street_ids.drop_duplicates(inplace=True, ignore_index=True)

        variables = street_ids.merge(modalities_series, how='cross')

        # nowwe can use the apply function, but unclear if parallellisation is possible in gurobi python interface
        # street segments are the unique to model things
        def create_variable(row):
            name_variable = f'density_{row.street_object_id}_{row.modality}'
            return model.addVar(vtype=GRB.CONTINUOUS, name=name_variable)

        variables['variable'] = variables.apply(create_variable, axis=1)
        variables['upper_bound'] = variables.apply(
            lambda row: self.modality_mapping[row.modality]['max_density'], axis=1)
        self.densities = variables

    # we create an alpha slack variable for each (data_source, data_source_index) combination
    def create_alpha_variables(self):
        model = self.model
        street_segments = self.street_segments

        alphas = street_segments[['data_source',
                                  'data_source_index', 'is_edge']].copy()
        # missing: filtering out the not-edge streets
        alphas = alphas.loc[~alphas['is_edge']]
        alphas = alphas[['data_source', 'data_source_index']]
        alphas.drop_duplicates(inplace=True, ignore_index=True)

        def create_variable(row):
            name_variable = f'alpha_{row.data_source}_{row.data_source_index}'
            return model.addVar(vtype=GRB.CONTINUOUS, name=name_variable)

        alphas['variable'] = alphas.apply(create_variable, axis=1)
        self.alphas = alphas

    # we create 1 or 2 density delta variables for each street when using modality_mixing
    # we create 1 or 2 density delta variables for each (street, modality) combination otherwise
    # with the exception of the bg_density modality, which don't require delta variables
    # creating 1 or 2 variables depends on how many intersections a street touches
    # deadend streets only get 1 variable, regular streets get 2
    def create_delta_variables(self):
        model = self.model
        intersections = self.intersections[[
            'intersection_id', 'street_object_id', 'intersection_type', 'end_type']].copy()
        intersections = intersections[intersections['intersection_type'] == 'physical']

        # don't create density delta variables for the 'bg_density'
        modalities = [
            modality for modality in self.modalities if modality != 'bg_density']
        modalities_series = pd.Series(modalities, name="modality")

        # if using modality mixing, create density delta variables for each street, regardless of modalities
        # otherwise, create 1 for each (street, modality) combination
        variables = intersections if self.modality_mixing else intersections.merge(
            modalities_series, how='cross')

        def create_variable(row):
            name_variable = f'density_delta_{row.street_object_id}'
            if not self.modality_mixing:
                name_variable += f'_{row.modality}'
            name_variable += f'_{row.end_type}'

            return model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name_variable)

        variables['variable'] = variables.apply(create_variable, axis=1)
        self.density_deltas = variables

    #####################
    # set cost function #
    #####################
    def set_cost_function_quadratic_term(self):
        # compute the squared street length for each street in each data source cell
        street_cell_length = self.street_cell_length
        street_cell_squared_length = street_cell_length ** 2

        # we need to sum over potential multiple data source cells
        coefficients = street_cell_squared_length.groupby(
            'street_object_id').sum()

        # retrieve all possible products of 2 variables of the same street
        products = self.densities.merge(self.densities, on='street_object_id', suffixes=(
            '_1', '_2')).set_index('street_object_id')

        # bring it all together
        products['coefficient'] = coefficients
        self.cost_function_quadratic_term = quicksum(
            products['coefficient'] * products['variable_1'] * products['variable_2'])

    @property
    def cost_function_linear_term(self):
        # compute the average density for each data source cell
        street_cell_length = self.street_cell_length
        counts = self.get_counts_for_timestamp().set_index(
            ['data_source', 'data_source_index'])['count']
        average_density = counts / \
            street_cell_length.groupby(
                ['data_source', 'data_source_index']).sum()

        # compute the coefficient of the linear term for each (street, data source cell) combination
        street_info = street_cell_length \
            .reset_index() \
            .merge(pd.Series(average_density, name='average_density'), left_on=['data_source', 'data_source_index'], right_index=True)
        street_info['coefficient'] = -2 * \
            street_info['street_cell_length'] ** 2 * \
            street_info['average_density']

        # we need to sum over potential multiple data source cells for a given street
        coefficients = street_info[['street_object_id', 'coefficient']].groupby(
            'street_object_id').sum()['coefficient']

        # link the variables
        densities = self.densities.copy().set_index('street_object_id')
        densities['coefficient'] = coefficients
        return LinExpr(densities['coefficient'], densities['variable'])

    @timer_func
    def set_cost_function(self):
        """creates the cost function for the model. The cost function is based on a steep penalty for the alpha variables, and a squared cost for the differences from average density in a cell
        The cost function is up for improvement. The diff_from_average is forcing the end solution to mimick the orginal geometrical shapes
        Attributes used:
            model: a (Gurobi) model
            alphas: alpha slack variables
        Returns:
            --
        Raises:
            --
        """
        alpha_cost = 10000

        # the difference from average density part
        quadratic_term = self.cost_function_quadratic_term
        linear_term = self.cost_function_linear_term

        self.model.setObjective(
            alpha_cost * quicksum(self.alphas['variable']) + quadratic_term + linear_term, GRB.MINIMIZE)

    ##########################
    # set static constraints #
    ##########################
    def set_static_constraints(self):
        self.set_max_density_constraints()
        if self.use_previous_solution:
            self.set_intersections_continuity_constraints()

    def set_max_density_constraints(self):
        for row in self.densities.itertuples():
            constraint_name = f'upper_bound_constraint_{row.street_object_id}_{row.modality}'
            self.model.addLConstr(
                row.variable <= row.upper_bound, name=constraint_name)

    # equation (13)
    def set_intersections_continuity_constraints(self):
        street_segments = self.street_segments
        street_lengths = street_segments[['street_object_id', 'street_object_length']] \
            .drop_duplicates(['street_object_id']) \
            .set_index('street_object_id')

        deltas = self.density_deltas.merge(
            street_lengths, left_on='street_object_id', right_index=True)

        groupby_columns = ['intersection_id'] if self.modality_mixing else [
            'intersection_id', 'modality']
        lin_expr_terms = deltas.groupby(groupby_columns).aggregate(
            {'street_object_length': list, 'variable': list})
        lin_expr_terms.rename(columns={
                              'street_object_length': 'coefficients', 'variable': 'variables'}, inplace=True)
        lin_expr_terms.reset_index(inplace=True)

        for row in lin_expr_terms.itertuples():
            constraint_name = f'intersection_continuity_constraint_{row.intersection_id}'
            if not self.modality_mixing:
                constraint_name += f'_{row.modality}'

            lhs = LinExpr(row.coefficients, row.variables)
            rhs = 0
            self.model.addLConstr(lhs == rhs, name=constraint_name)

    #################################
    # set static linear expressions #
    #################################
    def compute_static_lin_expr(self):
        # TODO: what about global modal split constraints ?
        self.compute_counts_constraints_lhs()
        if self.use_previous_solution:
            self.compute_density_update_constraints_lhs()

    # compute the left hand side linear expressions of equations (6) and (7)
    # since the right hand side expressions are counts data dependent, they cannot be computed statically
    def compute_counts_constraints_lhs(self):
        ss = self.street_segments[[
            'data_source', 'data_source_index', 'street_object_id', 'street_segment_length']]
        s = ss.groupby(['data_source', 'data_source_index', 'street_object_id']) \
            .sum() \
            .rename(columns={'street_segment_length': 'intersection_length'}) \
            .reset_index()
        s = s.merge(pd.Series(self.modalities, name="modality"), how='cross')

        index = ['street_object_id', 'modality']
        s.set_index(index, inplace=True)
        density_variables = self.densities.set_index(index)
        s['density_variable'] = density_variables['variable']
        s.reset_index(inplace=True)

        lin_expr_terms_single_modality = s.groupby(['data_source', 'data_source_index', 'modality']).aggregate({
            'intersection_length': list, 'density_variable': list})
        lin_expr_terms_all_modalities = s.groupby(['data_source', 'data_source_index']).aggregate(
            {'intersection_length': list, 'density_variable': list})
        lin_expr_terms_single_modality.reset_index(inplace=True)
        lin_expr_terms_all_modalities.reset_index(inplace=True)
        lin_expr_terms_all_modalities['modality'] = 'all'
        lin_expr_terms = pd.concat(
            [lin_expr_terms_single_modality, lin_expr_terms_all_modalities], ignore_index=True)
        lin_expr_terms.rename({'intersection_length': 'coefficients',
                              'density_variable': 'variables'}, axis=1, inplace=True)
        lin_expr_terms.set_index(
            ['data_source', 'data_source_index', 'modality'], inplace=True)

        def create_linear_expression(row):
            return LinExpr(row.coefficients, row.variables)

        self.counts_constraints_lhs = lin_expr_terms.apply(
            create_linear_expression, axis=1)

    def compute_density_update_constraints_lhs(self):
        densities = self.densities.copy()
        deltas = self.density_deltas.copy()

        if self.modality_mixing:
            lhs = densities.groupby(
                'street_object_id').aggregate({'variable': list})
            lhs.rename(columns={'variable': 'densities'}, inplace=True)
            lhs['deltas'] = deltas.groupby(
                'street_object_id').aggregate({'variable': list})

            # TODO: how to deal with the segments that do not intersect any other segment ?
            # those lead to a NaN deltas. 3 possibilities:
            # - fillna([]) -> constant densities
            # - dropna() -> unconstrained
            # - raise an error
            problems = lhs[lhs['deltas'].isna()]
            if problems.shape[0] > 0:
                print(problems)
                raise Exception(
                    'Could not link all deltas for density update constraints')

            def create_lin_expr(row):
                coefficients = [1] * len(row.densities) + \
                    [-1] * len(row.deltas)
                variables = row.densities + row.deltas
                return LinExpr(coefficients, variables)

            lhs['lhs'] = lhs.apply(create_lin_expr, axis=1)

        else:
            index_columns = ['street_object_id', 'modality']
            lhs = densities.set_index(index_columns)
            lhs.rename(columns={'variable': 'density'}, inplace=True)
            lhs['deltas'] = deltas.groupby(
                index_columns).aggregate({'variable': list})
            lhs.reset_index(inplace=True)

            # TODO: same comment as above
            problems = lhs[(lhs['modality'] != 'bg_density')
                           & (lhs['deltas'].isna())]
            if problems.shape[0] > 0:
                print(problems)
                raise Exception(
                    'Could not link all deltas for density update constraints')

            def create_lhs(row):
                if row.modality == 'bg_density':
                    return LinExpr(row.density)
                else:
                    variables = [row.density] + row.deltas
                    coefficients = [1] + [-1] * len(row.deltas)
                    return LinExpr(coefficients, variables)

            lhs['lhs'] = lhs.apply(create_lhs, axis=1)
            lhs.set_index(index_columns, inplace=True)

        if lhs['lhs'].isna().any():
            print(lhs[lhs['lhs'].isna()])
            raise Exception(
                'Some terms computed by density_update_constraints_lhs are problematic')

        self.density_update_constraints_lhs = lhs

    ##############################
    # update dynamic constraints #
    ##############################
    def set_counts_constraints(self):
        counts = self.get_counts_for_timestamp()
        counts = counts[['data_source', 'data_source_index',
                         'modality', 'count']].copy()

        # retrieve the associated alphas/slack variable
        alpha_index = ['data_source', 'data_source_index']
        counts.set_index(alpha_index, inplace=True)
        alphas = self.alphas.set_index(alpha_index)
        counts['alphas'] = alphas['variable']
        counts.reset_index(inplace=True)

        # retrieve the associated left hand side linear expression
        lhs_index = ['data_source', 'data_source_index', 'modality']
        counts.set_index(lhs_index, inplace=True)
        counts['lhs'] = self.counts_constraints_lhs
        counts.reset_index(inplace=True)
        if counts['lhs'].isna().any():
            raise Exception(
                "Some constraints left hand side terms couldn't be retrieved")

        counts_constraints = []
        for row in counts.itertuples():
            constraint_name_prefix = f'counts_constraint_{row.data_source}_{row.data_source_index}_{row.modality}'
            lhs = row.lhs
            rhs_1 = LinExpr((1 + row.alphas) * row.count)
            rhs_2 = LinExpr((1 - row.alphas) * row.count)
            counts_constraints.append(self.model.addLConstr(
                lhs <= rhs_1, name=constraint_name_prefix + '_1'))
            counts_constraints.append(self.model.addLConstr(
                lhs >= rhs_2, name=constraint_name_prefix + '_2'))

        self.dynamic_constraints.extend(counts_constraints)

    def set_density_update_constraints(self):
        constraints_expressions = self.density_update_constraints_lhs.copy()

        if self.modality_mixing:
            constraints_expressions['rhs'] = self.previous_solution.groupby(
                'street_object_id').aggregate({'solution': sum})
        else:
            constraints_expressions['rhs'] = self.previous_solution['solution']

        constraints_expressions.reset_index(inplace=True)

        def create_constraint(row):
            constraint_name = f'density_update_constraint_{row.street_object_id}'
            if not self.modality_mixing:
                constraint_name += f'_{row.modality}'
            return self.model.addLConstr(row.lhs == row.rhs, name=constraint_name)

        density_update_constraints = [create_constraint(
            row) for row in constraints_expressions.itertuples()]
        self.dynamic_constraints.extend(density_update_constraints)

        # TODO: global influx outflux is determined by net telco difference

    #############################
    # clean dynamic constraints #
    #############################
    def clean_dynamic_constraints(self):
        self.model.remove(self.dynamic_constraints)
        self.dynamic_constraints = []

    #####################
    # verification code #
    #####################
    def verify(self):
        # let's do a check and compare the measures counts, the solved counts, and the slack variables
        # we will combine everything in the checks dataframe
        # let's start by retrieving the measured counts, the easiest part...
        data = self.get_counts_for_timestamp()
        if data[data['modality'] != 'all'].shape[0] > 0:
            print("Warning: only veryfing constraints on 'all' modality")
            data = data[data['modality'] == 'all']
        data.set_index(['data_source', 'data_source_index'], inplace=True)

        checks = pd.DataFrame(index=data.index)
        checks['data_count'] = data['count']

        # now let's add the counts obtained by Gurobi for each (data_source, data_source_index) combination
        self.densities['solved_value'] = self.densities['variable'].apply(
            lambda v: v.getAttr("x"))

        # this is ok because at this timestamp we know that we have counts for 'all' modality for cropland, and nothing for bikes
        street_all_density = self.densities.groupby(
            ['street_segment_id']).aggregate({'solved_value': sum})
        street_all_density.rename(
            columns={'solved_value': 'density'}, inplace=True)

        street_segments = self.street_segments[self.street_segments['data_source'] == 'proximus'].copy(
        )
        street_segments.set_index('street_segment_id', inplace=True)
        street_segments['density'] = street_all_density
        street_segments['count'] = street_segments.apply(
            lambda row: row.density * row.street_segment_length, axis=1)
        datasource_cell_counts = street_segments.groupby(
            ['data_source', 'data_source_index']).aggregate({'count': sum})
        datasource_cell_counts.rename(
            columns={'count': 'solved_count'}, inplace=True)
        checks['solved_count'] = datasource_cell_counts['solved_count']

        # finally, let's add the value of the slack variable associated to (data_source, data_source_index)
        self.alphas['solved_value'] = self.alphas['variable'].apply(
            lambda v: v.getAttr("x"))
        alphas = self.alphas.set_index(['data_source', 'data_source_index'])
        checks['alphas'] = alphas['solved_value']

        # let's compute if the lower and upper bounds are satisfied
        checks['equation_6_satisfied'] = checks.apply(
            lambda row: row.solved_count <= row.data_count + row.alphas, axis=1)
        checks['equation_7_satisfied'] = checks.apply(
            lambda row: row.solved_count >= row.data_count - row.alphas, axis=1)

        # probably just some rounding errors...
        # besides that, it looks like the constraints are quite well satisfied
        # hence probably well implemented 💪
        print(checks[(checks['equation_6_satisfied'] == False)
              | (checks['equation_7_satisfied'] == False)])

        # if you want to inspect in QGIS
        # output = street_segments[['street_segment_geometry', 'density', 'count']]
        # output.to_csv('/Users/quent/Desktop/output.csv')


if __name__ == "__main__":
    import os

    from src.model.data_mappings import (
        filter_counts, load_and_preprocess_counts_data)
    from src.model.modality import get_modality_objects
    from src.model.street_grid.intersections import load_intersections
    from src.model.street_grid.street_segments import \
        load_street_segments

    # load and transform data
    data_folder = 'data/managed_data_files/test_set'

    if data_folder == 'data/managed_data_files/':
        filename_streets = 'AAA/input/street_segments.csv'
        filename_intersections = 'AAA/input/intersections.csv'
        filename_data = 'AAA/input/counts/cropland_2020_01_01.csv'

        timestamp = '2020-01-01 12:00:00+00:00'
    elif data_folder == 'data/managed_data_files/test_set':
        filename_streets = 'street_segments.csv'
        filename_intersections = 'intersections.csv'
        filename_data = 'all_data_small_cropland_velo.csv'

        timestamp = '2020-05-11 12:00:00+00:00'
    filename_modality = 'modality_mapping.json'

    THIS_DIR = os.path.dirname(__file__)
    data_files_path = os.path.join(
        THIS_DIR, os.pardir, os.pardir, os.pardir, data_folder)
    street_path = os.path.join(data_files_path, filename_streets)
    intersections_path = os.path.join(data_files_path, filename_intersections)
    data_path = os.path.join(data_files_path, filename_data)
    modalities_path = os.path.join(THIS_DIR, os.pardir, filename_modality)

    street_segments = load_street_segments(street_path)
    intersections = load_intersections(intersections_path)
    modality_mapping = get_modality_objects(modalities_path)
    counts = load_and_preprocess_counts_data(data_path, modality_mapping)
    filtered_counts = filter_counts(counts, street_segments)

    # solving process starts here
    solver = Gurobi_Solver(False)
    solver.load_data(filtered_counts, street_segments,
                     intersections, modality_mapping)
    solver.prepare()
    solver.update(timestamp)
    solution = solver.solve_iteration().x

    print('Gurobi solution: ')
    print(solution.head())

    # solver = Gurobi_Solver(True, False)
    # solver.set_previous_solution(solution)
    # solver.load_data(filtered_counts, street_segments, intersections, modality_mapping)
    # solver.prepare()
    # solver.update(timestamp)
    # solution = solver.solve_iteration().x

    # solver = Gurobi_Solver(True, True)
    # solver.set_previous_solution(solution)
    # solver.load_data(filtered_counts, street_segments, intersections, modality_mapping)
    # solver.prepare()
    # solver.update(timestamp)
    # solution = solver.solve_iteration().x
