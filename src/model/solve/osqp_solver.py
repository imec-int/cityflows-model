import numpy as np
import osqp
import pandas as pd
from scipy.sparse import csc_matrix, lil_matrix, vstack
from src.utils.timer import timer_func

from .solver import Solution, Solver


class OSQP_Solver(Solver):
    # inherits from abstract class Solver

    # public methods
    def __init__(self, use_previous_solution, modality_mixing=False, silent_output=True, debugging=False, debugging_directory=None):
        """debugging set to True will slow down the calculation!"""
        super().__init__(use_previous_solution, modality_mixing,
                         silent_output, debugging, debugging_directory)

        self.model = osqp.OSQP()
        """ Variables and constraints need to be implemented in advance for the get functions
        IMPORTANT USER DISCRETION:
            add the new type of variables or constraints here as well!
        """
        # variables
        self.densities = pd.DataFrame([])
        self.alphas = pd.DataFrame([])
        self.density_deltas = pd.DataFrame([])

        # constraints
        self.df_max_density_constraints = pd.DataFrame([])
        self.df_max_alpha_constraints = pd.DataFrame([])
        self.df_counts_constraints = pd.DataFrame([])
        self.df_intersection_continuity_constraints = pd.DataFrame([])
        self.df_density_update_constraints = pd.DataFrame([])

    @timer_func
    def prepare(self):
        super().prepare()
        self.set_variables()
        self.set_static_constraints()
        self.compute_static_lin_expr()
        self.set_cost_function_linear_term_helper_dataframe()

    @timer_func
    def update(self, timestamp):
        super().update(timestamp)
        self.update_constraints()
        self.update_cost_function()

    @timer_func
    def update_constraints(self):
        self.clean_dynamic_constraints()

        self.set_count_constraints()
        if self.use_previous_solution:
            self.set_density_update_constraints()

    @timer_func
    def update_cost_function(self):
        self.set_cost_function_mathematical_objects()
        self.set_cost_function_quadratic_term()
        self.set_cost_function_linear_term()
        self.set_cost_function_alpha_term()

    @timer_func
    def solve_iteration(self):
        """
        IMPORTANT USER DISCRETION:
            add the constraints (A,l,u) in the same order they are numbered by the osqp_row!"""
        # the boundary conditions on the densities
        self.A = self.A_max_density_constraints
        self.l = self.l_max_density_constraints
        self.u = self.u_max_density_constraints
        # the boundary conditions on the alpha variables
        self.A = vstack([self.A, self.A_max_alpha_constraints], format='csc')
        self.l = np.append(self.l, self.l_max_alpha_constraints, axis=0)
        self.u = np.append(self.u, self.u_max_alpha_constraints, axis=0)

        if self.use_previous_solution:
            # the intersection continuity and the density update conditions
            self.A = vstack([self.A, self.A_intersection_continuity_constraints,
                            self.A_density_update_constraints], format='csc')
            # for the intersection_continuity l and u are equal because this is an equality condition
            self.l = np.append(np.append(
                self.l, self.RHS_intersection_continuity_constraints, axis=0), self.RHS_density_update_constraints)
            self.u = np.append(np.append(
                self.u, self.RHS_intersection_continuity_constraints, axis=0), self.RHS_density_update_constraints)

        # the counts conditions
        self.A = vstack([self.A,  self.A_counts_constraints], format='csc')
        self.l = np.append(self.l, self.l_counts_constraints)
        self.u = np.append(self.u, self.u_counts_constraints)

        # P and q are already set in set_cost_function_* functions

        # setting up and solving the iteration
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l,
                         u=self.u, verbose=not(self.silent_output))
        self.res = self.model.solve()

        if self.debugging:
            self.write_debugging_files()

        if self.res.info.status_val != 1:
            # more about this status_values on: https://osqp.org/docs/interfaces/status_values.html#status-values
            raise Exception('Failed to solve this iteration')

        index_columns = ['street_object_id', 'modality', 'osqp_index']
        solution = self.densities[index_columns].copy()
        solution['solution'] = self.res.x[self.densities.osqp_index]
        solution.drop(columns='osqp_index', inplace=True)

        alphas = self.alphas[['data_source', 'data_source_index']].copy()
        alphas['solution'] = self.res.x[self.alphas.osqp_index]

        return Solution(solution.set_index(['street_object_id', 'modality']), alphas, self.res.info.obj_val)

    def write_debugging_files(self):
        variables = pd.concat([self.alphas, self.densities, self.density_deltas])[
            ['variable', 'osqp_index']].set_index('osqp_index')

        def format_coefficient(coefficient, first_term=False):
            sign = '+' if coefficient > 0 else '-'
            sign = sign if not(first_term) or coefficient < 0 else ''
            coefficient_str = '' if abs(
                coefficient) == 1 else f'{abs(coefficient)}'
            return f'{sign} {coefficient_str}'.strip()

        basename = super().debugging_file_basename
        with open(f'{basename}.txt', 'w') as f:
            f.write('Minimize\n')

            # quadratic terms
            i_indices, j_indices = self.P.nonzero()
            for k in range(len(i_indices)):
                i = i_indices[k]
                j = j_indices[k]

                coefficient = 0.5 * self.P[i, j]

                if i == j:
                    quadratic_variable = f'{variables.loc[i,"variable"]}^2'
                else:
                    quadratic_variable = f'{variables.loc[i,"variable"]} * {variables.loc[j,"variable"]}'

                term = f'\t{format_coefficient(coefficient, first_term=k==0)} {quadratic_variable}\n'

                f.write(term)

            # linear terms
            ni = self.q.shape[0]
            for i in range(ni):
                coefficient = self.q[i]
                if coefficient == 0:
                    continue

                variable_name = variables.loc[i, 'variable']
                term = f'\t{format_coefficient(coefficient)} {variable_name}\n'
                f.write(term)

            f.write('such that\n')

            i_indices, j_indices = self.A.nonzero()
            for i in range(self.A.shape[0]):
                # find the indices in i_indices & j_indices relating to this row/constraint
                # following expression is inspired by the Note from https://numpy.org/doc/stable/reference/generated/numpy.where.html?highlight=where#numpy.where
                constraint_indices = np.asarray(i_indices == i).nonzero()[0]
                variable_osqp_indexes = j_indices[constraint_indices]

                lower_bound = self.l[i]
                upper_bound = self.u[i]
                linear_expression = ''
                for j in range(len(variable_osqp_indexes)):
                    index = variable_osqp_indexes[j]
                    variable_name = variables.loc[index, 'variable']
                    coefficient = self.A[i, index]

                    linear_expression += f'{format_coefficient(coefficient, first_term=j==0)} {variable_name} '

                f.write(
                    f'\t{lower_bound} <= {linear_expression.strip()} <= {upper_bound}\n')

    # private methods

    def get_osqp_row(self):
        """ returns the number of constraints already implemented
        IMPORTANT USER DISCRETION:
            when adding a new type of constraints also add the here!
        """
        N_max_dens = len(self.df_max_density_constraints.index)
        N_max_alpha = len(self.df_max_alpha_constraints.index)
        N_counts = len(self.df_counts_constraints.index)
        N_intersections = len(
            self.df_intersection_continuity_constraints.index)
        N_update = len(self.df_density_update_constraints.index)
        number_of_constraints = N_max_dens + N_max_alpha + \
            N_counts + N_intersections + N_update
        return number_of_constraints

    def get_number_of_osqp_index(self):
        """ returns the number of variables already implemented
        IMPORTANT USER DISCRETION:
            when adding a new type of variable also add the here!
        """
        N_segments = len(self.densities.index)
        N_alphas = len(self.alphas.index)
        N_deltas = len(self.density_deltas.index)
        number_of_variables = N_segments + N_alphas + N_deltas
        return number_of_variables

    #################
    # set variables #
    #################
    def set_variables(self):
        """initialising all the different types of variables
        IMPORTANT USER DISCRETION:
            Don't add a variable without subsequently calling the get_osqp_index function and setting the correct 'ospq_index' atrribute correctly
        """
        # Setting the variables (order doesn't matter)
        self.densities = self.create_density_variables()
        self.alphas = self.create_alpha_variables()
        if self.use_previous_solution:
            self.density_deltas = self.create_delta_variables()

    def get_osqp_index(self, number_needed):
        """Returns the amount of osqp_indices that's needed, starting from the ones that are already implemented"""
        number_of_variables = self.get_number_of_osqp_index()
        # sets the unique index
        return range(number_of_variables, number_of_variables+number_needed)

    def create_density_variables(self):
        """Creates a variable for every street for every modality"""
        street_segments = self.street_segments
        modalities = self.modalities

        modalities_series = pd.Series(modalities, name="modality")
        street_objects_ids = street_segments[['street_object_id']].copy()
        street_objects_ids.drop_duplicates(inplace=True, ignore_index=True)
        df_variables = street_objects_ids.merge(modalities_series, how='cross')

        df_variables['variable'] = pd.Series("dens_" + str(tuple.modality) + "_street_" + str(
            tuple.street_object_id) for tuple in df_variables.itertuples())
        df_variables['upper_bound'] = pd.Series(
            self.modality_mapping[tuple.modality]['max_density'] for tuple in df_variables.itertuples())
        df_variables['osqp_index'] = self.get_osqp_index(
            len(df_variables.index))  # sets the unique index
        return df_variables

    def create_alpha_variables(self):
        """Creates a slack variable for every data source"""
        alpha = self.street_segments[['data_source', 'data_source_index']].drop_duplicates(
            ignore_index=True).copy()

        alpha['variable'] = pd.Series("alpha_" + str(tuple.data_source) + "_" + str(
            tuple.data_source_index) for tuple in alpha.itertuples())
        alpha['osqp_index'] = self.get_osqp_index(
            len(alpha.index))  # sets the unique index

        return alpha

    def create_delta_variables(self):
        """Creates a variable for every street segment on every intersection"""
        # don't create density delta variables for the 'bg_density'
        modalities = [
            modality for modality in self.modalities if modality != 'bg_density']
        modalities_series = pd.Series(modalities, name="modality")

        intersections = self.intersections[[
            'intersection_id', 'is_edge', 'street_object_id', 'intersection_type', 'end_type']].copy()
        intersections = intersections[intersections['intersection_type'] == 'physical']

        # if using modality mixing, create density delta variables for each street, regardless of modalities
        # otherwise, create 1 for each (street, modality) combination
        df_variables = intersections if self.modality_mixing else intersections.merge(
            modalities_series, how='cross')

        def create_variable(row):
            name_variable = f'density_delta_{row.street_object_id}'
            if not self.modality_mixing:
                name_variable += f'_{row.modality}'
            name_variable += f'_{row.end_type}'
            return name_variable

        df_variables['variable'] = df_variables.apply(create_variable, axis=1)
        df_variables['osqp_index'] = self.get_osqp_index(
            len(df_variables.index))  # sets the unique index
        return df_variables

    #####################
    # set cost function #
    #####################
    def set_cost_function_linear_term_helper_dataframe(self):
        # TODO: review carefully, very work in progress...
        tmp1 = self.street_cell_length.reset_index().merge(
            self.datasource_cell_modalities, on=['data_source', 'data_source_index'])

        # TODO: self.weights does not contain the modality bg_density yet... needs to be fixed
        tmp2 = tmp1.merge(self.weights, on=['street_object_id', 'modality'])
        tmp2['l_sdc * w_sm'] = tmp2['street_cell_length'] * tmp2['weight']
        w_dc = tmp2.groupby(['data_source', 'data_source_index']).aggregate(
            {'l_sdc * w_sm': sum}).rename(columns={'l_sdc * w_sm': 'w_dc'})

        tmp3 = tmp2.merge(w_dc, on=['data_source', 'data_source_index'])
        tmp3['w_smdc'] = -2 * tmp3['street_cell_length'] ** 2 * \
            tmp3['weight'] / tmp3['w_dc']
        self.cost_function_linear_term_helper_dataframe = tmp3

    def set_cost_function_mathematical_objects(self):
        n_variables = self.get_number_of_osqp_index()
        self.P = lil_matrix((n_variables, n_variables))
        self.q = np.zeros(n_variables)

    def set_cost_function_quadratic_term(self):
        # square l_{s,d,c}
        l_squared = (self.street_cell_length **
                     2).rename('coefficient').reset_index()

        # keep only datasource_cell_modalities of datasource cells that have a measurement at the current timestamp
        datasource_cells = self.timestamp_counts[[
            'data_source', 'data_source_index']].drop_duplicates()
        datasource_cell_modalities = self.datasource_cell_modalities.merge(
            datasource_cells, on=['data_source', 'data_source_index'])

        # append the modalities concerned by the measurements of datasource cell (d,c)
        tmp = pd.merge(l_squared, datasource_cell_modalities,
                       on=['data_source', 'data_source_index'])

        # coefficient for rho^2_{s,m} is the sum of l^2_{s,d,c} over all datasource cells that comprise street s and modality m
        coefficients = tmp.groupby(['street_object_id', 'modality']).aggregate(
            {'coefficient': sum}).reset_index()

        # join with the densities variables to retrieve the indices of the P matrix where those coefficients belong
        tmp2 = pd.merge(coefficients, self.densities, on=[
            'street_object_id', 'modality'])
        indices = tmp2['osqp_index']
        values = tmp2['coefficient']

        # factor 2 is there because OSQP uses the quadratic form: 0.5 * x^T P x
        self.P[indices, indices] = 2 * values
        self.P = self.P.tocsc()

    def set_cost_function_linear_term(self):
        # TODO: review carefully, very work in progress...
        tmp1 = self.cost_function_linear_term_helper_dataframe.merge(
            self.timestamp_counts, on=['data_source', 'data_source_index'])
        tmp1['coefficient'] = tmp1['w_smdc'] * tmp1['count']
        coefficients = tmp1.groupby(['street_object_id', 'modality_x']) \
            .aggregate({'coefficient': sum}) \
            .reset_index() \
            .rename(columns={'modality_x': 'modality'})
        tmp2 = coefficients.merge(
            self.densities, on=['street_object_id', 'modality'])
        indices = tmp2['osqp_index']
        values = tmp2['coefficient']
        self.q[indices] = values

    def set_cost_function_alpha_term(self):
        alpha_cost = 10000  # for comparison set to 10000
        indices = self.alphas['osqp_index']
        self.q[indices] = alpha_cost

    ##########################
    # set static constraints #
    ##########################

    def set_static_constraints(self):
        self.set_max_density_constraints()
        self.set_max_alpha_constraints()
        if self.use_previous_solution:
            self.set_intersections_continuity_constraints()

    def set_max_density_constraints(self):
        """ Function sets the bounds for the density variables:
                lower bound: 0
                upper bound: depending on the variable (it's modality)
        """
        variables = self.densities

        def set_max_density_constraints_helper(osqp_index, upper_bound, modality, street_object_id):
            dict_constraints['osqp_row'].append(
                self.get_osqp_row()+len(dict_constraints['osqp_row']))
            dict_constraints['type'].append('max_dens')
            dict_constraints['osqp_indices'].append(osqp_index)
            dict_constraints['u'].append(upper_bound)
            dict_constraints['constr_name'].append(
                'max_dens_'+str(modality)+'_'+str(street_object_id))

        if self.debugging:
            dict_constraints = {'constr_name': [], 'osqp_row': [
            ], 'type': [], 'u': [], 'osqp_indices': []}
            for tuple in variables.itertuples():
                set_max_density_constraints_helper(
                    tuple.osqp_index, tuple.upper_bound, tuple.modality, tuple.street_object_id)

            self.df_max_density_constraints = pd.DataFrame.from_dict(
                data=dict_constraints)
        # creating the matrices
        # setting the rhs (l,u)
        self.l_max_density_constraints = np.zeros((len(variables.index)))
        self.u_max_density_constraints = variables['upper_bound']
        # creating the A matrix
        # matrix has 1's as coefficients
        data = np.ones((len(variables.index)))
        # every row has 1 element different from zero
        A_row_indices = np.arange(len(variables.index))
        A_col_indices = variables['osqp_index']
        self.A_max_density_constraints = csc_matrix((data, (A_row_indices, A_col_indices)), shape=(
            len(variables.index), self.get_number_of_osqp_index()))

    def set_max_alpha_constraints(self):
        """ Function sets the bounds for the alpha variables, they need to be positive so:
                lower bound: 0
                upper bound: infinity
        """

        variables = self.alphas
        dict_constraints = {'constr_name': [],
                            'osqp_row': [], 'type': [], 'osqp_indices': []}

        def set_max_alpha_constraints_helper(osqp_index, variable):
            dict_constraints['osqp_row'].append(
                self.get_osqp_row()+len(dict_constraints['osqp_row']))
            dict_constraints['type'].append('max_alpha')
            dict_constraints['osqp_indices'].append(osqp_index)
            dict_constraints['constr_name'].append('max_'+str(variable))
        if self.debugging:
            for tuple in variables.itertuples():
                set_max_alpha_constraints_helper(
                    tuple.osqp_index, tuple.variable)

        df_alpha_boundary_constraints = pd.DataFrame.from_dict(
            data=dict_constraints)
        # creating the matrices
        self.l_max_alpha_constraints = np.zeros((len(variables.index)))
        self.u_max_alpha_constraints = len(variables.index)*[np.inf]
        # matrix has 1's as coefficients
        data = np.ones((len(variables['osqp_index'])))
        # every row has 1 element different from zero
        A_row_indices = np.arange(len(variables['osqp_index']))
        A_col_indices = variables['osqp_index'].to_list()
        self.A_max_alpha_constraints = csc_matrix((data, (A_row_indices, A_col_indices)), shape=(len(
            variables.index), self.get_number_of_osqp_index()))  # matrix with a 1 for the corresponding alpha variable.

        self.df_max_alpha_constraints = df_alpha_boundary_constraints

    def set_intersections_continuity_constraints(self):
        """ Function fixes the continuity on the intersections: net flow is zero, equation (13) of the paper:
                sum over all deltas of streets corresponding to an intersection = 0
                both lower and upper bound (RHS) are zero
        """
        street_segments = self.street_segments
        street_lengths = street_segments[['street_object_id', 'street_object_length']] \
            .drop_duplicates(['street_object_id']) \
            .set_index('street_object_id')

        deltas = self.density_deltas.merge(
            street_lengths, left_on='street_object_id', right_index=True)

        groupby_columns = ['intersection_id'] if self.modality_mixing else [
            'intersection_id', 'modality']
        lin_expr_terms = deltas.groupby(groupby_columns).aggregate(
            {'street_object_length': list, 'osqp_index': list})
        lin_expr_terms.rename(columns={
                              'street_object_length': 'coefficients', 'osqp_index': 'variables'}, inplace=True)
        lin_expr_terms.reset_index(inplace=True)

        dict_constraints = {'constr_name': [], 'osqp_row': [],
                            'type': [], 'coefficients': [], 'osqp_indices': []}
        # using lil_matrix type because of the incremental updates to the matrix
        A_lil = lil_matrix(
            (len(lin_expr_terms.index), self.get_number_of_osqp_index()))
        for i, tuple in enumerate(lin_expr_terms.itertuples()):
            # the length of the constr_name list functions as a counter
            row_indices = len(tuple.coefficients)*[i]
            A_lil[row_indices, tuple.variables] = tuple.coefficients
            if self.debugging:
                constr_name = f'intersection_continuity_constraint_{tuple.intersection_id}'
                if not self.modality_mixing:
                    constr_name += f'_{tuple.modality}'

                dict_constraints['type'].append('intersection_continuity')
                dict_constraints['constr_name'].append(constr_name)
                dict_constraints['osqp_indices'].append(tuple.variables)
                dict_constraints['coefficients'].append(tuple.coefficients)
                dict_constraints['osqp_row'].append(
                    self.get_osqp_row()+len(dict_constraints['osqp_row']))

        self.df_intersection_continuity_constraints = pd.DataFrame.from_dict(
            data=dict_constraints)
        self.A_intersection_continuity_constraints = A_lil.tocsc()
        self.RHS_intersection_continuity_constraints = np.zeros(
            len(lin_expr_terms.index))  # the righthand are all zero (net flow is zero)

    #################################
    # set static linear expressions #
    #################################
    def compute_static_lin_expr(self):
        self.compute_counts_constraints_lhs()
        if self.use_previous_solution:
            self.compute_density_update_constraints_lhs()

    def compute_counts_constraints_lhs(self):
        """ Function sets static part of the count constraints:
                creates matrix with lengths of the streets as coefficients.
                This creates a too many rows (!), there are some variables with wrong modalities added aswell.
        """
        ss = self.street_segments[[
            'street_object_id', 'street_segment_length', 'data_source', 'data_source_index']]
        s = ss.groupby(['data_source', 'data_source_index', 'street_object_id']) \
            .sum() \
            .rename(columns={'street_segment_length': 'intersection_length'}) \
            .reset_index()

        s = s.merge(pd.Series(self.modalities, name="modality"), how='cross')
        index = ['street_object_id', 'modality']
        s.set_index(index, inplace=True)
        density_variables = self.densities.set_index(index)
        s['density_variable'] = density_variables['osqp_index']
        s.reset_index(inplace=True)

        lin_expr_terms_single_modality = s.groupby(['data_source', 'data_source_index', 'modality']).aggregate({
            'intersection_length': list, 'density_variable': list})
        lin_expr_terms_all_modalities = s.groupby(['data_source', 'data_source_index']).aggregate(
            {'intersection_length': list, 'density_variable': list})
        lin_expr_terms_single_modality.reset_index(inplace=True)
        lin_expr_terms_all_modalities.reset_index(inplace=True)
        lin_expr_terms_all_modalities['modality'] = 'all'
        lin_expr_terms = pd.concat(
            [lin_expr_terms_single_modality, lin_expr_terms_all_modalities])
        lin_expr_terms.rename({'intersection_length': 'coefficients',
                              'density_variable': 'variables'}, axis=1, inplace=True)
        lin_expr_terms.set_index(
            ['data_source', 'data_source_index', 'modality'], inplace=True)

        self.counts_constraints_lhs = lin_expr_terms

        # setting static part of LHS counts constraints
        counts_constraints_lhs = self.counts_constraints_lhs.reset_index(
            inplace=False)
        # using lil_matrix type because of the incremental updates to the matrix
        # 2 times the number of constraints (for lower and upper)
        A_counts_constraints_lil = lil_matrix(
            (2*len(counts_constraints_lhs.index), self.get_number_of_osqp_index()))
        dict_counts = {'data_source': [], 'data_source_index': [],
                       'constr_name': [], 'type': [], 'modality': []}

        counter = 0
        for tuple in counts_constraints_lhs.itertuples():
            row_indices_lower = len(tuple.coefficients)*[counter]
            row_indices_upper = len(tuple.coefficients)*[counter+1]
            A_counts_constraints_lil[row_indices_lower,
                                     tuple.variables] = tuple.coefficients
            A_counts_constraints_lil[row_indices_upper,
                                     tuple.variables] = tuple.coefficients

            dict_counts['data_source'].extend(2*[tuple.data_source])
            dict_counts['data_source_index'].extend(
                2*[tuple.data_source_index])
            dict_counts['constr_name'].extend(
                ['counts_lower_'+tuple.data_source+'_'+tuple.data_source_index, 'counts_upper_'+tuple.data_source+'_'+tuple.data_source_index])
            dict_counts['type'].extend(['counts_lower', 'counts_upper'])
            dict_counts['modality'].extend(2*[tuple.modality])

            counter += 2

        self.df_counts_constraints_lhs = pd.DataFrame.from_dict(
            data=dict_counts)
        self.df_counts_constraints_lhs = self.df_counts_constraints_lhs[[
            'data_source', 'data_source_index', 'type', 'constr_name', 'modality']]

        self.A_counts_constraints_big = A_counts_constraints_lil.tocsc()

    def compute_density_update_constraints_lhs(self):
        """ Function sets static part of the density update constraints:
                creates matrix wich couples densities and the delta variables.
                uses equation (10)-(12) from the paper
        """
        densities = self.densities.copy()
        deltas = self.density_deltas.copy()

        dict_constraints = {'constr_name': [], 'type': [],
                            'coefficients': [], 'osqp_indices': []}  # ,'osqp_row':[]
        if self.modality_mixing:
            lhs = densities.groupby('street_object_id').aggregate(
                {'osqp_index': list})
            lhs.rename(columns={'osqp_index': 'densities'}, inplace=True)
            lhs['deltas'] = deltas.groupby(
                'street_object_id').aggregate({'osqp_index': list})
            lhs.reset_index(inplace=True)

            # TODO: how to deal with the segments that do not intersect any other segment ?
            # those lead to a NaN deltas. 3 possibilities:
            # - fillna([]) -> constant densities
            # - dropna() -> unconstrained
            # - raise an error
            problems = lhs[lhs['deltas'].isna()]
            if problems.shape[0] > 0:
                for tuple in problems.itertuples():
                    if any(self.street_segments[self.street_segments['street_object_id'] == tuple.street_object_id].is_edge):
                        lhs.drop(index=tuple.Index, inplace=True)
                    else:
                        lhs.at[tuple.Index, 'deltas'] = []

            remaining_problems = lhs[lhs['deltas'].isna()]
            if remaining_problems.shape[0] > 0:
                print(remaining_problems)
                raise Exception(
                    'Could not link all deltas for density update constraints')

            # using lil_matrix type because of the incremental updates to the matrix
            A_lil = lil_matrix(
                (len(lhs.index), self.get_number_of_osqp_index()))

            for i, tuple in enumerate(lhs.itertuples()):
                constraint_name = f'density_update_constraint'

                variables = tuple.densities + tuple.deltas
                row_indices = len(variables)*[i]
                coefficients = len(tuple.densities)*[1]+len(tuple.deltas)*[-1]
                A_lil[row_indices, variables] = coefficients

                dict_constraints['type'].append('density_update')
                dict_constraints['coefficients'].append(coefficients)
                dict_constraints['osqp_indices'].append(variables)
                dict_constraints['constr_name'].append(constraint_name)

        else:
            index_columns = ['street_object_id', 'modality']
            lhs = densities.set_index(index_columns)
            lhs.rename(columns={'osqp_index': 'density'}, inplace=True)
            lhs['deltas'] = deltas.groupby(
                index_columns).aggregate({'osqp_index': list})
            lhs.reset_index(inplace=True)

            # TODO: same comment as above
            # delta variables are not created for the modality 'bg_density', hence the filter
            problems = lhs[(lhs['modality'] != 'bg_density')
                           & (lhs['deltas'].isna())]
            if problems.shape[0] > 0:
                for tuple in problems.itertuples():
                    if any(self.street_segments[self.street_segments['street_object_id'] == tuple.street_object_id].is_edge):
                        lhs.drop(index=tuple.Index, inplace=True)
                    else:
                        lhs.at[tuple.Index, 'deltas'] = []

            remaining_problems = lhs[(lhs['modality'] != 'bg_density')
                                     & (lhs['deltas'].isna())]
            if remaining_problems.shape[0] > 0:
                print(remaining_problems)
                raise Exception(
                    'Could not link all deltas for density update constraints')

            # using lil_matrix type because of the incremental updates to the matrix
            A_lil = lil_matrix(
                (len(lhs.index), self.get_number_of_osqp_index()))

            for i, tuple in enumerate(lhs.itertuples()):
                constraint_name = f'density_update_constraint_{tuple.street_object_id}_{tuple.modality}'

                if tuple.modality == 'bg_density':
                    # The background is made to be fixed over timesteps, equation (12) in the paper
                    variables = [tuple.density]
                    coefficients = [1]
                else:
                    variables = [tuple.density] + tuple.deltas
                    coefficients = [1] + len(tuple.deltas) * [-1]

                row_indices = len(variables) * [i]
                A_lil[row_indices, variables] = coefficients

                dict_constraints['type'].append('density_update')
                dict_constraints['coefficients'].append(coefficients)
                dict_constraints['osqp_indices'].append(variables)
                dict_constraints['constr_name'].append(constraint_name)

            lhs.set_index(index_columns, inplace=True)

        self.A_density_update_constraints = A_lil.tocsc()

        self.df_density_update_constraints = pd.DataFrame.from_dict(
            data=dict_constraints)
        self.density_update_constraints_lhs = lhs

    ##############################
    # update dynamic constraints #
    ##############################

    def set_count_constraints(self):
        """ Function sets dynamic part of the count constraints:
                filters the static part on the actual counts
                sets RHSlengths of the streets as coefficients.
                and adds the alpha variables with the count as coefficient.

        """
        counts = self.get_counts_for_timestamp()
        if 'index' in counts:
            counts.rename({'index': 'data_source_index'}, axis=1, inplace=True)
            print('Had to change the column name index to data_source_index')
        counts = counts[['data_source', 'data_source_index',
                         'modality', 'count']].copy()
        bounds = self.bounds

        # retrieve the associated alpha/slack variable
        alpha_index = ['data_source', 'data_source_index']
        counts.set_index(alpha_index, inplace=True)
        df_counts = self.df_counts_constraints_lhs.copy()

        df_counts.set_index(alpha_index, inplace=True)
        alpha = self.alphas.set_index(alpha_index)
        counts['alpha'] = alpha['osqp_index']
        counts.reset_index(inplace=True)

        counts['type'] = counts['data_source'].apply(lambda x : bounds[x])
        counts = counts.explode('type')

        # retrieve the associated left hand side linear expression
        lhs_index = ['data_source', 'modality', 'data_source_index','type']
        df_counts['static_counts_index'] = np.arange(len(df_counts.index))
        # selecting the statically prepared constraints that have actual counts.
        df_constraints = counts.merge(df_counts, on=lhs_index)

        df_constraints.reset_index(inplace=True)
        df_constraints['osqp_row'] = np.arange(
            self.get_osqp_row(), self.get_osqp_row()+len(df_constraints.index))
        # giving each row of the filtered down A_counts_constraints an index
        df_constraints['dynamic_counts_index'] = np.arange(
            len(df_constraints.index))
        self.df_counts_constraints = df_constraints

        # statically prepared A matrix for the counts is too large, thus we select only the constraints (rows) for which there are actual counts.
        self.A_counts_constraints = self.A_counts_constraints_big[
            df_constraints['static_counts_index'], :]
        df_constraints.drop(columns=['static_counts_index'], inplace=True)

        # allocating the l and u vector
        self.l_counts_constraints = np.empty(
            self.A_counts_constraints.shape[0])
        self.u_counts_constraints = np.empty(
            self.A_counts_constraints.shape[0])

        # grouping the lower and upper constraints and setting their bounds and the corresponding count
        constr = df_constraints.groupby(['type']).aggregate(
            {'alpha': list, 'dynamic_counts_index': list, 'count': list})
        constr['l'] = [constr.loc['counts_lower']['count'], len(
            df_constraints[df_constraints['type'] == 'counts_upper'].index)*[-np.inf]]
        constr['u'] = [len(df_constraints[df_constraints['type'] == 'counts_lower'].index)
                       * [np.inf], constr.loc['counts_upper']['count']]
        constr.loc['counts_upper']['count'] = np.multiply(
            constr.loc['counts_upper']['count'], -1)

        # using lil_matrix type because of the incremental updates to the matrix
        A_lil = lil_matrix(self.A_counts_constraints.shape)
        for tuple in constr.itertuples():
            A_lil[tuple.dynamic_counts_index, tuple.alpha] = tuple.count
            self.l_counts_constraints[tuple.dynamic_counts_index] = tuple.l
            self.u_counts_constraints[tuple.dynamic_counts_index] = tuple.u

        self.A_counts_constraints += A_lil  # here we do need to append to the static part

        df_constraints.drop(columns=['dynamic_counts_index'], inplace=True)

    def set_density_update_constraints(self):
        """ Function set RHS for the density update constraints (equation (10)-(12) in the paper)
        """
        constraints_expressions = self.density_update_constraints_lhs.copy()

        # setting the rhs
        if self.modality_mixing:
            prev_solution = self.previous_solution.groupby(
                'street_object_id').aggregate({'solution': sum})
            prev_solution = prev_solution.rename(columns={'solution': 'rhs'})
            prev_solution = prev_solution.reset_index()
            constraints_expressions = constraints_expressions.merge(
                prev_solution)
        else:
            constraints_expressions['rhs'] = self.previous_solution['solution']

        constraints_expressions.reset_index(inplace=True)
        self.RHS_density_update_constraints = constraints_expressions['rhs']

        if self.debugging:
            for tuple in constraints_expressions.itertuples():
                constraint_name = f'density_update_constraint_{tuple.street_object_id}'
                if not self.modality_mixing:
                    constraint_name += f'_{tuple.modality}'
                self.df_density_update_constraints.loc[tuple.Index,
                                                       'constr_name'] = constraint_name

            # start with -len() because the dataframe is already counted.
            self.df_density_update_constraints['osqp_row'] = np.arange(self.get_osqp_row(
            )-len(self.df_density_update_constraints.index), self.get_osqp_row())

    #############################
    # clean dynamic constraints #
    #############################

    def clean_dynamic_constraints(self):
        self.df_counts_constraints = pd.DataFrame([])
        self.df_density_update_constraints = pd.DataFrame([])

    def get_osqp_index_of_density(self, street_object_id, modality):  # for debugging
        """returns the osqp index for a given modality and street"""
        df = self.densities
        osqp_index = list(df.loc[(df['street_object_id'] == street_object_id) & (
            df['modality'] == modality), 'osqp_index'].unique())[0]
        return osqp_index


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt
    from numpy.core.fromnumeric import shape
    from src.model.data_mappings import (
        filter_counts, load_and_preprocess_counts_data)
    from src.model.count_constraints import get_bounds
    from src.model.modality import get_modality_objects
    from src.model.street_grid.intersections import load_intersections
    from src.model.street_grid.street_segments import \
        load_street_segments
    from src.model.street_grid.ml_weights import load_ml_weights

    # load and transform data
    data_folder = 'data/managed_data_files/mobiele_stad/test_set_LC3_b/input'
    filename_streets = 'street_segments.csv'
    filename_intersections = 'intersections.csv'
    filename_data = 'all_data.csv'
    filename_weights = 'ml_weights.csv'
    filename_modality = 'modality_mapping.json'
    filename_bounds = 'count_constraints_bounds.json'

    timestamp = '2020-05-13 13:00:00+00:00'

    THIS_DIR = os.path.dirname(__file__)
    data_files_path = os.path.join(
        THIS_DIR, os.pardir, os.pardir, os.pardir, data_folder)
    street_path = os.path.join(data_files_path, filename_streets)
    intersections_path = os.path.join(data_files_path, filename_intersections)
    data_path = os.path.join(data_files_path, filename_data)
    modalities_path = os.path.join(THIS_DIR, os.pardir, filename_modality)
    bounds_path =  os.path.join(THIS_DIR, os.pardir, filename_bounds)
    weights_path = os.path.join(data_files_path, filename_weights)

    street_segments = load_street_segments(street_path)
    intersections = load_intersections(intersections_path)
    modality_mapping = get_modality_objects(modalities_path)
    bounds = get_bounds(bounds_path)
    counts = load_and_preprocess_counts_data(
        data_path, street_segments, modality_mapping, '60S', False)
    filtered_counts = filter_counts(counts, street_segments)
    ml_weights = load_ml_weights(weights_path, modality_mapping)

    # solving process starts here
    # first-timestamp & mixing
    solver = OSQP_Solver(False, True, silent_output=True, debugging=False)
    solver.load_data(filtered_counts, street_segments,
                     intersections, modality_mapping, ml_weights, bounds)
    solver.prepare()
    solver.update(timestamp)
    solution = solver.solve_iteration().x

    # first-timestamp & no mixing
    solver = OSQP_Solver(False, False, silent_output=True, debugging=False)
    solver.load_data(filtered_counts, street_segments,
                     intersections, modality_mapping, ml_weights, bounds)
    solver.prepare()
    solver.update(timestamp)
    solution = solver.solve_iteration().x

    solver = OSQP_Solver(True, False, silent_output=True, debugging=False)
    solver.set_previous_solution(solution)
    solver.load_data(filtered_counts, street_segments,
                     intersections, modality_mapping, ml_weights, bounds)
    solver.prepare()
    solver.update(timestamp)
    solution = solver.solve_iteration().x

    solver = OSQP_Solver(True, True, silent_output=True, debugging=False)
    solver.set_previous_solution(solution)
    solver.load_data(filtered_counts, street_segments,
                     intersections, modality_mapping, ml_weights, bounds)
    solver.prepare()
    solver.update(timestamp)
    solution = solver.solve_iteration().x

    # """
    # TESTS for the variables
    print('\n VARIABLES')
    print(solver.alphas)
    print(solver.densities)
    print(solver.density_deltas)

    # TESTS for the constraints
    if solver.debugging:
        print('\n CONSTRAINTS')
        print('df_max_density_constraints:\n',
              solver.df_max_density_constraints)
        print('df_max_alpha_constraints:\n', solver.df_max_alpha_constraints)
        print('df_counts_constraints:\n', solver.df_counts_constraints)
        if solver.use_previous_solution:
            print('df_intersection_continuity_constraints:\n',
                  solver.df_intersection_continuity_constraints)
            print('df_density_update_constraints:\n',
                  solver.df_density_update_constraints)

    plt.scatter(np.arange(len(solver.l)), solver.l,
                label='lower bounds', plotnonfinite=True)
    plt.scatter(np.arange(len(solver.u)), solver.u, c='r',
                marker='x', label='upper bound', plotnonfinite=True)
    plt.title('bounds')
    plt.legend()
    plt.show()
    plt.spy(solver.A, marker='+')
    plt.title('spy of the A matrix')
    plt.show()

    print('l :', len(solver.l))
    print('u :', len(solver.u))
    print('A :', shape(solver.A))
    # """

    print('q:', len(solver.q))
    # plt.scatter(np.arange(len(solver.q)),solver.q,label='q')
    plt.scatter(solver.densities['osqp_index'].to_list(
    ), solver.q[solver.densities['osqp_index'].to_list()], color='b', label='densities')
    plt.scatter(solver.alphas['osqp_index'].to_list(
    ), solver.q[solver.alphas['osqp_index'].to_list()], color='r', label='alphas')
    if solver.use_previous_solution:
        plt.scatter(solver.density_deltas['osqp_index'].to_list(
        ), solver.q[solver.density_deltas['osqp_index'].to_list()], color='g', label='density_deltas')
    plt.title('q vector')
    plt.legend()
    plt.show()
    print('P:', shape(solver.P))
    plt.spy(solver.P, marker='+')
    plt.title('spy of the P matrix')
    plt.show()

    # #get a specific modeled value
    # print('DEBUGGING')
    # print(solver.densities.head())
    # street_object_id = 1123458
    # modality = 'mot_density'
    # osqp_index = solver.get_osqp_index_of_density(street_object_id, modality)
    # print('The calculated solution for', street_object_id, ' with modality ', modality, ' equals: ', solution.x[osqp_index])

    plt.title('solution')
    plt.scatter(np.arange(len(solution.index)), solution['solution'])
    plt.show()

    # this step is optional and is for debugging
    # solver.verify(timestamp)

    # this is repeated to test the constraints update doesn't throw errors
    # solver.update_constraints(timestamp)
    # solver.update_constraints(timestamp)
