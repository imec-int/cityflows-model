# Class structure of the solvers

In this document the classes defined in the src/model/solve directory and their underlying structure will be described. There was a need for multiple solvers as discussed in [model_explanation.md](../../../model_explanation.md)

# smart_solver class
This class is defined in `smart_solver.py`. Object of this class will be used to solve a series of iterations. Like discussed in the `model_explanation.md` this means we have three sets of constraints and thus we need three solvers. These will be created in a solver class object. One of each of the three will be implemented in a smart_solver object.

# solver class
Like suggested above there are multiple solvers required. There will be one for the first timestep. Then for the other iterations there will be a solver which doesn't mix the modalities and for every plural of the modality_mixing_step it will mix the modalities.
The set of constraints isn't the only factor that is to be taken into account. There are also multiple software packages to use to solve the problem. In this repo there are already 2 implemnted: the opensource OSQP package and the licensed Gurobi package.

Because of all the varieties required the chose has been made to make an abstract solver class in `solver.py`. For the different software packages there will be a subclass of this abstract solver class, they will be called solver_types.

The osqp solver_type is the one that is most up to date and is recomended to be used. The other type, Gurobi, doesn't have everything mentioned in the `model_explanation.md` and also can contain errors. It is more of a legacy solver, which was abandoned for the sake of using opensource software.

## osqp_solver class
In `osqp_solver.py` a subclass osqp_solver of the solver class will be defined. Here it has the option to adhere to one of the three sets of constraints mentioned in `model_explanation.md`. It is written so that the problem is in the format the [osqp](https://osqp.org/) software package can be used to solve it. 


## gurobi_solver class
In `gurobi_solver.py` a subclass gurobi_solver of the solver class will be defined. Here it has the option to adhere to one of the three sets of constraints mentioned in `model_explanation.md`. It is written so that the problem is in the format the [gurobi](https://www.gurobi.com/) software package can be used to solve it. this is the legacy solver at this point.