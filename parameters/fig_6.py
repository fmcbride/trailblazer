"""
This file contains all the parameters for a given trial
When the parameters required are functions, they can be drawn from the file "func_list.py"

The class "Parameters" is the sole argument for the solver in side_solver.py
A copy of the parameter file is saved there as well as a record.
"""
import numpy as np
import func_list


class Parameters:
    # Trial Setup
    trial_name = 'fig_6'  # should match file name
    # starting values of x and t
    x = np.array([0, 0])  # 2-vector
    t = 0
    t_span = 200
    save = True  # whether to save the path output. Saving plane crossings or events is determined separately below
    output_directory = 'outputs'  # output directory for saving the path, plane crossings, or events

    # PARAMETERS AND FUNCTIONS
    # Parameter names match parameter names in paper
    # If function takes multiple parameters, give all parameters as a tuple
    # Variables with suffix "_param" MUST be in tuple form, even in they are scalars,
    # e.g. if k is the parameter "example_param", it must be in the form "(k,)"

    # Long Term Memory
    alpha = None  # scalar or none. Positive in our work for attractive long-term memory
    alpha_multiplier = 0  # positive scalar
    calculate_alpha = True  # boolean
    # if calculate_alpha = True, alpha is calculated based on beta, and the short- and long- term memories such that
    # the relative intensity of the short- and long- term memories ("I" in the paper) equals alpha multiplier. In this
    # case, the value for alpha above is not used. Otherwise the value explicitly given for alpha is used
    # calculate_alpha can only be used if short- and long- term memories use the same distance function
    lt_mem = func_list.exp_decay  # function from the func_list.py file
    lt_mem_params = (0.1,)  # parameters for the function above
    lt_dist = func_list.logistic_inverse  # function from the func_list.py file
    lt_dist_params = (5,)  # parameters for the function above

    # Short Term Memory
    beta = -50000  # scalar or none. Negative in our work for repulsive short-term memory
    st_mem = func_list.exp_decay  # function from the func_list.py file
    st_mem_params = (10,)  # parameters for the function above
    st_dist = lt_dist  # function from the func_list.py file
    st_dist_params = lt_dist_params  # parameters for the function above

    # Non-Memory Drift
    gamma = 1  # scalar
    drift = func_list.ornstein_uhlenbeck  # function from the func_list.py file
    drift_params = (np.zeros(2),)  # parameters for the function above

    # Noise Term
    delta = 0.5  # scalar, stdev of distribution of noise over 1 time step

    # CROSSING PLANE
    # Records when the trajectory crosses a given curve
    # Records when crossing curve = 0 and whether its value is crossing from positive to negative or vice versa
    # If crossing_curve == func_list.crossing_plane, exact point on trajectory intersecting crossing plane is returns
    # Otherwise, last point before crossing is returned
    plane_crossings = True  # whether to record plane crossings
    # Multiple crossing curves can be listed in list format
    # If multiple curves are listed, crossing_curve[i] will be paired with crossing_curve_params[i]
    crossing_curve = func_list.crossing_plane
    # crossing curve params must be listed as tuples, or list of tuples, if applicable
    crossing_curve_params = [(0, 1, 0), (1, 0, 0)]

    # EVENTS
    # records times when certain defined events occur
    # events can be terminal or non-terminal
    events = False
    terminal = False
    st_discont = 0
    lt_discont = 0
    current_index = 0

    # COMPUTATIONAL CONTROLS

    n = 10000  # Euler method steps per unit time: should be at least 5000
    seed = None  # seed for pseudo RNG. If "None", defaults to clock time
