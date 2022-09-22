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
    trial_name = 'fig_a2'  # should match file name
    # starting values of x and t
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
    alpha = 0.8  # scalar or none. Positive in our work for attractive long-term memory
    alpha_multiplier = None  # positive scalar
    calculate_alpha = False  # boolean
    # if calculate_alpha = True, alpha is calculated based on beta, and the short- and long- term memories such that
    # the relative intensity of the short- and long- term memories ("I" in the paper) equals alpha multiplier. In this
    # case, the value for alpha above is not used. Otherwise the value explicitly given for alpha is used
    # calculate_alpha can only be used if short- and long- term memories use the same distance function
    lt_mem = func_list.exp_decay  # function from the func_list.py file
    lt_mem_params = (0.05,)  # parameters for the function above
    lt_dist = func_list.logistic_inverse  # function from the func_list.py file
    lt_dist_params = (10,) # parameters for the function above

    # Short Term Memory
    beta = -20  # scalar or none. Negative in our work for repulsive short-term memory
    st_mem = func_list.exp_decay  # function from the func_list.py file
    st_mem_params = (5,)  # parameters for the function above
    st_dist = lt_dist  # function from the func_list.py file
    st_dist_params = lt_dist_params  # parameters for the function above

    # Non-Memory Drift
    def polygon(k, r):
        # returns a k-gon with radius r centered on the origin, first point at (r,0)
        # returns array with form [xs, ys]
        t = 2 * np.pi * np.arange(0, 1, 1 / k)
        return np.array([r * np.cos(t), r * np.sin(t)])

    poly_sides = 3
    target_list = polygon(poly_sides, 3)

    gamma = 80  # scalar
    drift = func_list.logistic_inverse_well  # function from the func_list.py file
    current_target = 1
    old_target = 0
    x = target_list[..., 0]
    drift_params = (10, target_list[..., 1])  # parameters for the function above

    # Noise Term
    delta = 0.1  # scalar, stdev of distribution of noise over 1 time step

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
    events = True
    terminal = False
    terminal_message = ''

    # events will be functions called on the current location which will return True or False,
    # change some bool to terminal, and maybe alter other parameters

    def inside(self, center, radius, norm=2, inside=True, strict=False):
        # checks whether x is within distance radius of center
        # returns True or False
        dist = np.linalg.norm(self.x - center, norm)
        if strict:
            return_bool = dist < radius
        else:
            return_bool = dist <= radius
        if inside:
            return return_bool
        else:
            return not return_bool

    st_discont = 0
    lt_discont = 0
    current_index = 0

    def arrive_destination(self):
        # checks to see if trajectory is within distance 0.1 of some target point
        # then changes to new target, resets short-term memory
        if self.inside(self, self.drift_params[1], 0.1):
            self.old_target = self.current_target
            while self.current_target==self.old_target:
                self.current_target = np.random.choice(self.poly_sides)
            print('Target: {}'.format(self.current_target))
            self.drift_params = (10, self.target_list[..., self.current_target])
            self.st_discont = self.current_index
            # TODO enable short- and long-term resets
            return True
        else:
            return False

    def leave_box(self):
        # checks if trajectory has left box of size length 16 centered at the origin
        # terminal if True
        if self.inside(self, np.zeros(2), 8, norm=np.inf, inside=False):
            self.terminal = True
            self.terminal_message = 'the trajectory got lost'
            return True
        else:
            return False

    event_list = [arrive_destination, leave_box]

    # COMPUTATIONAL CONTROLS

    n = 5000  # Euler method steps per unit time: should be at least 5000
    seed = None  # seed for pseudo RNG. If "None", defaults to clock time
