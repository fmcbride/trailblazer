"""
This file contains the possible functions that can be used as component functions in the model. Each takes the form of 
an outer function with name "[function name]", which takes a set of parameters as arguments, which returns an inner 
function with name "[function_name]_func", which takes a variables as an argument. For example, the first function
on the list takes a parameter "a", and returns the function f(t) = e^{-at}.

The general setup of each definition is the same and can be used as a template for creating additional functions.
"""

import numpy as np

# Component Functions

def logistic(max):
    # logistic curve, derivative 1 at 0, approaches max_speed asymptotically
    # Takes positive scalar inputs, gives positive square outputs
    # Used as a building block of several later functions

    def logistic_func(x):
        return max * (2 / (1 + np.exp(-2 * x / max)) - 1)

    return logistic_func


def inverse_capped(rho, k):
    # inverse function, capped at some high value to avoid division by zero
    def inverse_component(r):
        return 1 / (r ** k)


    def inverse_capped_func(r):
        return np.piecewise(r, (r <= rho, r > rho), (1 / (rho ** k), inverse_component))

    return inverse_capped_func

# MEMORY AND DISTANCE FUNCTIONS
# These functions fill the role of m(t) and D(distance) described in the paper. For all functions f(u) in this section:
#   >f(0) = 1
#   >lim as u-->infinity = 0
#   >Function is non-strictly monotonically decreasing
# These functions take positive scalars as inputs and have positive scalars as outputs


def exp_decay(a):
    # a >= 0
    # exponential decay function
    def exp_decay_func(t):
        return np.exp(-a * t)

    return exp_decay_func


def bump(r1, r2):
    # r2 >=r1 >= 0,
    # returns a bump function, func(r) = 1 for 0<=r<r1, 0 for r >= r2

    def exp_component(x):
        constant = 1 / (r2 - r1) ** 2
        return np.exp(constant + (1 / ((x - r1) ** 2 - (r2 - r1) ** 2)))

    def bump_func(x):
        return np.piecewise(x, (x <= r1, ((x > r1) & (x < r2)), x >= r2), (1, exp_component, 0))

    return bump_func


def logistic_inverse(k):
    # composition of logistic function L(u) defined above with f(x) = 1/(kx^2)
    # L(f(x)) ~= 1 for x~=0 (in some neighborhood of 0, size of the neighborhood depends on k)
    # L(f(x)) ~= f(x) for x>>0
    inverse_comp = inverse_capped(1E-3, 2)
    logistic_comp = logistic(k)

    def logistic_inverse_func(x):
        return logistic_comp(inverse_comp(x)) / k

    return logistic_inverse_func


def constant(a):
    # returns a constant value a for all x
    # can act placeholder for a zero function with a=0
    # violates rule about limit going to zero for a!=0
    def constant_func(x):
        return a

    return constant


# DRIFT FUNCTIONS
# These functions are for the non-memory driven deterministic drift term Theta of the model
# These functions take position vectors as input and give displacement vectors of the same size as output
# All the examples given are gradients of scalar fields (the effect of potential wells, no circulation) but this is not
# the only option for this case

def ornstein_uhlenbeck(mean):
    # takes a mean mu (2D vector) to be the center of attraction
    # output is vector from the input position towards mu with magnitude proportional to distance from mu
    # equivalent to a quadratic (paraboloid) potential well centered at mu
    def ornstein_uhlenbeck_func(x, t):
        return mean - x

    return ornstein_uhlenbeck_func


def conical(mean):
    # takes mean (2D vector) as the center of attraction
    # output is a vector of constant magnitude from the input position to the mean
    # equivalent to a conical potential well centered at the mean
    def conical_func(x, t):
        return (mean - x) / np.linalg.norm(mean - x)

    return conical_func


def logistic_inverse_well(k, mu):
    # takes mean (2D vector) as the center of attraction
    # output is a vector from the input position to mean, with maginitude given by the logistic inverse function above
    # approximately equal to a conical potential well near the mean, and an inverse square potential well far from mean
    direction = conical(mu)
    strength = logistic_inverse(k)

    def logistic_inverse_well_func(x, t):
        return direction(x, t) * strength(np.linalg.norm(x - mu))

    return logistic_inverse_well_func


def adrift():
    # takes no arguments
    # placeholder function for system with no non-memory drift
    # equivalent to a planar potential well
    def adrift_func(x, t):
        return 0

    return adrift_func


# CROSSING PLANE FUNCTIONS

def crossing_plane(a, b, c):
    # defines a crossing plane in implicit form ax + by + c = 0
    # crossing plane component of the code identifies when this function is zero

    def crossing_plane_func(x):
        # x = x[0], y = x[1]
        return a * x[0] + b * x[1] + c

    return crossing_plane_func
