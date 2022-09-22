"""

"""
import numpy as np
from scipy import integrate
import time
import datetime
import os
from parameters import fig_2, fig_3, fig_4, fig_5, fig_6, fig_a1, fig_a2
import func_list


def parameter_integrator(params, cutoff, term='long'):
    # this function calculates a value associated with the intensities of the short- and long- term memories
    assert not (
                params.calculate_alpha and params.lt_dist is not params.st_dist), "calculate_alpha can only be True when long- and short- term" \
                                                                                  " memory use the same distance function. Please declare your " \
                                                                                  "alpha value explicitly"
    if term == 'long':
        memory = params.lt_mem(*params.lt_mem_params)
        return integrate.quad(memory, 0, cutoff)[0]
    elif term == 'short':
        memory = params.st_mem(*params.st_mem_params)
        return integrate.quad(memory, 0, cutoff)[0]


def integrator(x, t, path, mem_func, dist_func):
    # line integral of function of mem_func, dist_func over past path
    # Riemann sum: evaluates value at all past points then sums
    times = t - path[0]
    mem_weights = mem_func(times)
    dist_vector = path[1:] - x.reshape(2, 1)
    distance = np.linalg.norm(dist_vector, axis=0)
    nonzero = distance.astype(bool)
    mem_weights = mem_weights[nonzero]
    distance = distance[nonzero]
    dist_vector = dist_vector[..., nonzero]
    path = path[..., nonzero]
    vectors = mem_weights * dist_func(distance) * dist_vector / distance
    hs1 = np.zeros(np.size(path[0]) + 2)
    hs2 = np.zeros(np.size(path[0]) + 2)
    hs1[:-2] = path[0]
    hs2[2:] = path[0]
    hs = 0.5 * (hs1 - hs2)[1:-1]
    hs[0] = (path[0, 1] - path[0, 0]) / 2
    hs[-1] = (path[0, -1] - path[0, -2]) / 2
    return np.sum(vectors * hs, axis=1)


def det_step(path, p, cutoff=False):
    # calculates long-term, short-term, and non-memory drift components of the next trajectory step
    # calculates based on current location, current time, and past track
    # returns 2-vector of step (unscaled for time step length)
    point_count = len(path[0])
    # long-term memory component
    # long-term cutoff not included because it is applied to path before input in main
    if p.alpha and point_count > 1:
        lt_comp = integrator(p.x, p.t, path, p.lt_mem(*p.lt_mem_params), p.lt_dist(*p.lt_dist_params))
    else:
        lt_comp = np.zeros(2)
    # short-term memory component
    path = path[..., p.st_discont:]  # trims path before last discontinuity
    point_count = len(path[0])
    if p.beta and point_count > 1:
        # integrates short-term memory component over path section beginning with t-st_cutoff if st_cutoff given
        cutoff_n = int(p.n * p.st_cutoff)
        if cutoff_n <= len(path[0]):
            st_comp = integrator(p.x, p.t, path, p.st_mem(*p.st_mem_params),
                                 p.st_dist(*p.st_dist_params))
        else:
            st_comp = integrator(p.x, p.t, path[..., -cutoff_n:], p.st_mem(*p.st_mem_params),
                                 p.st_dist(*p.st_dist_params))
    else:
        st_comp = np.zeros(2)
    # non-memory drift component
    if p.gamma:
        drift_comp = p.drift(*p.drift_params)(p.x, p.t)
    else:
        drift_comp = np.zeros(2)
    return p.alpha * lt_comp + p.beta * st_comp + p.gamma * drift_comp


def main(p):
    """
    Stochastic Integro-Differential Equation (SIDE) solver.
    Argument:
        >class Parameters from parameters.py. Individual parameters explained in that file
    Returns:
        >path: ndarray [ts, xs, ys] of the solution trajectory
        >plane_crossings: ndarray [crossing_ts, crossing_xs, crossing_ys, crossing directions]
        See description in parameters.py section PLANE CROSSINGS
        >events: ndarray [event_types, event_ts]
        See description in parameters.py section EVENTS

    If parameters "save" and/or "plane_crossing" are True, an output folder is created with the trajectory and plane crossing
    record saved as csv files. In this case, copy of the parameter file is saved there as well as a record.
    """
    # SOLVER SETUP
    tic_main = time.time()
    # setting cutoffs for integration over the past track: only time t-cutoff will be integrated over
    proportion = 5E-3
    # this sets the long-term memory integration cutoff such that the integral of the long-term memory function
    # from 0 to the cutoff point is equal to 1-proportion of the integral from 0 to infinity. For proportion == 5E-3
    # this corresponds to a range of integration that captures 99.5% of the integral. The smaller the value of
    # proportion, the more accurate the solution will be.
    # this proportion is only used for exponential decay since the full integral can be captures over a finite range of
    # integration in the case of the bump function. If a different function is used, a default value is assigned.
    if p.lt_mem == func_list.exp_decay:
        p.lt_cutoff = -np.log(proportion) / p.lt_mem_params
    elif p.lt_mem == func_list.bump:
        p.lt_cutoff = p.lt_mem_params[1]
    else:
        p.lt_cutoff = 10
    print('Long-Term Memory Integration Timeframe: {0:.3f}s'.format(p.lt_cutoff[0]))
    if p.st_mem == func_list.exp_decay:
        p.st_cutoff = -np.log(proportion) / p.st_mem_params
    elif p.st_mem == func_list.bump:
        p.st_cutoff = p.st_mem_params[1]
    else:
        p.st_cutoff = 1
    print('Short-Term Memory Integration Timeframe: {0:.3f}s'.format(p.st_cutoff[0]))

    # initializing the functions used
    # p.lt_mem(*p.lt_mem_params), p.lt_dist(*p.lt_dist_params), p.st_mem(*p.st_mem_params), p.st_dist(*p.st_dist_params)

    # if alpha is defined by the relative intensity of short- and long- term memories, this section determines its value
    if p.calculate_alpha:
        if p.alpha_multiplier:
            int_crit = -p.beta * parameter_integrator(p, p.st_cutoff, term='short')
            integral_long = parameter_integrator(p, p.lt_cutoff, term='long')
            p.alpha = p.alpha_multiplier * int_crit / integral_long
            print("Calculated alpha: {}".format(p.alpha))
        else:
            p.alpha = 0

    # solver setup
    h = 1 / p.n
    path = np.reshape(np.append(p.t, p.x), (2 + 1, 1))
    x_old = p.x
    t_0 = p.t
    index_0 = np.shape(path)[1] - 1
    np.random.seed(p.seed)
    segment_count = 0
    last_save = 0
    print(p.__name__)
    # output file setup
    if p.save or (p.plane_crossings or p.events):
        if not os.path.exists(p.output_directory + '/{}'.format(p.trial_name)):
            os.mkdir(p.output_directory + '/{}'.format(p.trial_name))
            os.system('cp parameters/' + p.trial_name + '.py ' + p.output_directory + '/{}/trial_parameters.py'.format(p.trial_name))
            if p.save:
                os.mkdir(p.output_directory + '/{}/partial_saves'.format(p.trial_name))

    # progress bar setup
    report_frequency = 20
    report_interval = p.t_span / report_frequency
    report_time = t_0 + report_interval
    print('Evaluating...', end='')

    # plane-crossing and events setup:
    if p.plane_crossings:
        plane_count = len(p.crossing_curve_params)
        crossing_funcs = []
        linear_plane = []  # bool list storing whether crossing curve is linear
        for i in range(plane_count):
            if type(p.crossing_curve) == list:
                crossing_funcs.append(p.crossing_curve[i](*p.crossing_curve_params[i]))
                linear_plane.append(p.crossing_curve[i] == func_list.crossing_plane)
            else:
                crossing_funcs.append(p.crossing_curve(*p.crossing_curve_params[i]))
                linear_plane.append(p.crossing_curve == func_list.crossing_plane)
        crossing_ts = []
        crossing_xs = []
        crossing_ys = []
        crossing_directions = []
        crossing_plane_index = []
    if p.events:
        event_count = len(p.event_list)
        event_types = []
        event_ts = []

    # SOLVER
    while p.t < t_0 + p.t_span and not p.terminal:
        # extends path array to make space for 1 time unit's results (n steps per time unit)
        new_path = np.zeros((2 + 1, p.n))
        path = np.append(path, new_path, axis=1)

        # integrates the trajectory over one time unit using Euler's method
        for i in range(1, p.n + 1):
            i += index_0
            if p.delta:
                rand_comp = p.delta * np.random.randn(2)
            else:
                rand_comp = 0
            if int(p.n * p.lt_cutoff) >= len(path[0, :i - 1]):
                path_start = 0
            else:
                # restricts path integrated over to the section beginning with t-lt_cutoff
                path_start = i - int(p.n * p.lt_cutoff) - 1
            # Euler step: combines the deterministic part given by det_step with Weiner process
            p.x = p.x + h * (det_step(path[..., path_start:i - 1], p)) + np.sqrt(h) * rand_comp
            t_old = p.t
            p.t += h
            # progress bar: reports percentage complete in terms of total time range, every 5%
            if p.t > report_time:
                print('{}%...'.format(round(100 * (report_time - t_0) / p.t_span)), end='')
                report_time += report_interval

            # plane crossing check
            if p.plane_crossings:
                for k in range(plane_count):
                    old_sign = np.sign(crossing_funcs[k](x_old))
                    new_sign = np.sign(crossing_funcs[k](p.x))
                    if old_sign != new_sign:
                        # for linear plane crossings, finds exact intersection of linear interpolation
                        # between points and the plane. For nonlinear plane crossings, simply records
                        # last point before the crossing
                        if linear_plane[k]:
                            a, b, c = p.crossing_curve_params[k]
                            x0, y0 = x_old
                            x1, y1 = p.x
                            s_crit = (a * x0 + b * y0 + c) / (a * (x0 - x1) + b * (y0 - y1))
                            crossing_ts.append(t_old + s_crit * (p.t - t_old))
                            crossing_xs.append(x0 + (x1 - x0) * s_crit)
                            crossing_ys.append(y0 + (y1 - y0) * s_crit)
                        else:
                            crossing_ts.append(t_old)
                            crossing_xs.append(x_old[0])
                            crossing_ys.append(x_old[1])
                        if old_sign > new_sign:
                            crossing_directions.append(-1)
                        else:
                            crossing_directions.append(1)
                        crossing_plane_index.append(k)
            # events_check
            if p.events:
                for v in range(event_count):
                    p.current_index = i
                    if p.event_list[v](p):
                        event_ts.append(p.t)
                        event_types.append(v)

            # partial saves
            if p.t > last_save + 1:
                if last_save:
                    save_index = np.argmax(path[0] >= last_save)
                else:
                    save_index = 0
                if p.save:
                    np.savetxt('outputs/{}/partial_saves/segment_{}.csv'.format(p.trial_name, segment_count),
                               path[..., save_index:i],
                               delimiter=',')
                if p.plane_crossings:
                    np.savetxt('outputs/{}/plane_crossings.csv'.format(p.trial_name),
                               np.array(
                                   [crossing_ts, crossing_xs, crossing_ys, crossing_directions, crossing_plane_index]),
                               delimiter=',')
                if p.events:
                    np.savetxt('outputs/{}/events.csv'.format(p.trial_name),
                               np.array([event_types, event_ts]), delimiter=',')
                last_save = p.t
                segment_count += 1
            # removes unneeded points in active memory
            if p.t - path[0, 0] > p.lt_cutoff + 1:
                t_to_keep = p.t - p.lt_cutoff
                cutoff_index = np.argmax(path[0] > t_to_keep)
                path = path[..., cutoff_index:]
                index_0 -= cutoff_index
                p.st_discont -= cutoff_index
                p.lt_discont -= cutoff_index
                i -= cutoff_index

            path[0, i] = p.t
            path[1:, i] = p.x
            x_old = p.x
            # exits if solving is complete
            if (p.t > t_0 + p.t_span) or p.terminal:
                path = path[..., :i + 1]
                if p.terminal:
                    print('A terminal event occurred: ' + p.terminal_message)
                break
        index_0 += p.n

    # CONCLUDING STEPS
    # reassembling partial saves into single path and deletes partial saves
    save_index = np.argmax(path[0] > last_save)
    path = path[..., save_index:]
    segment_list = []
    for j in range(segment_count):
        segment_list.append(
            np.loadtxt('outputs/{}/partial_saves/segment_{}.csv'.format(p.trial_name, j), delimiter=','))
    segment_list.append(path)
    if segment_count:
        os.system('rm -r outputs/{}/partial_saves'.format(p.trial_name))
    path = np.concatenate(segment_list, axis=1)
    plane_crossings = None
    events = None
    if p.plane_crossings:
        plane_crossings = np.array([crossing_ts, crossing_xs, crossing_ys, crossing_directions, crossing_plane_index])
        np.savetxt('outputs/{}/plane_crossings.csv'.format(p.trial_name), plane_crossings, delimiter=',')
    if p.events:
        events = np.array([event_types, event_ts])
        np.savetxt('outputs/{}/events.csv'.format(p.trial_name), events, delimiter=',')
    if p.save:
        np.savetxt('outputs/{}/path.csv'.format(p.trial_name), path, delimiter=',')
    print('\nTotal Runtime: {}'.format(datetime.timedelta(seconds=round(time.time() - tic_main))))
    return path, plane_crossings, events


if __name__ == '__main__':
    # make sure that your parameter file has been imported
    # the parameters to produce the tracks displayed in figs 2-6 and A1, A2 are imported
    param_file = fig_a1
    main(fig_a1.Parameters)
