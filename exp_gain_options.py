import numpy as np

import gaussian_cdf as norm
import irwin_hall
import utilities as ut

def size_gain(d, paths_pred, time, pos_obs, neg_obs, candidates_set, s, approx, 
        noise=0, tol_c=1):

    """Compute the expected gain"""

    #pos_obs, neg_obs = observations[0], observations[1]
    first_observation = ut.first_inf_node(pos_obs)
    #dist_to_sent, closer_sent = min([(d[s][z], z) for z in pos_obs])
    #closer_observation = (closer_sent, pos_obs[s_1])
    
    # time of first observation
    s_1, t_1 = first_observation
    
    if noise == 0:
        min_t = min([d[n][s] - d[n][s_1] + t_1 for n in candidates_set])
        max_t = max([d[n][s] - d[n][s_1] + t_1 for n in candidates_set])
        min_t = int(min(min_t, time))
        max_t = int(min(max_t, time))
    else:
        upper_bound = []
        lower_bound = []
        for n in candidates_set:
            lower_bound.append(max([d[n][s] - d[n][o] - noise * (d[n][s] + d[n][o])
                + pos_obs[o] for o in pos_obs] + [d[n][s] - d[n][o] - noise * (d[n][s] + d[n][o])
                + neg_obs[o] for o in neg_obs]))
            upper_bound.append(min([d[n][s] - d[n][o] + noise * (d[n][s] + d[n][o])
                + pos_obs[o] for o in pos_obs]))

        min_t = int(min(min(lower_bound), time))
        max_t = int(min(max(upper_bound), time))
        #max_dist = max([d[n][s_1] + ut.tol(d[n][s_1], noise=noise, tol_c=tol_c) for n
        #        in candidates_set])
        #min_t = int(t_1 - max_dist)
        #max_t = int(time)

    step_h = 1 #stepsize for the computation of the integral 
    h_values = range(min_t, max_t + 1, step_h)

    n = len(candidates_set)
    m = len(h_values) + 1
    compat = np.zeros([n, m])
    weights = np.zeros([n, m])

    obs, times = pos_obs.keys(), pos_obs.values()
    neg_obs, neg_times = neg_obs.keys(), neg_obs.values()

    for (i, cand) in enumerate(candidates_set):
        for (j, h) in enumerate(h_values):
            compat[i,j] = h_compatible_all(d, obs, times, neg_obs, neg_times, cand, s, h, 
                    noise=noise, tol_c=tol_c)
            # compute weight h
            if approx == 'det':
                weights[i,j] = compat[i,j] / float(n)
            if approx == 'gauss':
                # assuming uniform prior on the position of the source
                weights[i, j] = prob_s_inf_at_h_gauss(cand, d, paths_pred, noise, s, 
                        h, s_1, t_1, time, step_h=step_h) / float(n)
            if approx == 'unif':
                # assuming uniform prior on the position of the source
                weights[i, j] = prob_s_inf_at_h_unif(cand, d,
                        paths_pred, noise, s, h, s_1, t_1, time, 
                        step_h=step_h)
        
        # nodes that are compatible with s not yet infected
        compat[i, m-1] = later_compatible_all(d, time, obs, times, cand,
                s, noise=noise, tol_c=tol_c)
        
        if approx == 'det':
            weights[i, m-1] = compat[i, m-1] / float(n)

        if approx == 'gauss':
            # assuming uniform prior on the position of the source
            weights[i, m-1] = prob_s_inf_later_gauss(cand, d, paths_pred,
                    noise, s, s_1, t_1, time, step_h=step_h) / float(n)
        if approx == 'unif':
            # assuming uniform prior on the position of the source
            weights[i, m-1] = prob_s_inf_later_unif(cand, d, paths_pred, 
                    noise, s, s_1, t_1, time, step_h=step_h) / float(n)

    return np.dot(n*np.ones(m) - np.sum(compat, axis=0), np.sum(weights, axis=0))


def h_compatible_all(d, obs, times, neg_obs, neg_times, cand, c, t_c, noise=0, tol_c=1):
    """ Select candidates sources that are compatible with c being infected at
    time t_c
    
    """
    
    for o, t in zip(obs, times):
        if not (abs(d[o][cand] - d[c][cand] - t + t_c) <=
                ut.tol(d[o][cand]+d[c][cand], noise=noise, tol_c=tol_c)):
            return False
    
    for o, t in zip(neg_obs, neg_times):
        if not (t - t_c - d[o][cand] + d[c][cand] <=
                ut.tol(d[o][cand]+d[c][cand], noise=noise, tol_c=tol_c)):
            return False

    return True
    

def later_compatible_all(d, time, obs, times, cand, c, noise=0, tol_c=1):
    """Select candidates sources that are compatible with c being not infected
        yet
    
    """
    
    if time == np.inf:
        return False
    else:
        for o, t in zip(obs, times):
            if not (d[o][cand] + time - t < d[c][cand] +
                    ut.tol(d[o][cand]+d[c][cand], noise=noise, tol_c=tol_c)):
                return False
        return True


def prob_s_inf_at_h_gauss(cand, d, paths_pred, sigma, s, h, s_1, t_1,
        t, step_h=1):
    """Computes the (approximated) probability that s is infected in [h-1/2,
        h+1/2]
    
       (s_1, t_1) is the reference observation

       t is the actual time
    """
    
    var = var_path(d, paths_pred, sigma, cand, s, s_1)

    assert h <= t

    # mean of t_s - t_1
    mu_y = d[cand][s] - d[cand][s_1]
    
    # std_dev of t_s - t_1
    sigma_y = np.sqrt(var)

    # interval for normalized t_s - t_1
    c_1 = (h - step_h*0.5 - t_1 - mu_y) / sigma_y
    c_2 = (min(t, h + step_h*0.5) - t_1 - mu_y) / sigma_y

    # compute probability as cdf difference
    p = (norm.cdf(c_2) - norm.cdf(c_1))
    
    return p


def prob_s_inf_at_h_unif(cand, d, paths_pred, noise, s, h, s_1, t_1, t, 
        step_h=1):
    """Computes the (approximated) probability that s is infected in [h-1/2,
        h+1/2]
    
       (s_1, t_1) is the reference observation

       t is the actual time
    """

    assert h <= t

    if isinstance(paths_pred[0][0], list):
        m = len(non_comm_path(paths_pred, cand, s, s_1))
    else:
        m = len(recover_non_comm_paths(paths_pred, cand, s, s_1))
    transl = - (d[cand][s] - d[cand][s_1]) + noise*m

    # interval 
    c_1 = (transl + h - step_h*0.5 - t_1) / (2*noise)
    c_2 = (transl + min(t, h + step_h*0.5) - t_1) / (2*noise)
    # compute probability as cdf difference
    p = (irwin_hall.cdf(m, c_2) - irwin_hall.cdf(m, c_1))

    # assuming uniform prior on the position of the source

    return p


def prob_s_inf_later_gauss(cand, d, paths_pred, sigma, s, s_1, t_1, t,
        step_h=1):
    """Computes the (approximated) probability that s is infected in [t,
        + inf]
    
       (s_1, t_1) is the reference observation

       t is the actual time
    """

    if t == np.inf:
        return 0

    var = var_path(d, paths_pred, sigma, cand, s, s_1)

    # mean of t_s - t_1
    mu_y = d[cand][s] - d[cand][s_1]
    
    # std_dev of t_s - t_1 
    sigma_y = var * (sigma**2) / 3.0 

    # lower bound for normalized t_s - t_1
    c = (t - step_h*0.5 - t_1 - mu_y) / sigma_y

    # compute probability as cdf difference
    p = 1 - norm.cdf(c)
    
    return p


def prob_s_inf_later_unif(cand, d, paths_pred, noise, s, s_1, t_1, t, 
        step_h=1):
    """Computes the (approximated) probability that s is infected in [t,
        + inf]
    
       (s_1, t_1) is the reference observation

       t is the actual time
    """

    if t == np.inf:
        return 0
    
    if isinstance(paths_pred[0][0], list):
        m = len(non_comm_path(paths_pred, cand, s, s_1))
    else:
        m = len(recover_non_comm_paths(paths_pred, cand, s, s_1))

    transl = - (d[cand][s] - d[cand][s_1]) + noise*m
 
    c = (transl + t - step_h*0.5 - t_1) / (2*noise)

    # compute probability as cdf difference
    p = 1 - irwin_hall.cdf(m, c)

    # assuming uniform prior on the position of the source
    
    return p

def var_path(d, paths_pred, sigma, n, s, s_1):

    if isinstance(paths_pred[0][0], list):
        non_comm_edges = non_comm_path(paths_pred, n, s, s_1)
    else:
        non_comm_edges = recover_non_comm_paths(paths_pred, n, s, s_1)
       
    var = 0

    for (u,v) in non_comm_edges:
        var += ((d[u][v] * sigma)**2)/3.0

    return var


def non_comm_path(paths, n, s, s_1):

    path_1 = paths[n][s_1]
    path_1_edges = [(path_1[i], path_1[i+1]) for i in range(len(path_1)-1)]
    
    path_s = paths[n][s]
    path_s_edges = [(path_s[i], path_s[i+1]) for i in range(len(path_s)-1)]

    non_comm_edges = set(path_1_edges) ^ set(path_s_edges)

    return non_comm_edges


def recover_non_comm_paths(pred, s, t1, t2):
    p1 = [t1]
    p2 = [t2]
    while p1[-1] not in p2 and p2[-1] not in p1:
        p1.append(pred[s][p1[-1]])
        p2.append(pred[s][p2[-1]])
    try:
        i = p2.index(p1[-1])
        p2 = p2[:i+1]
    except ValueError:
        "Do nothing"
    try:
        j = p1.index(p2[-1])
        p1 = p1[:j+1]
    except ValueError:
        "Do nothing"
    
    p1 = zip(p1[:-1], p1[1:])
    p2 = zip(p2[:-1], p2[1:])
    return p1 + p2 


def drs_gain(d, processed, cand_sources, c, time):
    """DRS Gain taking into account the time in the detection process
    
    When adding c, the number of equivalence classes is the number of different
    values t_1 - d(v, s_1) + d(v, c) that are smaller or equal than 'time'. If
    there are values that are larger, we add 1 to denote the unique class among
    which we cannot distinguish yet.

    If time == np.inf we have the standard DRS gain (which counts the number
    of classes in which the cand sources are divided given that all infection
    times can be observed).

    d = distance matrix
    s_1 = reference observer
    t_1 = inf time of s_1
    cand_sources = list of cand sources at time 'time'
    c = candidate observer
    time = absolute time in the detection process

    """

    s_1, t_1 = ut.first_inf_node(processed)
    values = set()
    not_observed = False
    for v in cand_sources:
        if t_1 - d[s_1][v] + d[c][v] <= time:
            values.add(int(10**8)*(t_1 - d[s_1][v] + d[c][v]))
        else: 
            not_observed = True
    if not_observed:
        return len(values) + 1
    else:
        return len(values)
