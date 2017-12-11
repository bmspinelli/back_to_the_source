import utilities as ut

def update_cand_sources(d, o, t_o, processed, cand_sources, pres_time,
        sensors, noise=0, tol_c=1.0, online=True):
   
    cand_sources, stop = update_cand_set(d, processed, cand_sources, o, t_o, 
            pres_time, sensors, noise=noise, tol_c=tol_c, online=online)
    
    if len(cand_sources) > 0:
        # the algorithm continues
        stop = False
    else:
        # nothing else to do :( (should occurr only if tol_c < 1)
        assert tol_c < 1
        stop = True

    return cand_sources, stop


def update_cand_set(d, processed, candidates_set, new, t_new, pres_time,
        sensors, noise=0, tol_c=1, online=True):

    # initialize new candidates set
    updated = []
    if online:
        neg_obs = filter(lambda x: x not in processed.keys(), sensors)
        neg_times = [pres_time for x in neg_obs]
    else:
        neg_obs, neg_times = [], []

    for cand in candidates_set:
        # check compatibility with all observations
        if compatible_all(d, processed.keys(), processed.values(), 
                neg_obs, neg_times, cand, new, t_new, pres_time, noise=noise, tol_c=tol_c):
            updated.append(cand)
    
    if len(updated) > 0:
        # the algorithm continues
        stop = False
        return updated, stop 
    else:
        # nothing else to do :( (should occurr only if tol_c < 1)
        #assert tol_c < 1
        stop = True
        # return the candidate set as at the previous iteration
        return candidates_set, stop


def update_algo_state(graph, seq_cand_sources, cand_sources, success, infected,
        infection_times, pres_time, real_source):
    
    seq_cand_sources.append(len(cand_sources))
    infected.append(len(filter(lambda x: infection_times[x] <= pres_time,
            graph.nodes())))
    success.append(int(real_source in cand_sources))
    
    return seq_cand_sources, infected, success


def compatible_all(d, obs, times, neg_obs, neg_times, cand, c, t_c, pres_time, noise=0, tol_c=1):
    """ Select candidates sources that are compatible with c being infected at
    time t_c
    
    """
    if t_c == None:
        for o, t in zip(obs, times):
            if not (pres_time - t - d[c][cand] + d[o][cand] <=
                    ut.tol(d[o][cand]+d[c][cand], noise=noise, tol_c=tol_c)):
                return False
    else:
        for o, t in zip(obs, times):
            if not (abs(d[o][cand] - d[c][cand] - t + t_c) <=
                    ut.tol(d[o][cand]+d[c][cand], noise=noise, tol_c=tol_c)):
                return False
    
        for o, t in zip(neg_obs, neg_times):
            if not (t - t_c - d[o][cand] + d[c][cand] <=
                    ut.tol(d[o][cand]+d[c][cand], noise=noise, tol_c=tol_c)):
                return False

    return True
