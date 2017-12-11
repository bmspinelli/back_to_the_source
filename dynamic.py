import random
import numpy as np
from scipy.stats import rv_discrete

import updates as up
import exp_gain_options as eg
import utilities as ut


def dynamic(graph, d, d_1, paths_pred, init_sensors, infection_times, budget, delay, 
        noise=0, tol_c=0, gain ='size', approx='det', tested_max=float('INFINITY'), 
        real_source=None, online=True):

    assert budget >= 1 
    
    sensors = init_sensors[:]

    budget = min(budget, (len(graph) - len(sensors)))

    assert delay > 0

    sorted_to_process = sorted(zip([infection_times[s] for s in sensors], sensors))

    min_time = sorted_to_process[0][0]
    max_time = max(infection_times.values())
    
    if online:
        pres_time = min_time
    else: 
        pres_time = max_time

    cand_sources = graph.nodes()
    
    if online:
        cand_sources, stop = up.update_cand_sources(d, sorted_to_process[0][1],
                min_time, {}, cand_sources, pres_time, 
                sensors, noise=noise, tol_c=tol_c, online=online)
    else:
        pass 
    processed = {sorted_to_process[0][1]: min_time}

    # times at which I choose a new sensor
    # put inf as sensor so that they come later in order 
    dyn_to_process = [(pres_time + delay * x, np.inf) for x in range(budget)]

    # all times at which I will update with possibly the related sensor
    to_process = sorted(sorted_to_process[1:] + dyn_to_process, reverse = True)

    seq_cand_sources = [len(cand_sources)]
    
    infected = [len(filter(lambda x: infection_times[x] <= pres_time,
            graph.nodes()))]

    success = [int(real_source in cand_sources)]
    stop = False
    restricted = False
    # While did not converge to only 1 candidate source
    while len(cand_sources) > 1 and (not stop) and len(to_process)>0:
        
        tmp_time, tmp_node = to_process.pop()

        if online:
            pres_time = tmp_time
        else:
            pass

        # if I should select a dynamic sensor
        if tmp_node == np.inf:
            din_sens = best_dynamic_sensor(d, d_1, paths_pred, sensors, 
                    processed, cand_sources, pres_time, gain=gain, 
                    tested_max=tested_max, noise=noise, tol_c=tol_c,
                    approx=approx, online=online, restricted=restricted)
            # if it is not yet infected
            if infection_times[din_sens] > pres_time:
                to_process.append((infection_times[din_sens], din_sens))
                to_process = sorted(to_process, reverse=True)
                cand_sources, stop = up.update_cand_sources(d, din_sens,
                        None, processed, cand_sources, pres_time, 
                        sensors, noise=noise, tol_c=tol_c, online=online)
            # if it is already infected
            else:
                obs_time = infection_times[din_sens]
                cand_sources, stop = up.update_cand_sources(d, din_sens,
                        obs_time, processed, cand_sources, pres_time, 
                        sensors, noise=noise, tol_c=tol_c, online=online)
                processed[din_sens] = obs_time
            sensors.append(din_sens)
        # if I am updating on an old sensor that got infected
        else:
            cand_sources, stop = up.update_cand_sources(d, tmp_node,
                tmp_time, processed, cand_sources, pres_time, 
                sensors, noise=noise, tol_c=tol_c, online=online)
            processed[tmp_node] = tmp_time

        seq_cand_sources, infected, success = up.update_algo_state(graph, 
                seq_cand_sources, cand_sources, success, infected,
                infection_times, pres_time, real_source)
        #if no progress
        if seq_cand_sources[-1] == seq_cand_sources[-2]:
            restricted = True
        
        print processed
        print seq_cand_sources[-1], success[-1]

    time = pres_time - min(infection_times.values())

    return sensors, cand_sources, seq_cand_sources, time, infected, success


def best_dynamic_sensor(d, d_1, paths_pred, sensors, processed, cand_sources,
        pres_time, gain='size', tested_max=float('INFINITY'), noise=0, tol_c=1,
        approx='det', restricted=False, online=True):
    
    if gain == 'random':
        return random.choice(cand_sources)
    
    all_candidates = filter(lambda x: x not in sensors, range(len(d_1)))
    random.shuffle(all_candidates)
    
    if restricted:
        all_candidates = filter(lambda x: x in cand_sources, all_candidates)

    if tested_max < len(all_candidates):
        candidates = cand_to_test(all_candidates, d_1, processed, tested_max)
    else:
        candidates = all_candidates
    
    if gain == 'size':
        pos_obs = processed
        if online:
            neg_sensors = filter(lambda x: x not in processed.keys(), sensors)
            neg_obs = {x: pres_time for x in neg_sensors}
        else:
            neg_obs = {}

    # index for nodes to tested
    best_sensor = None
    best_g = 0
    # while I do not find a sensor with pos gain I go on anyway
    while len(candidates)>0 or best_sensor == None:
        if len(candidates)>0:
            c = candidates.pop()
        #If all cands have 0 gain I return one at random e
        elif len(all_candidates) == 0:
            return random.choice(filter(lambda x: x not in sensors, range(len(d_1))))
        else:
            c = random.choice(all_candidates)
            all_candidates.remove(c)
        
        if gain == 'size':
            g = eg.size_gain(d, paths_pred, pres_time, pos_obs, neg_obs,
                    cand_sources, c, approx, noise=noise, tol_c=tol_c)
            # DRS -- gain
        if gain == 'drs':
            g = eg.drs_gain(d, processed, cand_sources, c, pres_time)
            # Update best values. 
            # (equivalent to take one at random)
        if g > best_g:
            best_sensor = c
            best_g = g
    return best_sensor


def cand_to_test(candidates, d_1, processed, max_to_test):
    
    # compute first infected node
    first_inf, _ = ut.first_inf_node(processed)

    # initialize set of nodes to be tested as sensors
    to_test = []
    
    # prepare prob distribution: inversely proportional to the (unweighted)
    # distance from the first (known) infected node
    p = [1.0/d_1[first_inf][c] for c in candidates]
    z = sum(p)
    p = [x/z for x in p]
    # the nodes from which I sample are all candidates
    sample = [c for c in candidates]
    
    while len(to_test) < max_to_test:
        # take one at random 
        to_test.append(rv_discrete(values=(sample, p)).rvs())
        # remove it from sample
        sample.remove(to_test[-1])
        # if there is still sth in sample, redefine the sample and the
        # distribution
        # TODO this can be optimized
        if len(sample) > 0:
            p = [1.0/d_1[first_inf][c] for c in sample]
            z = sum(p)
            p = [x/z for x in p]
        else:
            break
    return to_test
