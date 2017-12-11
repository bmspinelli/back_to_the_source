#!/usr/bin/env python
#Brunella Marta Spinelli (brunella.spinelli@epfl.ch)
"""Module for running variable neighborhood search algorithm for the
k-median optimization problem.

"""

import numpy as np
import random
   

def k_median_cost(d, observers):
    """Takes a distance matrix (in dict format) and a set of k nodes and return the k-medians
    value and the clusters assignment
    
    """
    clusterid = {i: i for i in range(len(d))}
    sum_dist = 0

    for i in range(len(d)):
        min_dist = np.inf 
        for j in observers:
            if d[i][j] < min_dist:
                min_dist = d[i][j]
                clusterid[i] = j
        sum_dist = sum_dist + min_dist
    return clusterid, sum_dist/float(len(d))


def local_change(d, k, max_iter):
    """Max_iter times k random nodes are selected as observers and then moved 
    around until there is no improvement in the solution
    
    """
    min_cost = np.inf
    n = len(d)
    for it in xrange(max_iter):
        actual_cost = np.inf
        obs = range(len(d))
        random.shuffle(obs)
        non_obs = obs[k:]
        obs = obs[0:k]
        assert len(obs) == k
        gain = True
        while gain:
            gain = False
            for i in xrange(k):
                for j in xrange(n-k):
                    obs[i], non_obs[j] = non_obs[j], obs[i]
                    (assignment, c) = k_median_cost(d, obs) 
                    if c < actual_cost:
                        gain = True
                        actual_cost = c
                        best_out, best_in = i, j
                    obs[i], non_obs[j] = non_obs[j], obs[i]
            obs[best_out], non_obs[best_in] = non_obs[best_in], obs[best_out]
        if actual_cost < min_cost:
            observers = obs
            min_cost = actual_cost
            print 'cost:', min_cost 
    return observers, min_cost


def greedy(d, k):
    """At every iteration an observer is added
    such that it minimizes the cost at every step   

    """
    n = len(d)
    observers = list()    
    for i in xrange(k): #for k times
        print i
        min_score = np.inf 
        candidate_to_add = None
        for candidate in range(len(d)):
            if candidate not in observers: #if it has not been chosen yet
                observers_ext = list(observers)
                observers_ext.extend([candidate]) #temporary placement
                tmp, score = k_median_cost(d, observers_ext)
                if score < min_score:
                    min_score = score
                    candidate_to_add = candidate
        print min_score
        assert candidate_to_add != None
        observers.append(candidate_to_add)
    return observers, min_score
