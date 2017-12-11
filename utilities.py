import networkx as nx
import numpy as np


def alarm(infection_times, static):
    """Find the first detected infection time together with the static sensors that
    detects it
    
    If noise = 0 possibly multiple static sensors infected

    """

 
    pos_obs = {}

    min_time = min([infection_times[s] for s in static])
    
    for s in static:
        if infection_times[s] == min_time:
            pos_obs[s] = min_time

    return pos_obs, min_time


def initialize_candidates(g, first_infected, static):
    
    # remove non-infected static and find graph components: all cand sources
    # are in the same component of the first infected

    g_copy = g.copy()
    non_inf_static = filter(lambda x: x not in first_infected, static)
    g_copy.remove_nodes_from(non_inf_static)

    init_candidates = []
    
    comp = {n: nx.node_connected_component(g_copy, n) for n in first_infected}

    for n in g:
        reach = [n in comp[f] for f in first_infected]
        if len(reach) == sum(reach):
            init_candidates.append(n)

    return init_candidates


#def resolvab_improvement(d, s_1, cand_sources, c):

#    values = set()
    
#    for n in cand_sources:
        # Multiply for 10**8 to avoid precision errors
#        values.add((10**8)*d[s_1][n] - d[c][n])
    
#    return len(values)



def preprocess(graph):
    """Computes distances and shortest paths in the graph"""

    path_lengths, paths = {}, {}

    for n in graph:
        path_lengths[n], paths[n] = nx.single_source_dijkstra(graph, n)

    return path_lengths, paths


def tol(hops, noise, tol_c=1):
    "Tolerance function, if noise is 0 it is 0"
    
    return tol_c * hops * noise


def create_matrix_sentinel(sentinel, n):
    PATH = '/dfs/ephemeral/storage/spinelli/online_inf/lookup/output_mat'
    A = np.zeros([n, n])
    for i in range(n):
        B = np.load('%s_%d.npy' %(PATH, i))
        A[i] = B[sentinel]
    return A


def create_matrix_sentinel_bridges(sentinel, n, noise):
    PATH = '/dfs/ephemeral/storage/spinelli/online_inf/lookup/output_bridges'
    A = np.zeros([n, n])
    for i in range(n):
        print i
        B = np.load('%s_%d_%d.npy' %(PATH, i, int(noise*10)))
        A[i] = B[sentinel]
    print "finished preprocessing"
    return A


def set_dist(x, s, d):

    return min([d[x][y] for y in s])


def first_inf_node(processed):
    min_time, first_inf = min(zip(processed.values(), processed.keys()))
    return first_inf, min_time
