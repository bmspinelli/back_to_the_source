import random
import numpy.random as nprand

#random.seed(100)
#nprand.seed(100)

def spread(graph, noise, delay_type='uniform', gamma_k=1, source=None):
    """Simulation of a diffusion process.

    - graph: networkx graph
    - noise: transmissions are uniform between weight*(1 - noise) and weight*(1 + noise)
      (default for weight is 1.0)
    - source: source node
    
    Output: dictionary mapping nodes to their infection time. Something like:
		{ node1: infection_time1, node2: infection_time2, ... }.

    """


    if source != None and source not in range(len(graph)):
        raise ValueError('Not a valid source') 

    if source == None:
        source = random.choice(range(len(graph)))

    infected    = {source: 0}  # Infection time per node
    processing  = {source: 0}  # Infected nodes to process and their infection
                               # times

    while processing:
        node, time = sorted(processing.items(), key=lambda x: x[1], reverse=True).pop()
        for neighbour in graph.neighbors(node):
            try:
                infection_time = time + edge_delay(graph[node][neighbour]['weight'], 
                        noise, delay_type=delay_type, gamma_k=gamma_k)
            except KeyError:
                infection_time = time + edge_delay(1.0, noise,
                        delay_type=delay_type, gamma_k=gamma_k)
            if neighbour not in infected or infected[neighbour] > infection_time:
                infected[neighbour] = infection_time
                processing[neighbour] = infection_time
        del processing[node]
    return infected, source


def edge_delay(weight, noise, delay_type='uniform', gamma_k=1):
    """Uniform r.v. between weight*(1 - noise) and weight*(1 + noise)"""

    assert noise >= 0 and noise < 1.0
    
    if noise == 0:
        return weight
    else:
        if delay_type == 'uniform':
            return nprand.uniform(low=weight*(1-noise), high=weight*(1+noise))
        if delay_type == 'gamma':
            if weight != 1:
                raise ValueError('If Gamma delays, weights should be 1!')
            return nprand.gamma(gamma_k, scale=1.0/gamma_k)
        else:
            raise ValueError('Unknown distribution!')
