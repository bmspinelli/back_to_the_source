import updates as up

def static(graph, d, sensors, infection_times, noise=0, tol_c=0, 
        real_source=None, online=True):

    sorted_times, sorted_sensors = zip(*sorted(zip([infection_times[s] for s in
            sensors], sensors)))

    min_time = sorted_times[0]
    max_time = max(infection_times.values())
    
    if online:
        # detection starts when the first sensor gets infected
        pres_time = min_time
    else: 
        # detection starts once all nodes are infected
        pres_time = max_time

    cand_sources = graph.nodes()

    if online:
        cand_sources, stop = up.update_cand_sources(d, sorted_sensors[0],
                min_time, {}, cand_sources, pres_time, 
                sensors, noise=noise, tol_c=tol_c, online=online)
    else:
        pass 

    processed = {sorted_sensors[0]: min_time}
    to_process = sorted_sensors[1:]
    time_to_process = sorted_times[1:]

    seq_cand_sources = [len(cand_sources)]
    
    infected = [len(filter(lambda x: infection_times[x] <= pres_time,
            graph.nodes()))]

    success = [int(real_source in cand_sources)]
    
    # While did not converge to only 1 condidate source
    stop = False
    i = 0
    while len(cand_sources) > 1 and not stop and i <= len(to_process)-1:
        
        if online:
            # update on every new infection: if deterministic, we can have
            # mutiple updates at the same time
            pres_time = time_to_process[i]
        else:
            pass

        cand_sources, stop = up.update_cand_sources(d, to_process[i],
                time_to_process[i], processed, cand_sources, pres_time, 
                sensors, noise=noise, tol_c=tol_c, online=online)

        seq_cand_sources, infected, success = up.update_algo_state(graph, 
                seq_cand_sources, cand_sources, success, infected,
                infection_times, pres_time, real_source) 
        
        processed[to_process[i]] = time_to_process[i]
        
        i = i + 1
    
    time = pres_time - min(infection_times.values())
    
    

    return cand_sources, seq_cand_sources, time, infected, success
