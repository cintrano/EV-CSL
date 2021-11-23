from datetime import datetime

import numpy as np

import problem
from arguments import args


def run_by_time(fitness, max_time):
    fits = []
    sols = []
    start = datetime.now()
    end = datetime.now()
    duration = end - start
    while duration.seconds < max_time:
        ind = np.empty(args.L, dtype=int)
        new_ind = problem.constructive_solution_by_zones(ind)[0]  # [0] for unwrap the individual
        print(new_ind)
        f1, f2 = fitness(new_ind)
        fits.append([f1, f2])
        sols.append(new_ind)
        # Update time
        end = datetime.now()
        duration = end - start
    return fits, sols


def run(fitness, max_time, max_iter):
    if max_time and max_time != 0:
        return run_by_time(fitness, max_time)
    else:  # Run by iters
        pass
