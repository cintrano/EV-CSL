# This is the main script.
import csv
import random
import datetime

import deap.tools
import numpy as np
from pygmo.core import hypervolume

from arguments import args
import ga_mo
import problem
# python3 363 33550 14 CxA_filter.ssv ExA.ssv cp.ssv facility_points.ssv substations_points_EkW.ssv --pop 20
# --ga-sel None --ga-sel TOURNAMENT2 --ga-cross 2POINT --ga-mut UNIFORM --ga-repl mu_lambda --pop 20 --iter 500
# --input ./inputs --output ./outputs

FITNESS_FILE = "fits.ssv"
POPULATION_FILE = "pop.ssv"


# Press the green button in the gutter to run the script.
def write_fitness_values(data, filename=FITNESS_FILE):
    with open(args.out_path + "/" + filename, 'w') as out:
        csv_out = csv.writer(out, delimiter=" ")
        csv_out.writerows(data)


def write_population(data, filename=POPULATION_FILE):
    with open(args.out_path + "/" + filename, 'w') as out:
        csv_out = csv.writer(out, delimiter=" ")
        csv_out.writerows(data)


if __name__ == '__main__':
    if args.seed:
        print(args)
        start = datetime.datetime.now()
        np.random.seed(seed=13 + args.seed)
        random.seed(a=13 + args.seed)
        fits, pop = [], []
        if args.algorithm == "NSGA2":
            print("Running NSGA2")
            fits, pop = ga_mo.run_nsga2(problem.fitness_mo, args.POP_SIZE, args.sel_mode, args.cross_mode, args.mut_mode,
                              args.repl_mode, cross_prob=args.prob_cross, mut_prob=args.prob_mut, max_iter=args.iter)
        if args.algorithm == "SPEA2":
            fits, pop = ga_mo.run_spea2(problem.fitness_mo, args.POP_SIZE, args.sel_mode, args.cross_mode, args.mut_mode,
                              args.repl_mode, cross_prob=args.prob_cross, mut_prob=args.prob_mut, max_iter=args.iter)
        # Execution time
        end = datetime.datetime.now()
        duration = end - start
        print('Running Time:', start, end, duration.seconds)

        # Writting files
        if args.write_files:
            write_fitness_values(fits, str(args.seed) + '-' + FITNESS_FILE)
            write_population(pop, str(args.seed) + '-' + POPULATION_FILE)
        # Calculating the HV
        # change sign of first objective to convert it into a minimization
        f = [[(-1.0 * x), y] for x, y in fits]
        nadir_x, nadir_y = 0, 100000000  # TODO: Change to a more fixed nadir point
        nadir_point = [nadir_x, nadir_y]
        hv = hypervolume(f)  #, [nadir_x, nadir_y])
        hv_value = hv.compute(nadir_point)
        print(hv_value)
    else:
        '''
        print(args)

        for i in range(30):
            print(f"=== EXECUTION {i} === ")
            start = datetime.datetime.now()
            np.random.seed(seed=13+i)
            random.seed(a=13+i)
            fits, pop = ga_mo.run(problem.fitness_mo, args.POP_SIZE, args.sel_mode, args.cross_mode, args.mut_mode,
                                  args.repl_mode, cross_prob=0.7, mut_prob=0.2, max_iter=args.iter)

            end = datetime.datetime.now()
            duration = end-start
            print(start)
            print(end)
            print(start, end, duration.seconds)
            if args.write_files:
                write_fitness_values(fits, str(i) + '-' + FITNESS_FILE )
                write_population(pop, str(i) + '-' + POPULATION_FILE )
        '''