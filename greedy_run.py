# This is the main script.
import csv
import random
import datetime

import numpy as np

from arguments import args
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ga_mo
import problem
# python3 363 33550 14 CxA_filter.ssv pertinencia.ssv
# cp.ssv facility_points.ssv substations_points.ssv
# --pop 10 --ga-sel None --ga-cross None --ga-mut None --ga-repl mu_lambda

FITNESS_FILE = "fits_greedy.ssv"
POPULATION_FILE = "pop_greedy.ssv"


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
    print(args)
    start = datetime.datetime.now()
    points = np.genfromtxt('greedy_data.ssv', delimiter=' ', skip_header=0)
    dict_sols = []
    zero = np.zeros(args.L)
    count = 0
    for i in range(args.L * 2):
        if i % 1000 == 0:
            print('.', end='')
        sol = np.copy(zero)
        type_station = 1
        location = int(points[i])
        if location >= args.L:
            type_station = 2
            location = location - args.L
        if sol[location] == 0:
            sol[location] = type_station
            if problem.satisfiability(sol):
                count = count + 1
                zero[location] = type_station
                f1, f2 = problem.fitness_mo(sol)
                dict_sols.append({'p': count, 'solution': sol.astype(int).tolist(), 'fitness1': f1, 'fitness2': f2})
    end = datetime.datetime.now()
    duration = end - start
    print(start)
    print(end)
    print(start, end, duration.seconds)
    with open(args.out_path + "/" + 'fits-greedy.ssv', 'w') as out:
        #csv_out = csv.writer(out, delimiter=" ")
        #csv_out.writerows(data)
        dict_writer = csv.DictWriter(out, fieldnames=["p", "fitness1", "fitness2"], extrasaction='ignore')
        dict_writer.writeheader()
        dict_writer.writerows(dict_sols)
    with open(args.out_path + "/" + 'sols-greedy.ssv', 'w') as out:
        #csv_out = csv.writer(out, delimiter=" ")
        #csv_out.writerows(data)
        dict_writer = csv.DictWriter(out, fieldnames=["p", "solution"], extrasaction='ignore')
        dict_writer.writeheader()
        dict_writer.writerows(dict_sols)
