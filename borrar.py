# This is the main script.
import csv
from arguments import args
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ga_mo
import problem
import glob
import os
import numpy as np
import problem
import pandas as pd
import re
# python3 363 33550 14 CxA_filter.ssv pertinencia.ssv
# cp.ssv facility_points.ssv substations_points.ssv
# --pop 10 --ga-sel None --ga-cross None --ga-mut None --ga-repl mu_lambda

FITNESS_FILE = "fits.ssv"
POPULATION_FILE = "pop.ssv"


# Press the green button in the gutter to run the script.
def write_fitness_values(data):
    with open(args.out_path + "/" + FITNESS_FILE, 'w') as out:
        csv_out = csv.writer(out, delimiter=" ")
        csv_out.writerows(data)


def write_population(data):
    with open(args.out_path + "/" + POPULATION_FILE, 'w') as out:
        csv_out = csv.writer(out, delimiter=" ")
        csv_out.writerows(data)


def read_sol(file):
    return np.genfromtxt(file, delimiter=' ', skip_header=0)


def border(solution):
    sol = solution.astype(int).tolist()
    mins = np.min(problem.distances[:, sol], axis=1)
    return np.max(mins)


def border_list(solution):
    sol = solution
    mins = np.min(problem.distances[:, sol], axis=1)
    return np.max(mins)


def fitness_base(solution):
    #sol = [i for i, e in enumerate(solution) if e != 0]
    sol = solution.astype(int).tolist()
    mins = np.min(problem.distances[:, sol], axis=1)
    return np.dot(mins, problem.customer_points[:, 2])#/567953
    #return np.sum(mins ** problem.customer_points[:, 2])/500000


def fitness_base_list(solution):
    #sol = [i for i, e in enumerate(solution) if e != 0]
    sol = solution
    mins = np.min(problem.distances[:, sol], axis=1)
    return np.dot(mins, problem.customer_points[:, 2])#/567953
    #return np.sum(mins ** problem.customer_points[:, 2])/500000


def fitness_base_list_different(solution):
    """Como el anterior pero con otro calculo
    :param solution:
    :return:
    """
    #sol = [i for i, e in enumerate(solution) if e != 0]
    sol = solution
    mins = np.min(problem.distances[:, sol], axis=1)
    out = np.multiply(mins, problem.customer_points[:, 2])
    return np.sum(out)


if __name__ == '__main__':
    print(args)
    path = '/media/cintrano/6334-3733/electric-car/experiments/v2/'
    choices = ['irace-1/', 'irace-2/']
    ps = ['10/', '20/', '30/', '40/', '45/', '50/']
    df = pd.DataFrame(columns=['p', 'ind', 'fitness', 'algo'])

    print("\n\n*** FITNESS ***")
    for p in ps:
        fitness = []
        i = 0
        for c in choices:
            os.chdir(path+p+c)
            for file in glob.glob("*best_*"):
                #print("...............")
                #print(file)
                sol = read_sol(file)[1:]
                #print(sol)
                f = fitness_base(sol)
                #print(f)
                fitness.append(f)
                algo = re.search(r"(?<=_).*?(?=_)", file).group(0)
                df = df.append({'p': p, 'ind': i, 'fitness': f, 'algo': algo}, ignore_index=True)
                i += 1
        print(f'P: {p}, Distance: {min(fitness)}')
    raw = [260,260,260,562,562,562,562,562,562,562,562,562,562,754,754,754,754,754,754,754,1554,1554,1554,1554,2326,2326,2326,2326,2326,2326,2326,2326,3075,3075,3075,3210,3210,3210,3210,3210,3210,3210,3800,3800,3800,3800,3800,3800,3800,3800,3800,3800,3800,4039,4039,4039,4039,4039,4039,4039,4039,4039,4039,4039,4039,6203,6203,6203,6203,6203,6203,6203,6503,6503,6503,6503,6503,6503,6503,8103,8103,8103,8103,8103,8103,8103,8103,8103,8103,8169,8169,8169,8169,8169,8169,8169,8169,8169,8169,8169,8169,8169,8638,8638,8638,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,8946,9412,9412,9412,9412,9412,9412,9412,9412,10254,10254,10254,10254,10820,10820,10820,10820,10820,10820,10820,10820,10870,10870,10870,10870,10870,10870,11863,11863,11863,11863,13142,13142,13142,13142,13142,13142,13642,13642,13642,13642,13642,13642,14396,14396,14396,14396,14396,14396,14396,14396,16033,16033,16033,17688,17688,18426,18426,18426,18426,18426,19351,19351,19351,19351,19389,19389,19389,20744,20744,20744,20883,20883,20883,20883,20883,20883,20883,20883,20883,21734,21734,21734,21734,21734,21734,21870,21870,21870,21870,21870,21870,21870,21870,21870,21870,21870,21870,21870,21870,21870,21870,21870,21870,22113,22113,22113,22113,22113,22113,22271,22271,22470,22470,22470,22470,22470,23507,23507,23507,23507,23507,23507,23507,23507,23507,23507,23525,23525,23525,23525,23525,23525,23525,23525,23525,23525,23588,23588,23588,23588,23588,23588,23588,23588,24635,24635,24635,24635,24635,24635,24635,24635,24635,24635,24906,24906,24906,24906,24906,24906,26473,26473,26473,26473,26473,28295,28295,28295,28295,28295,28295,28295,28390,28390,28390,28824,28824,28824,28824,28824,28824,28865,28865,28865,28865,28865,31430,31430,31430,31430,31430,31430,31430,31430,31430,31430,31430,31482,31482,31482,31482,31482,31482,31482,31482,31482,31482,31482,33019,33019,33019,33019,33019,33019,33241,33241,33241,33241,33241,33241,33277,33277,33277,33277,33277,33277,33277,33277,33277,33277]
    myset = set(raw)
    mynewlist = list(myset)
    # f = fitness_base_list(mynewlist)
    # print(f'P: CPl, Distance: {f}')
    # print(len(mynewlist))
    f = fitness_base_list([x-1 for x in mynewlist])
    df = df.append({'p': p, 'ind': 0, 'fitness': f, 'algo': 'CPLEX'}, ignore_index=True)
    print(f'P: CPl, Distance: {f}')
    print(df.head().to_string())
    print(df.groupby(['p', 'algo'])['fitness'].describe().to_string())
    print("\n\n*** BORDERS ***")
    for p in ps:
        fitness = []
        for c in choices:
            os.chdir(path+p+c)
            for file in glob.glob("*best_*"):
                #print("...............")
                #print(file)
                sol = read_sol(file)[1:]
                #print(sol)
                f = border(sol)
                #print(f)
                fitness.append(f)
        print(f'P: {p}, Distance: Max={max(fitness)} Min={min(fitness)}')
    print("== END ==")

