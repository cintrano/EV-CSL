# This is the main script.
# from arguments import args
import glob
import os

import matplotlib
import numpy as np
import problem
import pandas as pd
import re
# import graph_map_prints as gmp
# python3 363 33550 14 CxA_filter.ssv pertinencia.ssv
# cp.ssv facility_points.ssv substations_points.ssv
# --pop 10 --ga-sel None --ga-cross None --ga-mut None --ga-repl mu_lambda

# Estas son algunas estadísticas para trabajar


distances_energy_stations = np.genfromtxt(f'{os.path.dirname(__file__)}/ExA.ssv', delimiter=' ', skip_header=0)


def read_sol(file):
    return np.genfromtxt(file, delimiter=' ', skip_header=0)


def border(solution):
    return np.max(get_distances(solution))


def read_ilp_solution(file):
    raw = read_sol(file)[:, 2].T
    mynewlist = list(set(raw.astype(int)))
    return [x-1 for x in mynewlist]


def get_distances(solution):
    """Get a numpy vector with the min distances of each customer to their nearest station
    :param solution:
    :return: Numpy vector
    """
    if isinstance(solution, list):
        sol_list = solution
    else:
        sol_list = solution.astype(int).tolist()
    # print(sol_list)
    return np.min(problem.distances[:, sol_list], axis=1)


def get_distances_energy_stations(solution):
    """Get a numpy vector with the min distances of each customer to their nearest station
    :param solution:
    :return: Numpy vector
    """
    if isinstance(solution, list):
        sol_list = solution
    else:
        sol_list = solution.astype(int).tolist()
    return np.max(distances_energy_stations[:, sol_list], axis=0)


def fitness_base(solution):
    return np.dot(get_distances(solution), problem.customer_points[:, 2])  # /567953


def print_table(dataframe, filename):
    """Save a dataframe into a LaTeX file
    :param dataframe: Pandas dataframe
    :param filename: Name of the file
    :return:
    """
    latex_text = dataframe.to_latex(index=False, float_format="%.2f")
    with open(f'{os.path.dirname(__file__)}/{filename}', "w") as writer:
        writer.write(latex_text)


def print_ssv(dataframe, filename):
    with open(f'{os.path.dirname(__file__)}/{filename}', "w") as writer:
        writer.write(dataframe.to_csv(sep=' '))


def print_dict_ssv(data, filename):
    with open(f'{os.path.dirname(__file__)}/{filename}', "w") as writer:
        for i in range(20, 81):
            try:
                writer.write(' '.join(map(str, data[i].tolist())) + '\n')
            except:
                print(f'ERROR: print_dict_ssv: {i}')


def simple_cull(inputPoints, dominates):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints

def dominates(row, candidateRow):
    return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)

import random
#inputPoints = [[random.randint(70,100) for i in range(3)] for j in range(500)]
inputPoints = [[random.randint(70,100) for i in range(2)] for j in range(500)]
paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)#, projection='2d')
dp = np.array(list(dominatedPoints))
pp = np.array(list(paretoPoints))
print(pp.shape,dp.shape)
#ax.scatter(dp[:,0],dp[:,1],dp[:,2])
#ax.scatter(pp[:,0],pp[:,1],pp[:,2],color='red')
ax.scatter(dp[:,0],dp[:,1])
ax.scatter(pp[:,0],pp[:,1],color='red')

import matplotlib.cm as cm

colors = matplotlib.cm.rainbow(np.linspace(0, 1, 30))
'''
for i in range(30):
    path = '/home/cintrano/PycharmProjects/out-NSGAII/'
    file = f'{i}-fits.ssv'
    ex = np.genfromtxt(path + "/" + file, delimiter=' ', skip_header=0)
    ax.scatter(ex[:, 0], ex[:, 1], color=colors[i], marker='+')
for i in range(30):
    path = '/home/cintrano/PycharmProjects/out-NSGAII-500iteraciones/'
    file = f'{i}-fits.ssv'
    ex = np.genfromtxt(path + "/" + file, delimiter=' ', skip_header=0)
    ax.scatter(ex[:, 0], ex[:, 1], color=colors[i], marker='o')
'''
#plt.colorbar()
path_greedy = '/home/cintrano/PycharmProjects/location_problem_new/outputs/'
file_greedy = 'fits-greedy.ssv'
ex_greedy = np.genfromtxt(path_greedy + file_greedy, delimiter=',', skip_header=1)
ax.scatter(ex_greedy[:, 1], ex_greedy[:, 2], color='black', marker='+')


import matplotlib.tri as mtri
#triang = mtri.Triangulation(pp[:,0],pp[:,1])
#ax.plot_trisurf(triang,pp[:,2],color='red')
#ax.plot_trisurf(triang,color='red')
plt.show()


'''
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
                sol = read_sol(file)[1:]
                f = fitness_base(sol)
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
    f = fitness_base([x-1 for x in mynewlist])
    df = df.append({'p': 50, 'ind': 0, 'fitness': f, 'algo': 'CPLEX'}, ignore_index=True)
    print(f'P: CPl, Distance: {f}')
    print(df.head().to_string())
    print(df.groupby(['p', 'algo'])['fitness'].describe().to_string())

    print("=== DISTANCES ===")
    dist = pd.DataFrame(columns=['P', 'Ind', 'Fitness', 'Algo', 'Distance'])
    for d in get_distances([x-1 for x in mynewlist]):
        dist = dist.append({'P': 50, 'Ind': 0, 'Fitness': f, 'Algo': 'CPLEX', 'Distance': d}, ignore_index=True)

    print(dist.groupby(['P', 'Algo'])['Distance'].describe().to_string())
    print_table(dist.groupby(['P', 'Algo'])['Distance'].describe(), 'statistics.tex')

    print("=== MAPS ===")
    dist = pd.DataFrame(columns=['P', 'Ind', 'Fitness', 'Algo', 'Distance'])
    dict_pxc = {}
    dict_pxe = {}
    for i in range(20, 60):
        try:
            sol = read_ilp_solution(f'/home/cintrano/PycharmProjects/location_problems/data/Soluciones/FIsolMs{i}.txt')
            dict_pxc[i] = get_distances(sol)
            dict_pxe[i] = get_distances_energy_stations(sol)
            # Print the stations
            gmp.main(sol, f'FIsolMs{i}_map.pdf')
        except:
            print(f'ERROR: reading: {i}')
            pass
    for i in range(60, 81):
        try:
            sol = read_ilp_solution(f'/home/cintrano/PycharmProjects/location_problems/data/Soluciones/solMs{i}.txt')
            dict_pxc[i] = get_distances(sol)
            dict_pxe[i] = get_distances_energy_stations(sol)
            # Print the stations
            gmp.main(sol, f'FIsolMs{i}_map.pdf')
        except:
            print(f'ERROR: reading: {i}')
            pass
    for i in range(20, 81):
        try:
            sol = read_ilp_solution(f'/home/cintrano/PycharmProjects/location_problems/data/Soluciones-No-rest/FIsolMs{i}.txt')
            dict_pxc[i] = get_distances(sol)
            dict_pxe[i] = get_distances_energy_stations(sol)
            # Print the stations
            gmp.main(sol, f'No-rest-FIsolMs{i}_map.pdf')
        except:
            print(f'ERROR: reading: {i}')
            pass
    # Station x Clients
    dist_pxc = pd.DataFrame(dict_pxc)
    dist_pxc = dist_pxc.transpose()
    print_ssv(dist_pxc, 'dist_pxc.ssv')
    # Stations x Electric substations
    print_dict_ssv(dict_pxe, 'dist_pxe.ssv')
    # Statistics
    print(dist.groupby(['P', 'Algo'])['Distance'].describe().to_string())
    print_table(dist.groupby(['P', 'Algo'])['Distance'].describe(), 'statistics.tex')

    print("== END ==")
'''