import os

import numpy as np
import pandas as pd
import pygmo
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def simple_cull(input_points, dominates):
    pareto_points = set()
    candidate_row_nr = 0
    dominated_points = set()
    while True:
        candidate_row = input_points[candidate_row_nr]
        input_points.remove(candidate_row)
        rowNr = 0
        non_dominated = True
        while len(input_points) != 0 and rowNr < len(input_points):
            row = input_points[rowNr]
            if dominates(candidate_row, row):
                # If it is worse on all features remove the row from the array
                input_points.remove(row)
                dominated_points.add(tuple(row))
            elif dominates(row, candidate_row):
                non_dominated = False
                dominated_points.add(tuple(candidate_row))
                rowNr += 1
            else:
                rowNr += 1

        if non_dominated:
            # add the non-dominated point to the Pareto frontier
            pareto_points.add(tuple(candidate_row))

        if len(input_points) == 0:
            break
    return pareto_points, dominated_points


def dominates(row, candidate_row):
    """Always minimize
    :param row:
    :param candidate_row:
    :return:
    """
    #return sum([row[x] >= candidate_row[x] for x in range(len(row))]) == len(row)
    return sum([row[0] <= candidate_row[0], row[1] <= candidate_row[1]]) == len(row)


def read_sol(file):
    return np.genfromtxt(file, delimiter=' ', skip_header=0)


def read_ilp_solution(file):
    raw = read_sol(file)[:, 2].T
    mynewlist = list(set(raw.astype(int)))
    return [x-1 for x in mynewlist]




def get_distances_energy_stations(solution):
    """Get a numpy vector with the min distances of each customer to their nearest station
    :param solution:
    :return: Numpy vector
    """
    if isinstance(solution, list):
        sol_list = solution
    else:
        sol_list = solution.astype(int).tolist()

    distances_energy_stations = np.genfromtxt(f'{os.path.dirname(__file__)}/ExA.ssv', delimiter=' ', skip_header=0)
    return np.max(distances_energy_stations[:, sol_list], axis=0)


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


colors = plt.cm.rainbow(np.linspace(0, 1, 30))


def load_data(path, lower, upper, plot, marker, load_solutions=False):
    print('loading solutions from', path)
    file = f'{lower}-fits.ssv'
    ex = np.genfromtxt(path + "/" + file, delimiter=' ', skip_header=0)
    fit = ex
    sols = None
    if load_solutions:
        file_pop = f'{lower}-pop.ssv'
        ex_pop = np.genfromtxt(path + "/" + file_pop, delimiter=' ', skip_header=0)
        sols = ex_pop
    for i in range(lower+1, upper):
        file = f'{i}-fits.ssv'
        file_pop = f'{i}-pop.ssv'
        ex = np.genfromtxt(path + "/" + file, delimiter=' ', skip_header=0)
        if load_solutions:
            ex_pop = np.genfromtxt(path + "/" + file_pop, delimiter=' ', skip_header=0)
        if isinstance(plot, list):
            for p in plot:
                p.scatter(ex[:, 0], ex[:, 1], color=colors[i % 30], marker=marker)
        else:
            plot.scatter(ex[:, 0], ex[:, 1], color=colors[i % 30], marker=marker)
        fit = np.vstack((fit, ex))
        if load_solutions:
            sols = np.vstack((sols, ex_pop))
    return fit, sols


def prepare_dataframe(data, algo, label, marker):
    """Create a dataframe from the fitness data
    :param data: fitness value of the population
    :param algo: algorithm
    :param label: experiment label
    :param marker: marker to plot
    :return: the DataFrame
    """
    df = pd.DataFrame(data, columns=['f1', 'f2'])
    df['algorithm'] = algo
    df['label'] = label
    df['marker'] = marker
    df['color'] = 'b'
    return df


def print_pareto_hv(data, nadir_point, ax, label, relative=1., color='red', marker='o'):
    """
    :param data: Numpy array
    :param nadir_point:
    :param ax:
    :param label:
    :param color:
    :param marker:
    :return:
    """
    d = data.copy()
    d[:, 0] *= -1  # to minimize the first objective and compute the hv
    inputPoints = d.tolist()
    paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)
    dp = np.array(list(dominatedPoints))
    pp = np.array(list(paretoPoints))
    # print(pp.shape,dp.shape)
    hv = pygmo.hypervolume(pp)
    print(label, '&', hv.compute(nadir_point)/relative)
    pp[:, 0] *= -1  # undo the change
    ax.scatter(pp[:, 0], pp[:, 1], color=color, marker=marker)
    return pp


def compute_hv(data, nadir_point):
    d = data.copy()
    d[:, 0] *= -1  # to minimize the first objective and compute the hv
    inputPoints = d.tolist()
    paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)
    dp = np.array(list(dominatedPoints))
    pp = np.array(list(paretoPoints))
    # print(pp.shape,dp.shape)
    hv = pygmo.hypervolume(pp)
    return hv.compute(nadir_point), pp


def print_legend(ax, df, color_exist=False):
    legend_elements = []
    for a in df['algorithm'].unique().tolist():
        for l in df['label'].unique().tolist():
            q = f'algorithm == "{a}" and label == "{l}"'
            marker = df.query(q).iloc[[0]]['marker'].values[0]
            if color_exist:
                color = df.query(q).iloc[[0]]['color'].values[0]
                marker_color = color
            else:
                color = 'w'
                marker_color = 'black'
            legend_elements.append(Line2D([0], [0], marker=marker, color=color, markerfacecolor=marker_color, label=f'{a}'))
    ax.legend(handles=legend_elements) #, loc='center')
