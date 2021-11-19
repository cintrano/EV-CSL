import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import graph_map_prints as gmp

from my_utilities import simple_cull, dominates
from matplotlib.lines import Line2D
import pygmo
from pymoo.factory import get_performance_indicator

distances_energy_stations = np.genfromtxt(f'{os.path.dirname(__file__)}/ExA.ssv', delimiter=' ', skip_header=0)


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


fig, ((ax, ax1, ax2), (ax, ax3, ax4)) = plt.subplots(2, 3, figsize=(15, 5))


colors = plt.cm.rainbow(np.linspace(0, 1, 30))

'''
file = '/home/cintrano/_current/abg_last_test/NSGA2/0.5-0.1-0.4/'
ex = np.genfromtxt(file + '1-fits.ssv', delimiter=' ', skip_header=0)
ex_sol = np.genfromtxt(file + '1-pop.ssv', delimiter=' ', skip_header=0)
ax.scatter(ex[:, 0], ex[:, 1], color='red', marker='o')
for i, e in enumerate(ex):
    ax.annotate(str(ex[i,0]) + " " + str(ex[i,1]), (ex[i,0], ex[i,1]))
    if ex[i, 1] == 1171641:
        sol = [i for i, e in enumerate(ex_sol[i,:]) if e != 0]
        gmp.main(sol, '0.5-0.1-0.4_map.pdf', color_sol='red')

file = '/home/cintrano/_current/abg_last_test/NSGA2/0.5-0.2-0.3/'
ex = np.genfromtxt(file + '2-fits.ssv', delimiter=' ', skip_header=0)
ex_sol = np.genfromtxt(file + '2-pop.ssv', delimiter=' ', skip_header=0)
ax.scatter(ex[:, 0], ex[:, 1], color='blue', marker='o')
for i, e in enumerate(ex):
    ax.annotate(str(ex[i,1]), (ex[i,0], ex[i,1]))
    if ex[i, 1] == 1173189:
        sol = [i for i, e in enumerate(ex_sol[i,:]) if e != 0]
        gmp.main(sol, '0.5-0.2-0.3_map.pdf', color_sol='blue')

file = '/home/cintrano/_current/abg_last_test/NSGA2/0.5-0.3-0.2/'
ex = np.genfromtxt(file + '3-fits.ssv', delimiter=' ', skip_header=0)
ex_sol = np.genfromtxt(file + '3-pop.ssv', delimiter=' ', skip_header=0)
ax.scatter(ex[:, 0], ex[:, 1], color='green', marker='o')
for i, e in enumerate(ex):
    ax.annotate(str(ex[i,1]), (ex[i,0], ex[i,1]))
    if ex[i, 1] == 1166480:
        sol = [i for i, e in enumerate(ex_sol[i,:]) if e != 0]
        gmp.main(sol, '0.5-0.3-0.2_map.pdf', color_sol='green')

file = '/home/cintrano/_current/abg_last_test/NSGA2/0.5-0.4-0.1/10-fits.ssv'
ex = np.genfromtxt(file, delimiter=' ', skip_header=0)
ax.scatter(ex[:, 0], ex[:, 1], color='yellow', marker='o')
for i, e in enumerate(ex):
    ax.annotate(str(ex[i,1]), (ex[i,0], ex[i,1]))

file = '/home/cintrano/_current/abg_last_test/NSGA2/0.5-0.25-0.25/'
ex = np.genfromtxt(file + '4-fits.ssv', delimiter=' ', skip_header=0)
ex_sol = np.genfromtxt(file + '4-pop.ssv', delimiter=' ', skip_header=0)
ax.scatter(ex[:, 0], ex[:, 1], color='black', marker='o')
for i, e in enumerate(ex):
    ax.annotate(str(ex[i,1]), (ex[i,0], ex[i,1]))
    if ex[i, 1] == 1164445:
        sol = [i for i, e in enumerate(ex_sol[i,:]) if e != 0]
        gmp.main(sol, '0.5-0.25-0.25_map.pdf', color_sol='black')

plt.xlabel('Quality of service')
plt.ylabel('Cost of installation')

plt.show()
'''


def load_data(path, lower, upper, plot, marker):
    print('loading solutions from', path)
    file = f'{lower}-fits.ssv'
    file_pop = f'{lower}-pop.ssv'
    ex = np.genfromtxt(path + "/" + file, delimiter=' ', skip_header=0)
    ex_pop = np.genfromtxt(path + "/" + file_pop, delimiter=' ', skip_header=0)
    fit = ex
    sols = ex_pop
    for i in range(lower+1, upper):
        file = f'{i}-fits.ssv'
        file_pop = f'{i}-pop.ssv'
        ex = np.genfromtxt(path + "/" + file, delimiter=' ', skip_header=0)
        ex_pop = np.genfromtxt(path + "/" + file_pop, delimiter=' ', skip_header=0)
        if isinstance(plot, list):
            for p in plot:
                p.scatter(ex[:, 0], ex[:, 1], color=colors[i % 30], marker=marker)
        else:
            plot.scatter(ex[:, 0], ex[:, 1], color=colors[i % 30], marker=marker)
        fit = np.vstack((fit, ex))
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


# Load data

df = pd.DataFrame(columns=['algorithm', 'label', 'f1', 'f2', 'marker'])

path_experiments = '/home/cintrano/_current/EV-CSL-results/'

label = 'random-generation'
nsga2_data, nsga2_sols = load_data(path_experiments + label + '/' 'NSGA2', 20, 30, [ax, ax1], marker='o')
df = df.append(prepare_dataframe(nsga2_data, 'NSGA2', label, 'o'))

spea2_data, spea2_sols = load_data(path_experiments + label + '/' 'SPEA2', 10, 20, [ax, ax1], marker='+')
df = df.append(prepare_dataframe(spea2_data, 'SPEA2', label, '+'))

label = 'constructive-generation'
nsga2_generation_data, nsga2_generation_sols = load_data(path_experiments + label + '/' 'NSGA2', 20, 30, [ax, ax2], marker='s')
df = df.append(prepare_dataframe(nsga2_generation_data, 'NSGA2', label, 's'))
spea2_generation_data, spea2_generation_sols = load_data(path_experiments + label + '/' 'SPEA2', 10, 20, [ax, ax2], marker='^')
df = df.append(prepare_dataframe(spea2_generation_data, 'SPEA2', label, '^'))


label = 'random-generation-mut7'
m7nsga2_data, m7nsga2_sols = load_data(path_experiments + label + '/' 'NSGA2', 20, 30, [ax, ax3], marker='x')
df = df.append(prepare_dataframe(m7nsga2_data, 'NSGA2', label, 'x'))
m7spea2_data, m7spea2_sols = load_data(path_experiments + label + '/' 'SPEA2', 10, 20, [ax, ax3], marker='d')
df = df.append(prepare_dataframe(m7spea2_data, 'SPEA2', label, 'd'))

label = 'constructive-generation-mut7'
m7nsga2_generation_data, m7nsga2_generation_sols = load_data(path_experiments + label + '/' 'NSGA2', 20, 30, [ax, ax4], marker='1')
df = df.append(prepare_dataframe(m7nsga2_generation_data, 'NSGA2', label, '1'))
m7spea2_generation_data, m7spea2_generation_sols = load_data(path_experiments + label + '/' 'SPEA2', 10, 20, [ax, ax4], marker='v')
df = df.append(prepare_dataframe(m7spea2_generation_data, 'SPEA2', label, 'v'))


# Plot
plt.xlabel('Quality of service')
plt.ylabel('Cost of installation')

# df.groupby(by=['algorithm', 'label']).sum()
#for index, row in df.iterrows():

legend_elements = []
for a in df['algorithm'].unique().tolist():
    for l in df['label'].unique().tolist():
        q = f'algorithm == "{a}" and label == "{l}"'
        marker = df.query(q).iloc[[0]]['marker'].values[0]
        color = df.query(q).iloc[[0]]['color'].values[0]
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='black', label=f'{a}-{l}'))
ax.legend(handles=legend_elements, loc='center')


plt.savefig('pareto_front_sol.png', dpi=300, bbox_inches='tight')  # transparent=True,
# plt.show()
plt.close()


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



fig = plt.figure()
ax = fig.add_subplot(111)  # , projection='2d')


all_fits = np.vstack((nsga2_data, spea2_data, nsga2_generation_data, spea2_generation_data))
nadir_point = [df['f1'].min() * -1, df['f2'].max()]
print("Nadir point:", nadir_point)

pareto_optimal_hv, pareto_optimal_front = compute_hv(df[['f1', 'f2']].values, nadir_point=nadir_point)
print('Hv of pareto optimal: ', pareto_optimal_hv)


colors = ['red', 'blue', 'green', 'pink', 'black', 'yellow', 'gray', 'cyan']
next_color = 0
for a in df['algorithm'].unique().tolist():
    for l in df['label'].unique().tolist():
        q = f'algorithm == "{a}" and label == "{l}"'
        marker = df.query(q).iloc[[0]]['marker'].values[0]
        c = colors[next_color]
        df.loc[(df.algorithm == a) & (df.label == l), 'color'] = c
        tag = a + '-' + l
        _ = print_pareto_hv(df.query(q)[['f1', 'f2']].values, nadir_point, ax, tag, relative=pareto_optimal_hv, color=c, marker=marker)
        next_color = (next_color + 1) % len(colors)


plt.xlabel('Quality of service')
plt.ylabel('Cost of installation')


legend_elements = []
for a in df['algorithm'].unique().tolist():
    for l in df['label'].unique().tolist():
        q = f'algorithm == "{a}" and label == "{l}"'
        marker = df.query(q).iloc[[0]]['marker'].values[0]
        color = df.query(q).iloc[[0]]['color'].values[0]
        legend_elements.append(Line2D([0], [0], marker=marker, color=color, markerfacecolor=color, label=f'{a}-{l}'))
ax.legend(handles=legend_elements)


plt.savefig('pareto_front.png', dpi=300, bbox_inches='tight')  # transparent=True,
plt.show()


# MO Indicators

gd = get_performance_indicator("gd", pareto_optimal_front)
gd_plus = get_performance_indicator("gd+", pareto_optimal_front)
igd = get_performance_indicator("igd", pareto_optimal_front)
igd_plus = get_performance_indicator("igd+", pareto_optimal_front)

df_mo_metrics = pd.DataFrame(columns=['algorithm', 'label', 'hv', 'gd', 'gd+', 'igd', 'igd+'])
for a in df['algorithm'].unique().tolist():
    for l in df['label'].unique().tolist():
        q = f'algorithm == "{a}" and label == "{l}"'
        data = df.query(q)[['f1', 'f2']].values
        hv, pp = compute_hv(data, nadir_point)
        hv_rel = hv / pareto_optimal_hv
        line = [[a, l, hv_rel, gd.do(pp), gd_plus.do(pp), igd.do(pp), igd_plus.do(pp)]]
        df_line = pd.DataFrame(line, columns=['algorithm', 'label', 'hv', 'gd', 'gd+', 'igd', 'igd+'])
        df_mo_metrics = df_mo_metrics.append(df_line)

print(df_mo_metrics.to_latex(index=False, float_format='%.2f'))
