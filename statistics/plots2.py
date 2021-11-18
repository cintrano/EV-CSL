import os
import matplotlib.pyplot as plt
import numpy as np

from my_utilities import simple_cull, dominates

import pygmo

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
'''
fig = plt.figure()
fig1 = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(111)#, projection='2d')
ax1 = fig1.add_subplot(111)#, projection='2d')
ax2 = fig2.add_subplot(111)#, projection='2d')
'''

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


path = '/home/cintrano/_current/EV-CSL-results/random-generation/'
nsga2_data, nsga2_sols = load_data(path + 'NSGA2', 60, 90, [ax, ax1], marker='o')
spea2_data, spea2_sols = load_data(path + 'SPEA2', 30, 60, [ax, ax1], marker='+')

path = '/home/cintrano/_current/EV-CSL-results/constructive-generation/'
nsga2_generation_data, nsga2_generation_sols = load_data(path + 'NSGA2', 60, 90, [ax, ax2], marker='s')
spea2_generation_data, spea2_generation_sols = load_data(path + 'SPEA2', 30, 60, [ax, ax2], marker='^')


path = '/home/cintrano/_current/EV-CSL-results/random-generation-mut7/'
m7nsga2_data, m7nsga2_sols = load_data(path + 'NSGA2', 60, 90, [ax, ax3], marker='x')
m7spea2_data, m7spea2_sols = load_data(path + 'SPEA2', 30, 60, [ax, ax3], marker='d')

path = '/home/cintrano/_current/EV-CSL-results/constructive-generation-mut7/'
m7nsga2_generation_data, m7nsga2_generation_sols = load_data(path + 'NSGA2', 60, 90, [ax, ax4], marker='1')
m7spea2_generation_data, m7spea2_generation_sols = load_data(path + 'SPEA2', 30, 60, [ax, ax4], marker='v')



print(np.count_nonzero(nsga2_sols) / (30*20), np.count_nonzero(spea2_sols) / (30*20), np.count_nonzero(nsga2_generation_sols) / (30*20), np.count_nonzero(spea2_generation_sols) / (30*20))


# Plot
plt.xlabel('Quality of service')
plt.ylabel('Cost of installation')

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='NSGA2-rand'),
                   Line2D([0], [0], marker='+', color='w', markerfacecolor='black', label='SPEA2-rand'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='black', label='NSGA2-constr'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='black', label='SPEA2-constr')]
ax.legend(handles=legend_elements)  # , loc='center')


plt.savefig('pareto_front_sol.png', dpi=300, bbox_inches='tight')  # transparent=True,
plt.show()




fig = plt.figure()
ax = fig.add_subplot(111)  # , projection='2d')


all_fits = np.vstack((nsga2_data, spea2_data, nsga2_generation_data, spea2_generation_data))
nadir_point = [np.max(all_fits[:, 0]), np.max(all_fits[:, 1])]
print("Nadir point:", nadir_point)


def print_pareto_hv(data, nadir_point, ax, label, color='red', marker='o'):
    inputPoints = data.tolist()
    paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)
    dp = np.array(list(dominatedPoints))
    pp = np.array(list(paretoPoints))
    # print(pp.shape,dp.shape)
    ax.scatter(pp[:, 0], pp[:, 1], color=color, marker=marker)
    pp[:, 0] *= -1
    hv = pygmo.hypervolume(pp)
    print(label, '&', hv.compute(nadir_point))
    return pp


pp_nsga2_rand = print_pareto_hv(nsga2_data, nadir_point, ax, 'NSGA2-rand', color='red', marker='o')
pp_spea2_rand = print_pareto_hv(spea2_data, nadir_point, ax, 'SPEA2-rand', color='blue', marker='+')
pp_nsga2_const = print_pareto_hv(nsga2_generation_data, nadir_point, ax, 'NSGA2-const', color='green', marker='s')
pp_spea2_const = print_pareto_hv(spea2_generation_data, nadir_point, ax, 'SPEA2-const', color='black', marker='^')


plt.xlabel('Quality of service')
plt.ylabel('Cost of installation')


legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='NSGA2-rand'),
                   Line2D([0], [0], marker='+', color='w', markerfacecolor='blue', label='SPEA2-rand'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='green', label='NSGA2-constr'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='black', label='SPEA2-constr')]
ax.legend(handles=legend_elements)


plt.savefig('pareto_front.png', dpi=300, bbox_inches='tight')  # transparent=True,
plt.show()



'''



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)#, projection='2d')

# All
#nadir_point = [np.max(whole_data[:, 0])+1, np.max(whole_data[:, 1])+1]
data = np.vstack((nsga2_data, greedy_data))
inputPoints = data.tolist()
paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)
dp = np.array(list(dominatedPoints))
pp = np.array(list(paretoPoints))

from pymoo.factory import get_performance_indicator

gd = get_performance_indicator("gd", pp)
gd_plus = get_performance_indicator("gd+", pp)
igd = get_performance_indicator("igd", pp)
igd_plus = get_performance_indicator("igd+", pp)

hv = hypervolume(pp)
hv_opt = hv.compute(nadir_point)
print('Pareto opt.', '&', hv_opt)
print("---------")
#print("GD", "EDS=", gd.do(greedy_data), "NSGAII=", gd.do(nsga2_data))
print("Pareto opt. &", "%.2f" % gd.do(pp), "&", "%.2f" % gd_plus.do(pp), "&", "%.2f" % igd.do(pp), "&", "%.2f" % igd_plus.do(pp))
print("NSGA-II &", "%.2f" % gd.do(pp_nsga2), "&", "%.2f" % gd_plus.do(pp_nsga2), "&", "%.2f" % igd.do(pp_nsga2), "&", "%.2f" % igd_plus.do(pp_nsga2))
print("EDS &", "%.2f" % gd.do(greedy_data), "&", "%.2f" % gd_plus.do(greedy_data), "&", "%.2f" % igd.do(greedy_data), "&", "%.2f" % igd_plus.do(greedy_data))

print("---------")



print("---------")

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)#, projection='2d')

ax.scatter(nsga2_sol_list[0][:, 0]*-1, nsga2_sol_list[0][:, 1], color='green', marker='.')
ax.scatter(nsga2_sol_list[1][:, 0]*-1, nsga2_sol_list[1][:, 1], color='yellow', marker='.')

plt.show()


i = 0
print(nadir_point)
hv_list = []
hvr_list = []
gd_list = []
gd_plus_list = []
igd_list = []
igd_plus_list = []

for s in nsga2_sol_list:
    #print(s)
    x = s
    inputPoints = x.tolist()
    paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)
    dp = np.array(list(dominatedPoints))
    pp = np.array(list(paretoPoints))
    hv = hypervolume(pp)
    hv_d = hv.compute(nadir_point)
    hv_list.append(hv_d)
    hvr_value = hv_d/hv_opt
    hv_value = "{:.2e}".format(hv_d)
    hvr_list.append(hvr_value)
    gd_value = round(gd.do(x), 2)
    gd_list.append(gd_value)
    gd_plus_value = round(gd_plus.do(x), 2)
    gd_plus_list.append(gd_plus_value)
    igd_value = round(igd.do(x), 2)
    igd_list.append(igd_value)
    igd_plus_value = round(igd_plus.do(x), 2)
    igd_plus_list.append(igd_plus_value)
    print(f"{i} & {hvr_value} & {gd_value} & {gd_plus_value} & {igd_value} & {igd_plus_value} \\\\")
    i += 1

print("---------")
print("---------")


from scipy import stats
from scipy.stats import iqr
x = hv_list
print(f'{"{:.2e}".format(np.min(x))} & {"{:.2e}".format(np.max(x))} & {"{:.2e}".format(np.mean(x))} & {"{:.2e}".format(np.std(x))} & {"{:.2e}".format(np.median(x))} & {"{:.2e}".format(iqr(x))} \\\\')
x = hvr_list
print(f'{"{:.2e}".format(np.min(x))} & {"{:.2e}".format(np.max(x))} & {"{:.2e}".format(np.mean(x))} & {"{:.2e}".format(np.std(x))} & {"{:.2e}".format(np.median(x))} & {"{:.2e}".format(iqr(x))} \\\\')
x = gd_list
print(f'{"{:.2e}".format(np.min(x))} & {"{:.2e}".format(np.max(x))} & {"{:.2e}".format(np.mean(x))} & {"{:.2e}".format(np.std(x))} & {"{:.2e}".format(np.median(x))} & {"{:.2e}".format(iqr(x))} \\\\')
x = gd_plus_list
print(f'{"{:.2e}".format(np.min(x))} & {"{:.2e}".format(np.max(x))} & {"{:.2e}".format(np.mean(x))} & {"{:.2e}".format(np.std(x))} & {"{:.2e}".format(np.median(x))} & {"{:.2e}".format(iqr(x))} \\\\')
x = igd_list
print(f'{"{:.2e}".format(np.min(x))} & {"{:.2e}".format(np.max(x))} & {"{:.2e}".format(np.mean(x))} & {"{:.2e}".format(np.std(x))} & {"{:.2e}".format(np.median(x))} & {"{:.2e}".format(iqr(x))} \\\\')
x = igd_plus_list
print(f'{"{:.2e}".format(np.min(x))} & {"{:.2e}".format(np.max(x))} & {"{:.2e}".format(np.mean(x))} & {"{:.2e}".format(np.std(x))} & {"{:.2e}".format(np.median(x))} & {"{:.2e}".format(iqr(x))} \\\\')

'''
