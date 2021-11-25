"""This file perform the basic statistics of the paper:
Multi-objective approach for electric vehicle charging station location: a real case study in M\'alaga, Spain"""
import scipy

import problem
from my_utilities import *
from pymoo.factory import get_performance_indicator
from scipy.stats import iqr, kruskal, wilcoxon
import graph_map_prints as gmp

colors = plt.cm.rainbow(np.linspace(0, 1, 30))


def load_data_df(path, lower, upper, plot, algo, label, marker, load_solutions=False):
    df = pd.DataFrame(columns=['algorithm', 'label', 'run', 'f1', 'f2', 'marker'])
    print('loading solutions from', path)
    sols = None
    ex_pop = None
    for i in range(lower, upper):
        if i == lower and load_solutions:
            file_pop = f'{lower}-pop.ssv'
            ex_pop = np.genfromtxt(path + "/" + file_pop, delimiter=' ', skip_header=0)
            sols = ex_pop
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
        arr = np.array([list(range(ex[:, 0].size))]).T
        ex = np.append(ex, arr, axis=1)
        df_current = pd.DataFrame(ex, columns=['f1', 'f2', 'individual'])
        df_current['algorithm'] = algo
        df_current['label'] = label
        df_current['run'] = i - lower
        df_current['marker'] = marker
        df_current['color'] = 'b'
        df = df.append(df_current)
        # fit = np.vstack((fit, ex))
        if load_solutions:
            sols = np.vstack((sols, ex_pop))
    return df, sols


def load_data_pareto_df(path, lower, upper, plot, algo, label, marker, load_solutions=False):
    """As load_data_df() but we calculate the pareto fronts"""
    df = pd.DataFrame(columns=['algorithm', 'label', 'run', 'f1', 'f2', 'marker'])
    print('loading solutions from', path)
    sols = None
    ex_pop = None
    for i in range(lower, upper):
        if i == lower and load_solutions:
            file_pop = f'{lower}-pop.ssv'
            ex_pop = np.genfromtxt(path + "/" + file_pop, delimiter=' ', skip_header=0)
            sols = ex_pop
        file = f'{i}-fits.ssv'
        file_pop = f'{i}-pop.ssv'
        data_in_file = np.genfromtxt(path + "/" + file, delimiter=' ', skip_header=0)
        d = data_in_file.copy()
        d[:, 0] *= -1  # to minimize the first objective and compute the hv
        inputPoints = d.tolist()
        paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)
        dp = np.array(list(dominatedPoints))
        ex = np.array(list(paretoPoints))
        ex[:, 0] *= -1 # undone the previous transformation
        if load_solutions:
            ex_pop = np.genfromtxt(path + "/" + file_pop, delimiter=' ', skip_header=0)
        if isinstance(plot, list):
            for p in plot:
                p.scatter(ex[:, 0], ex[:, 1], color=colors[i % 30], marker=marker)
        else:
            plot.scatter(ex[:, 0], ex[:, 1], color=colors[i % 30], marker=marker)
        df_current = pd.DataFrame(ex, columns=['f1', 'f2'])
        df_current['algorithm'] = algo
        df_current['label'] = label
        df_current['run'] = i - lower
        df_current['marker'] = marker
        df_current['color'] = 'b'
        df = df.append(df_current)
        # fit = np.vstack((fit, ex))
        if load_solutions:
            sols = np.vstack((sols, ex_pop))
    return df, sols


# Load data of the final experiments
# The algorithms are a NSGA2, SPEA2, and a Random Search (RS).
# We perform 30 independent executions of each one
# The tunning parameters was the custom ones on each one (generation=ZONES, crosover=ZONES, mutation=ZONES)

df = pd.DataFrame(columns=['algorithm', 'label', 'run', 'f1', 'f2', 'marker'])

path_experiments = '/home/cintrano/_current/EV-CSL-results/'


fig, ax = plt.subplots()

label = 'constructive-generation'
nsga2_f, nsga2_s = load_data_df(path_experiments + label + '/' + 'NSGA2', 0, 30, [ax], 'NSGA2', '', marker='o')
spea2_f, spea2_s = load_data_df(path_experiments + label + '/' + 'SPEA2', 30, 60, [ax], 'SPEA2', '', marker='^')
# rs_f, rs_s = load_data_df(path_experiments + 'random-search/' + 'RS', 0, 30, [ax], 'RS', '', marker='x')
rs_f, rs_s = load_data_pareto_df(path_experiments + 'random-search/' + 'RS', 0, 30, [ax], 'RS', '', marker='X')
df = df.append(nsga2_f)
df = df.append(spea2_f)
df = df.append(rs_f)

# Plot
plt.xlabel('Quality of service')
plt.ylabel('Cost of installation')

print_legend(ax, df)

plt.savefig('pareto_front_sol_Final.png', dpi=300, bbox_inches='tight')  # transparent=True,
# plt.show()
plt.close()


# We are going to compute the different pareto fronts
# First we are compute the global pareto front for each algorithm,
# and then we compute the different metrics of quality

# fig = plt.figure()
# ax = fig.add_subplot(111)  # , projection='2d')
fig, ax = plt.subplots()

# We transform the first objective from a maximization to a minimization
nadir_point = [df['f1'].min() * -1, df['f2'].max()]
print("Nadir point:", nadir_point)

pareto_optimal_hv, pareto_optimal_front = compute_hv(df[['f1', 'f2']].values, nadir_point=nadir_point)
print('Hv of pareto optimal:', pareto_optimal_hv)


colors = ['red', 'blue', 'green', 'pink', 'black', 'yellow', 'gray', 'cyan']
next_color = 0
for a in df['algorithm'].unique().tolist():
    for l in df['label'].unique().tolist():
        q = f'algorithm == "{a}" and label == "{l}"'
        marker = df.query(q).iloc[[0]]['marker'].values[0]
        c = colors[next_color]
        df.loc[(df.algorithm == a) & (df.label == l), 'color'] = c
        tag = a + '-' + l
        data = df.query(q)[['f1', 'f2']].values
        _ = print_pareto_hv(data, nadir_point, ax, tag, relative=pareto_optimal_hv, color=c, marker=marker)
        next_color = (next_color + 1) % len(colors)

import matplotlib.patches as mpatches
x_tail = df['f1'].min()
y_tail = df['f2'].max()
x_head = df['f1'].max()
y_head = df['f2'].min()
dx = x_head - x_tail
dy = y_head - y_tail
arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head), mutation_scale=10)
ax.add_patch(arrow)


plt.xlabel('Quality of service')
plt.ylabel('Cost of installation')

print_legend(ax, df, color_exist=True)

plt.savefig('pareto_front_Final.png', dpi=300, bbox_inches='tight')  # transparent=True,
# plt.show()


# MO Indicators

# Now, we are going to calculate the multi-objective indicators for the different pareto fronts.

gd = get_performance_indicator("gd", pareto_optimal_front)
gd_plus = get_performance_indicator("gd+", pareto_optimal_front)
igd = get_performance_indicator("igd", pareto_optimal_front)
igd_plus = get_performance_indicator("igd+", pareto_optimal_front)

# Global metrics for each algorithm
df_mo_metrics = pd.DataFrame(columns=['algorithm', 'label', 'hv', 'gd', 'gd+', 'igd', 'igd+'])
for a in df['algorithm'].unique().tolist():
    for l in df['label'].unique().tolist():
        q = f'algorithm == "{a}" and label == "{l}"'
        data = df.query(q)[['f1', 'f2']].values
        hv, pp = compute_hv(data, nadir_point)
        print(f'size of {a} is {len(pp)}')
        hv_rel = hv / pareto_optimal_hv
        line = [[a, l, hv_rel, gd.do(pp), gd_plus.do(pp), igd.do(pp), igd_plus.do(pp)]]
        df_line = pd.DataFrame(line, columns=['algorithm', 'label', 'hv', 'gd', 'gd+', 'igd', 'igd+'])
        df_mo_metrics = df_mo_metrics.append(df_line)

print(df_mo_metrics.to_latex(index=False, float_format='%.3f'))

# Metrics for each run of each algorithm
df_mo_metrics = pd.DataFrame(columns=['algorithm', 'label', 'run', 'hv', 'gd', 'gd+', 'igd', 'igd+'])
for a in df['algorithm'].unique().tolist():
    for l in df[df.algorithm == a]['label'].unique().tolist():
        for r in df[(df.algorithm == a) & (df.label == l)]['run'].unique().tolist():
            q = f'algorithm == "{a}" and label == "{l}" and run == {r}'
            data = df.query(q)[['f1', 'f2']].values
            hv, pp = compute_hv(data, nadir_point)
            # print(f'size of {a} is {len(pp)}')
            hv_rel = hv / pareto_optimal_hv
            line = [[a, l, r, hv_rel, gd.do(pp), gd_plus.do(pp), igd.do(pp), igd_plus.do(pp)]]
            df_line = pd.DataFrame(line, columns=['algorithm', 'label', 'run', 'hv', 'gd', 'gd+', 'igd', 'igd+'])
            df_mo_metrics = df_mo_metrics.append(df_line)

print(df_mo_metrics.to_latex(index=False, float_format='%.3f'))

print(df_mo_metrics.groupby("algorithm").describe().to_latex(float_format='%.3f'))

# This is a small adjustment to generate the latex table.
metrics = ['hv', 'gd', 'gd+', 'igd', 'igd+']
for a in df_mo_metrics['algorithm'].unique().tolist():
    for l in df_mo_metrics[df_mo_metrics.algorithm == a]['label'].unique().tolist():
        for m in metrics:
            print(f'{a} & {m}', end=' & ')
            q = f'algorithm == "{a}" and label == "{l}"'
            x = df_mo_metrics.query(q)[m].values
            statistics_values = (np.min(x), np.mean(x), np.std(x), np.median(x), iqr(x), np.max(x))
            print(' %.2f & %.2f $\\pm$%.2f  & %.2f  & %.2f  & %.2f ' % statistics_values, end=' \\\\\n ')
    print()

# We calculate if there exist significative differences between the mo-indicators of each algorithm
# For this, we perform a Wilcoxon Signed-Rank test and a Kruskal-Wallis
for m in metrics:
    series = []
    for a in df_mo_metrics['algorithm'].unique().tolist():
        for l in df_mo_metrics[df_mo_metrics.algorithm == a]['label'].unique().tolist():
            q = f'algorithm == "{a}" and label == "{l}"'
            x = df_mo_metrics.query(q)[m].values
            series.append(x)
    stat, p = kruskal(series[0], series[1], series[2])
    print(p)
    print(m, 'Kruscal-Wallis', 'Statistics=%.3f, p=%f' % (stat, p), end='\t')
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    stat0, p0 = wilcoxon(series[0], series[1])
    stat1, p1 = wilcoxon(series[1], series[2])
    stat2, p2 = wilcoxon(series[2], series[0])
    print(m, 'Wilcoxon Signed-Rank', 'Stat.=%.3f, p=%f\tStat.=%.3f, p=%f\tStat.=%.3f, p=%f' % (stat0, p0, stat1, p1, stat2, p2))


# Finally, we calculate the QoS metrics for each algorithm
def qos_metrics(solution):
    sol_index = [i for i, e in enumerate(solution) if e != 0]
    values = np.copy(solution[solution != 0])
    values[values == 1] = 1.0
    values[values == 2] = 8.0
    matrix = problem.clients_assignment[:, sol_index] * problem.customer_points[:, 2][:, np.newaxis]  # overlapping duplicate the customer points
    clients = np.sum(matrix, axis=0)  # Sum down
    zeros = np.sum(problem.clients_assignment[:, sol_index], axis=1)
    zeros = np.where(zeros > 0, 0, 1)

    qos = np.sum(values * clients)
    not_service = np.sum(zeros * problem.customer_points[:, 2])

    # qos = problem.scale("qos", qos)
    # not_service = problem.scale("not_service", not_service)

    num_1 = len([i for i, e in enumerate(solution) if e == 1])  # Number of stations of type 1
    num_2 = len([i for i, e in enumerate(solution) if e == 2])  # Number of stations of type 2
    return qos, not_service, num_1, num_2


cost_min = df['f2'].min()
cost_max = df['f2'].max()
cost_50 = cost_min + 0.5 * (cost_max-cost_min)
cost_75 = cost_min + 0.75 * (cost_max-cost_min)
percentage = (df[df.algorithm == 'SPEA2']['f2'].max() - cost_min) / (cost_max-cost_min)
print('The max percentage of SPEA2 is ', percentage)
cost_90 = df[df.algorithm == 'SPEA2']['f2'].max()
cost_limit = {'cost_50': cost_50, 'cost_75': cost_75, 'cost_90': cost_90}

df_columns = ['cost', 'algorithm', 'run', 'individual', 'qos', 'not_service', 'num_1', 'num_2']
df_qos = pd.DataFrame(columns=df_columns)

for a in ['NSGA2', 'SPEA2']:
    lower = 0
    if a == 'SPEA2':
        lower = 30
    for t in ['cost_50', 'cost_75', 'cost_90']:
        # for l in df[df.algorithm == a]['label'].unique().tolist():
        q = f'algorithm == "{a}"'
        data = df.query(q)
        row = data.iloc[(data['f2'] - cost_limit[t]).abs().argsort()[:2]]
        row = row.loc[row['f2'].idxmax()]
        pwd = path_experiments + label + '/' + a
        run = row['run'] + lower
        individual = int(row['individual'])
        file_pop = f'{run}-pop.ssv'
        ex_pop = np.genfromtxt(pwd + "/" + file_pop, delimiter=' ', skip_header=0)
        qos, not_service, num_1, num_2 = qos_metrics(ex_pop[individual, :])

        line = [[t, a, run, individual, qos, not_service, num_1, num_2]]
        df_current = pd.DataFrame(line, columns=df_columns)

        df_qos = df_qos.append(df_current)


print(df_qos.to_latex(index=False, float_format='%.3f'))

# The statistics are not necessary
# metrics = ['qos', 'not_service', 'num_1', 'num_2']
# for m in metrics:
#     for a in df_qos['algorithm'].unique().tolist():
#         print(f'{m} & {a}', end=' & ')
#         q = f'algorithm == "{a}"'
#         x = df_qos.query(q)[m].values
#         statistics_values = (np.min(x), np.mean(x), np.std(x), np.median(x), iqr(x), np.max(x))
#         print(' %.2f & %.2f $\\pm$%.2f  & %.2f  & %.2f  & %.2f ' % statistics_values, end=' \\\\\n ')
#      print()

# And finally, we print the 50\%, 75\% and 90\% solutions

for t in ['cost_50', 'cost_75', 'cost_90']:
    sols = []
    for a in ['NSGA2', 'SPEA2']:
        lower = 0
        if a == 'SPEA2':
            lower = 30
        # for l in df[df.algorithm == a]['label'].unique().tolist():
        q = f'algorithm == "{a}"'
        data = df.query(q)
        row = data.iloc[(data['f2'] - cost_limit[t]).abs().argsort()[:2]]
        row = row.loc[row['f2'].idxmax()]
        pwd = path_experiments + label + '/' + a
        run = row['run'] + lower
        individual = int(row['individual'])
        file_pop = f'{run}-pop.ssv'
        ex_pop = np.genfromtxt(pwd + "/" + file_pop, delimiter=' ', skip_header=0)
        sol = ex_pop[individual, :]
        sol_index = [i for i, e in enumerate(sol) if e != 0]
        sols.append(sol_index)
    gmp.main2(sols, ['NSGA2', 'SPEA2'], f'{t}_map.pdf')
