"""This file perform the basic statistics of the paper:
Multi-objective approach for electric vehicle charging station location: a real case study in M\'alaga, Spain"""
from my_utilities import *
from pymoo.factory import get_performance_indicator
from scipy.stats import iqr


def load_data_df(path, lower, upper, plot, algo, label, marker, load_solutions=False):
    df = pd.DataFrame(columns=['algorithm', 'label', 'run', 'f1', 'f2', 'marker'])
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
nsga2_f, nsga2_s = load_data_df(path_experiments + label + '/' 'NSGA2', 0, 30, [ax], 'NSGA2', '', marker='o')
spea2_f, spea2_s = load_data_df(path_experiments + label + '/' 'SPEA2', 30, 60, [ax], 'SPEA2', '', marker='^')
rs_f, rs_s = load_data_df(path_experiments + 'random-search/' + 'RS', 0, 30, [ax], 'RS', '', marker='x')
df = df.append(nsga2_f)
df = df.append(spea2_f)
df = df.append(rs_f)

# Plot
plt.xlabel('Quality of service')
plt.ylabel('Cost of installation')

print_legend(ax, df)

plt.savefig('pareto_front_sol_Final_1.png', dpi=300, bbox_inches='tight')  # transparent=True,
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

plt.xlabel('Quality of service')
plt.ylabel('Cost of installation')

print_legend(ax, df, color_exist=True)

plt.savefig('pareto_front_Final_1.png', dpi=300, bbox_inches='tight')  # transparent=True,
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
