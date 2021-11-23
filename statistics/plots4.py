
from my_utilities import *
from matplotlib.lines import Line2D
import pygmo
from pymoo.factory import get_performance_indicator


# Load data

df = pd.DataFrame(columns=['algorithm', 'label', 'f1', 'f2', 'marker'])

path_experiments = '/home/cintrano/_current/EV-CSL-results/'


fig, ax = plt.subplots()

label = 'constructive-generation'
nsga2_f, nsga2_s = load_data(path_experiments + label + '/' 'NSGA2', 0, 30, [ax], marker='o')
df = df.append(prepare_dataframe(nsga2_f, 'NSGA2', '', 's'))
spea2_f, spea2_s = load_data(path_experiments + label + '/' 'SPEA2', 30, 60, [ax], marker='^')
df = df.append(prepare_dataframe(spea2_f, 'SPEA2', '', '^'))
rs_f, rs_s = load_data(path_experiments + 'random-search/' + 'RS', 0, 30, [ax], marker='x')
df = df.append(prepare_dataframe(rs_f, 'RS', '', '^'))

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
ax.legend(handles=legend_elements) #, loc='center')


plt.savefig('pareto_front_sol_Final.png', dpi=300, bbox_inches='tight')  # transparent=True,
# plt.show()
plt.close()



fig = plt.figure()
ax = fig.add_subplot(111)  # , projection='2d')


#all_fits = np.vstack((nsga2_data, spea2_data, nsga2_generation_data, spea2_generation_data))
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


plt.savefig('pareto_front_Final.png', dpi=300, bbox_inches='tight')  # transparent=True,
# plt.show()


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
        print(f'size of {a} is {len(pp)}')
        hv_rel = hv / pareto_optimal_hv
        line = [[a, l, hv_rel, gd.do(pp), gd_plus.do(pp), igd.do(pp), igd_plus.do(pp)]]
        df_line = pd.DataFrame(line, columns=['algorithm', 'label', 'hv', 'gd', 'gd+', 'igd', 'igd+'])
        df_mo_metrics = df_mo_metrics.append(df_line)

print(df_mo_metrics.to_latex(index=False, float_format='%.3f'))
