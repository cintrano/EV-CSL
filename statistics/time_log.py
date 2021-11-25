"""Process the time from the execution logs"""
import re
import glob
import numpy as np
from scipy import stats
from scipy.stats import iqr

folder = '/home/cintrano/_current/EV-CSL-results/logs/'
algorithm = ['NSGA2', 'SPEA2', 'RS']

lines = []
times = []

for a in algorithm:
    for file in glob.glob(folder + f'*{a}*.err'):
        with open(file) as f:
            for line in f.readlines():
                tokens = line.split('\t')
                if tokens[0] == 'user':
                    s = tokens[1]
                    for match in re.finditer(r'\b(\d+)m(\d+\.\d+)s\b', s):
                        t = float(match.groups()[0]) * 60 + float(match.groups()[1])
                        # print(t)
                        times.append(t)
    # print(times)

    x = np.array(times)
    statistics_values = (np.min(x), np.mean(x), np.std(x), np.median(x), iqr(x), np.max(x))
    print(' %.2f & %.2f $\\pm$%.2f  & %.2f  & %.2f  & %.2f ' % statistics_values, end=' \\\\\n ')
    print(a, stats.describe(x))

