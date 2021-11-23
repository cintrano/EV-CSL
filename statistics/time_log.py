"""Process the time from the execution logs"""
import re
import glob
import numpy as np
from scipy import stats

folder = '/home/cintrano/_current/EV-CSL-results/logs/'

lines = []
times = []
for file in glob.glob(folder + '*.err'):
    with open(file) as f:
        for line in f.readlines():
            tokens = line.split('\t')
            if tokens[0] == 'user':
                s = tokens[1]
                for match in re.finditer(r'\b(\d+)m(\d+\.\d+)s\b', s):
                    t = float(match.groups()[0]) * 60 + float(match.groups()[1])
                    print(t)
                    times.append(t)
print(times)
print(stats.describe(np.array(times)))

