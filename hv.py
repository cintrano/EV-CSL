import os

import numpy as np
import pygmo as pg

# python3 363 33550 14 CxA_filter.ssv pertinencia.ssv
# cp.ssv facility_points.ssv substations_points.ssv
# --pop 10 --ga-sel None --ga-cross None --ga-mut None --ga-repl mu_lambda

FITNESS_FILE = "fits.ssv"
POPULATION_FILE = "pop.ssv"


if __name__ == '__main__':
    combinations = [(0.5, 0.05), (0.5, 0.1), (0.5, 0.2), (0.7, 0.05), (0.7, 0.1), (0.7, 0.2), (0.9, 0.05), (0.9, 0.1), (0.9, 0.2)]
    for cr, mut in combinations:
        file = os.path.dirname(__file__) + "/outputs/" + str(cr) + "-" + str(mut) + '-' + FITNESS_FILE
        #print(file)
        data = np.genfromtxt(file, delimiter=' ', skip_header=0)
        data[:,0] = data[:,0] * -1
        data_list = data.tolist()
        hyp = pg.hypervolume(data)
        print(hyp.compute([100000000, 100000000]))
