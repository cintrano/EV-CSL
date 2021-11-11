# Monte Carlo to get the $\alpha$, $\beta$, and $\gamma$
import csv
import numpy as np
import pandas as pd

from arguments import args
import problem


def run_montecarlo(path='./outputs/', file='sols-greedy.ssv'):
    parameters = []
    data = np.genfromtxt(path + '/' + file, delimiter=',', skip_header=1)
    i = 0
    for sol in data:
        q, on, oc, nn, nc = problem.montecarlo_components(sol)
        num_stations = len([i for i, e in enumerate(sol) if e != 0])
        parameters.append({'id': i, 'num_stations': num_stations, 'solution': sol.astype(int).tolist(),
                          'qos': q, 'overlapping_neighbourhood': on, 'overlapping_citizen': oc,
                          'not_service_neighbourhood': nn, 'not_service_citizen': nc})
        i += 1
    return parameters


def save_fits(data, filename='fits-montecarlo_from_greedy.ssv'):
    with open(args.out_path + '/' + filename, 'w') as out:
        dict_writer = csv.DictWriter(out, fieldnames=['id', 'num_stations', 'qos', 'overlapping_neighbourhood',
                                                      'overlapping_citizen', 'not_service_neighbourhood',
                                                      'not_service_citizen'], extrasaction='ignore')
        dict_writer.writeheader()
        dict_writer.writerows(data)


def save_sols(data, filename='sols-montecarlo_from_greedy.ssv'):
    with open(args.out_path + '/' + filename, 'w') as out:
        dict_writer = csv.DictWriter(out, fieldnames=['id', 'solution'], extrasaction='ignore')
        dict_writer.writeheader()
        dict_writer.writerows(data)


def read_fits(path=args.out_path, filename='fits-montecarlo_from_greedy.ssv'):
    return np.genfromtxt(path + '/' + filename, delimiter=',', skip_header=1)


if __name__ == '__main__':
    print(args)
    dict_sols = run_montecarlo()
    save_fits(dict_sols)
    save_sols(dict_sols)

    df = pd.read_csv(args.out_path + '/' + 'fits-montecarlo_from_greedy.ssv')
    df.describe().to_csv('./statistics/' + 'components_qos_description.csv')
    print(df.describe())
