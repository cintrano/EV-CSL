import numpy as np


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


if __name__ == '__main__':
    pass
