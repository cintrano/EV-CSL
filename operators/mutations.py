import numpy as np
import numpy.ma as ma

import problem
from arguments import args


def mut_none(ind):
    return ind


def remove_station(ind):
    ind[np.random.choice(np.where(ind != 0)[0])] = 0
    return ind


def add_station(ind):
    station_type = np.random.randint(1, 3)
    ind[np.random.choice(np.where(ind == 0)[0])] = station_type
    return ind


def change_type(ind):
    station_type = np.random.randint(1, 3)
    ind[np.random.choice(np.where(ind != 0)[0])] = station_type
    return ind


def custom(ind):
    prob = np.random.random()
    if prob < 1/3:
        return remove_station(ind),
    elif prob < 2/3:
        return change_type(ind),
    else:
        return add_station(ind),


def mutation_by_substation_zones(ind, p=1/14):
    """Same operator as custom but limited by zones.
    :param ind:
    :param p: Probability of mutate a zone
    :return:
    """
    i1 = np.copy(ind)
    for station in range(args.E):
        if np.random.random() <= p:
            mask_matrix = problem.relevance_substations[station, :]#.astype(bool)
            mask = np.where(mask_matrix == 0, 1, 0)  # invert pertinence matrix to get a mask
            masked_array = ma.array(i1, mask=mask)
            prob = np.random.random()
            if prob < 1/3:
                # Remove station
                i1[np.random.choice(np.where(masked_array != 0)[0])] = 0
            elif prob < 2/3:
                # Change type
                station_type = np.random.randint(1, 3)
                i1[np.random.choice(np.where(masked_array != 0)[0])] = station_type
            else:
                # Add station
                station_type = np.random.randint(1, 3)
                i1[np.random.choice(np.where(masked_array == 0)[0])] = station_type
    return i1,
