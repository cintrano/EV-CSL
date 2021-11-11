import random
import numpy as np


def mut_none(ind):
    return ind


def remove_station(ind):
    ind[np.random.choice(np.where(ind != 0)[0])] = 0
    return ind


def add_station(ind):
    station_type = random.randint(1, 2)
    ind[np.random.choice(np.where(ind == 0)[0])] = station_type
    return ind


def change_type(ind):
    station_type = random.randint(1, 2)
    ind[np.random.choice(np.where(ind == 0)[0])] = station_type
    return ind


def custom(ind):
    prob = random.random()
    if prob < 1/3:
        return remove_station(ind),
    elif prob < 2/3:
        return change_type(ind),
    else:
        return add_station(ind),
