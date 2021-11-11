

# Crossover operators
import numpy as np

import problem
from arguments import args


def merging(ind1, ind2):
    pass


def cupcap(ind1, ind2):
    pass


def cross_none(ind1, ind2):
    return ind1, ind2


def cross_by_substations_zone(ind1, ind2):
    """Swap an entire zone of action by a random electric substation
    :param ind1: Individual
    :param ind2: Individual
    :return: The crossed individuals
    """
    i1 = np.copy(ind1)
    i2 = np.copy(ind2)
    station = np.random.randint(0, args.E, 1)
    mask_matrix = problem.relevance_substations[station, :].astype(bool)
    mask = mask_matrix[0]
    x = i1[mask]
    i1[mask] = i2[mask]
    i2[mask] = x
    return i1, i2
