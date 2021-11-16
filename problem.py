
import numpy as np
from numpy import ma

from operators import mutations
from arguments import args


# Constants
QOS_MIN, QOS_MAX = 175962, 1670029
OVERLAPPING_CITIZEN_MIN, OVERLAPPING_CITIZEN_MAX = 0, 356295
NOT_SERVICE_CITIZEN_MIN, NOT_SERVICE_CITIZEN_MAX = 353359, 567953  # 542540

QOS_DISTANCE = QOS_MAX - QOS_MIN
OVERLAPPING_CITIZEN_DISTANCE = OVERLAPPING_CITIZEN_MAX - OVERLAPPING_CITIZEN_MIN
NOT_SERVICE_CITIZEN_DISTANCE = NOT_SERVICE_CITIZEN_MAX - NOT_SERVICE_CITIZEN_MIN


def scale(mode, value):
    """Rescale the QoS components into the range [0, 1]
    :param mode: Component of the QoS
    :param value: Value to rescale
    :return: Re-scaled value
    """
    if mode == "qos":
        return (value - QOS_MIN) / QOS_DISTANCE
    if mode == "overlapping":
        return (value - OVERLAPPING_CITIZEN_MIN) / OVERLAPPING_CITIZEN_DISTANCE
    if mode == "not_service":
        return (value - NOT_SERVICE_CITIZEN_MIN) / NOT_SERVICE_CITIZEN_DISTANCE


distances = np.empty([args.C, args.L])
clients_assignment = np.empty([args.C, args.L])
distances_substations = np.empty([args.E, args.L])
relevance_substations = np.empty([args.E, args.L])  # Like distance_substation but binary
# neighborhood = np.empty(args.F, args.k)
customer_points = np.empty([args.C, 3])
locations_points = np.empty([args.L, 3])
substations_points = np.empty([args.E, 3])
max_customer_by_station = -1


def read_file(file):
    return np.genfromtxt(args.in_path + "/" + file, delimiter=' ', skip_header=0)


if args.matrix_file is not None:
    distances = read_file(args.matrix_file)
    clients_assignment = distances <= args.station_radius
    clients_assignment = clients_assignment.astype(int)
    max_customer_by_station = np.max(np.sum(clients_assignment, axis=0))
if args.energy_matrix_file is not None:
    distances_substations = read_file(args.energy_matrix_file)
    relevance_substations = np.where(distances_substations < 0, 0, 1)
if args.customer_file is not None:
    customer_points = read_file(args.customer_file)
    max_customer_by_station = max_customer_by_station * np.max(customer_points[:, 2])
if args.locations_file is not None:
    locations_points = read_file(args.locations_file)
if args.energy_stations_file is not None:
    substations_points = read_file(args.energy_stations_file)


# geopy.distance.vincenty(coords_1, coords_2).km

def rand_locations(num):
    return np.random.randint(args.L, size=num)


def rand_slots(param):
    return np.random.randint(3, size=param) + 1  # Just to clarify the vector


def penalty(sol):
    num_rows, _ = distances_substations.shape
    for station in np.arange(num_rows):
        station_row = distances_substations[station, :]
        max_stations = int(substations_points[station, 2])
        current_stations = sum(sol[station_row > 0])
        if current_stations > max_stations:
            return True
    return False


def fitness_cost(solution):
    """Cost of installation a charging station
    :param solution:
    :return:
    """
    dists = np.max(distances_substations, axis=0)  # Because nones are -1
    installation_cost = np.copy(solution)
    installation_cost[installation_cost == 1] = 13915  # Fast station
    installation_cost[installation_cost == 2] = 39939  # Super fast station
    wire_cost = np.copy(solution)
    wire_cost[wire_cost == 1] = 1.15  # Fast station
    wire_cost[wire_cost == 2] = 1.35  # Super fast station
    values = installation_cost + (dists * wire_cost)
    out = np.sum(values)
    return out


def get_distances(solution):
    """Get a numpy vector with the min distances of each customer to their nearest station
    :param solution:
    :return: Numpy vector
    """
    if isinstance(solution, list):
        sol_list = solution
    else:
        sol_list = solution.astype(int).tolist()
    return np.min(distances[:, sol_list], axis=1)


def serving_stations_by_clients(solution):
    sol = [i for i, e in enumerate(solution) if e != 0]
    matrix = distances[:, sol]
    closest_stations_to_clients = np.argmin(matrix, axis=1).astype(int).tolist()
    values = np.copy(solution[sol][closest_stations_to_clients])
    values[values == 1] = 1.0
    values[values == 2] = 8.0
    # if np.sum(values) == 0:
    #     print("----")
    #     print('sol', sol)
    #     print('closest_stations_to_clients', closest_stations_to_clients)
    #     print('values', values)
    return values.astype(float)


def fitness_qos(solution):
    sol = [i for i, e in enumerate(solution) if e != 0]
    serving = serving_stations_by_clients(solution)  # np.array([e for e in map(serving_user, solution) if e != 0])
    num_clients = customer_points[:, 2]  # num_clients_of_solution(sol)
    distances_customers = np.min(distances[:, sol], axis=1).astype(float)  # get_distances(sol)
    return np.sum(serving * num_clients / distances_customers)


def fitness_qos_all(solution):
    sol_index = [i for i, e in enumerate(solution) if e != 0]
    amount = 0
    for sol in sol_index:
        matrix = distances[:, sol]
        values = np.copy(solution[sol])
        values[values == 1] = 1.0
        values[values == 2] = 8.0
        serving = values.astype(float)
        num_clients = customer_points[:, 2]  # num_clients_of_solution(sol)
        distances_customers = matrix
        amount = amount + np.sum(serving * num_clients / distances_customers)
    return amount


def fitness_qos_super(solution):
    """Version with three terms
    $\sum (\alpha\cdot qos) + (\beta\cdot overlaping) + (\gamma\cdot not\ service)$
    $qos = #users \cdot type$
    :param solution: location assignments
    :return: fitness value
    """
    sol_index = [i for i, e in enumerate(solution) if e != 0]
    values = np.copy(solution[solution != 0])
    values[values == 1] = 1.0
    values[values == 2] = 8.0
    matrix = clients_assignment[:, sol_index] * customer_points[:, 2][:, np.newaxis]  # overlapping duplicate the customer points
    clients = np.sum(matrix, axis=0)  # Sum down
    zeros = np.sum(clients_assignment[:, sol_index], axis=1)
    zeros = np.where(zeros > 0, 0, 1)
    overlap = np.sum(clients_assignment[:, sol_index], axis=1)  # Sum right
    overlap = np.where(overlap > 0, overlap - 1, overlap)

    qos = np.sum(values * clients)
    overlapping = np.sum(overlap * customer_points[:, 2])
    not_service = np.sum(zeros * customer_points[:, 2])

    qos = scale("qos", qos)
    overlapping = scale("overlapping", overlapping)
    not_service = scale("not_service", not_service)

    f = qos * args.alpha - overlapping * args.beta - not_service * args.gamma
    return f


def montecarlo_components(solution):
    """Version with three terms
    $\sum (\alpha\cdot qos) + (\beta\cdot overlaping) + (\gamma\cdot not\ service)$
    $qos = #users \cdot type$
    :param solution: location assignments
    :return: fitness value
    """
    sol_index = [i for i, e in enumerate(solution) if e != 0]
    values = np.copy(solution[solution != 0])
    values[values == 1] = 1.0
    values[values == 2] = 8.0
    matrix = clients_assignment[:, sol_index] * customer_points[:, 2][:, np.newaxis]  # overlapping duplicate the customer points
    clients = np.sum(matrix, axis=0)  # Sum down
    #zeros = len(clients_assignment[:, 1]) - np.sum(clients_assignment[:, sol_index], axis=0)
    zeros = np.sum(clients_assignment[:, sol_index], axis=1)
    zeros = np.where(zeros > 0, 0, 1)
    overlap = np.sum(clients_assignment[:, sol_index], axis=1)  # Sum right
    overlap = np.where(overlap > 0, overlap - 1, overlap)

    qos = np.sum(values * clients)
    overlapping_neighbourhoods = np.sum(overlap)  # It don't take into account the importance of the location
    overlapping_citizens = np.sum(overlap * customer_points[:, 2])  # It don't take into account the importance of the location
    overlapping = np.sum(overlap)  # It don't take into account the importance of the location
    not_service_neighbourhoods = np.sum(zeros)
    not_service_citizens = np.sum(zeros * customer_points[:, 2])
    return qos, overlapping_neighbourhoods, overlapping_citizens, not_service_neighbourhoods, not_service_citizens


def fitness_distance(solution):
    sol = [i for i, e in enumerate(solution) if e != 0]
    minimums = np.min(distances[:, sol], axis=1)
    return np.dot(minimums, customer_points[:, 0].T)


def fitness_mono(solution):
    if penalty(solution):
        return np.inf,
    sol = [i for i, e in enumerate(solution) if e != 0]
    mins = np.min(distances[:, sol], axis=1)
    return np.dot(mins, customer_points[:, 1].T),


def repair_basic(ind, fitness):
    """Repair the solution
    :return: Solution
    """
    # print(ind.fitness.values)
    while np.isinf(ind.fitness.values[0]):
        ind = mutations.remove_station(ind)
        fit = fitness(ind)
        ind.fitness.values = fit
    return ind,


def consuming_energy_array(a):
    """Consuming energy
    a[a == 1] = 7.4 -> Fast station
    a[a == 2] = 50.0 -> Super fast station
    a[a == 0] = 0.0 -> No station
    :param a: solution
    :return:
    """
    return (a == 1).sum() * 7.4 + (a == 2).sum() * 50.0


def satisfiability(individual):
    num_rows, _ = distances_substations.shape
    for energy_station in np.arange(num_rows):
        energy_station_row = distances_substations[energy_station, :]  # distances for this station
        modified_energy_supply = energy_station_row
        modified_energy_supply[modified_energy_supply == -1] = 0  # Same as before but changing -1 to 0

        max_energy_supply = float(substations_points[energy_station, 2])
        mask = list(modified_energy_supply != 0)
        ind_in_this_energy_station = individual[mask]
        current_charging_demand = (ind_in_this_energy_station == 1).sum() * 7.4 +\
                                  (ind_in_this_energy_station == 2).sum() * 50.0
        if current_charging_demand > max_energy_supply:
            return False
    return True


def repair(individual):
    num_rows, _ = distances_substations.shape
    for energy_station in np.arange(num_rows):
        energy_station_row = distances_substations[energy_station, :]  # distances for this station
        modified_energy_supply = energy_station_row
        modified_energy_supply[modified_energy_supply == -1] = 0  # Same as before but changing -1 to 0

        max_energy_supply = float(substations_points[energy_station, 2])
        mask = list(modified_energy_supply != 0)
        ind_in_this_energy_station = individual[mask]
        # current_charging_demand = consuming_energy_array(ind_in_this_energy_station)
        current_charging_demand = (ind_in_this_energy_station == 1).sum() * 7.4 +\
                                  (ind_in_this_energy_station == 2).sum() * 50.0
        s = np.where(individual == 0, individual, modified_energy_supply)
        s_new = np.array([i for i, e in enumerate(s) if e != 0])
        # Calculate the minimun number of stations to be removed
        fast_remove = int((current_charging_demand - max_energy_supply) / 80)
        if fast_remove > 0:
            idxs = np.random.choice(s_new, size=fast_remove)
            for idx in idxs:
                s_new = s_new[s_new != idx]
            individual[idxs] = 0
            ind_in_this_energy_station = individual[mask]
            # current_charging_demand = consuming_energy_array(ind_in_this_energy_station)
            current_charging_demand = (ind_in_this_energy_station == 1).sum() * 7.4 +\
                                      (ind_in_this_energy_station == 2).sum() * 50.0
        while current_charging_demand > max_energy_supply:
            idx = np.random.choice(s_new)
            s_new = s_new[s_new != idx]
            individual[idx] = 0
            ind_in_this_energy_station = individual[mask]
            # current_charging_demand = consuming_energy_array(ind_in_this_energy_station)
            current_charging_demand = (ind_in_this_energy_station == 1).sum() * 7.4 +\
                                      (ind_in_this_energy_station == 2).sum() * 50.0
    #print('.', end='')
    return individual,


# Press the green button in the gutter to run the script.
def constructive_solution_by_zones(ind):
    for j in range(args.L):
        ind[j] = 0
    for station in range(args.E):
        mask_matrix = relevance_substations[station, :]  # .astype(bool)
        mask = np.where(mask_matrix == 0, 1, 0)  # invert pertinence matrix to get a mask
        masked_array = ma.array(ind, mask=mask)
        last_station_added = -1
        while satisfiability(ind):
            station_type = np.random.randint(1, 3)
            last_station_added = np.random.choice(np.where(masked_array == 0)[0])
            ind[last_station_added] = station_type
        ind[last_station_added] = 0  # because it is not satisfacibility
    return ind,


def fitness_mo(solution):
    # f_qos = fitness_qos(solution)
    # f_qos = fitness_qos_all(solution)
    f_qos = fitness_qos_super(solution)
    f_cost = fitness_cost(solution)
    return f_qos, f_cost,
