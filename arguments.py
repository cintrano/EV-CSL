import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('C', type=int, help='number of clients')
parser.add_argument('L', type=int, help='number of possible locations')
parser.add_argument('E', type=int, help='numdest='F', ber of neighborhoodposition to aprox.')
parser.add_argument('matrix_file', nargs='?')
parser.add_argument('energy_matrix_file', nargs='?')
parser.add_argument('customer_file', nargs='?')
parser.add_argument('locations_file', nargs='?')
parser.add_argument('energy_stations_file', nargs='?')

# Problem parameters
parser.add_argument('--radius', dest='station_radius', type=float, help='max distance radius for charging stations', default=500)
# alpha, beta, gamma
parser.add_argument('--qos-alpha', dest='alpha', type=float, help='')
parser.add_argument('--qos-beta', dest='beta', type=float, help='')
parser.add_argument('--qos-gamma', dest='gamma', type=float, help='')


# Input Output files managements
parser.add_argument('--input', dest='in_path', help='path of the input files', default='')
parser.add_argument('--output', dest='out_path', help='path of the output files', default='')
parser.add_argument('--write-output', dest='write_files', type=bool, default=True, const=False, nargs='?', help='no create files')

# Ending process
parser.add_argument('--timer', type=int, dest='MAX_TIME', nargs='?', help='max time of the execution')
parser.add_argument('--iter', type=int, dest='iter', help='max time of the execution')

# Random seed
parser.add_argument('--seed', type=int, dest='seed', help='random seed')

# Algorithm
parser.add_argument('--algo', dest='algorithm', nargs='?', help='name of the algorithm')
parser.add_argument('--pop', type=int, dest='POP_SIZE', nargs='?', help='population size')

# Algorithm params
parser.add_argument('--neighborhood', type=str, dest='NEIGHBORHOOD_MODE', nargs='?', help='type of neighborhood')


parser.add_argument('--k', type=int, dest='k', nargs='?', help='')
parser.add_argument('--kmax', type=int, dest='kmax', nargs='?', help='')
parser.add_argument('--Kmayus', type=int, dest='Kmayus', nargs='?', help='')

parser.add_argument('--m', type=float, dest='m', nargs='?', help='')

parser.add_argument('--next', type=str, dest='VNS_next_opt', nargs='?', help='')

# GA
parser.add_argument('--ga-sel', type=str, dest='sel_mode', nargs='?', help='')
parser.add_argument('--ga-cross', type=str, dest='cross_mode', nargs='?', help='')
parser.add_argument('--ga-mut', type=str, dest='mut_mode', nargs='?', help='')
parser.add_argument('--ga-mut-prob', type=float, dest='mut_prob', nargs='?', help='')
parser.add_argument('--ga-repl', type=str, dest='repl_mode', nargs='?', help='')

parser.add_argument('--prob-cross', type=float, dest='prob_cross', nargs='?', help='probability of crossover')
parser.add_argument('--prob-mut', type=float, dest='prob_mut', nargs='?', help='probability of mutation')


args = parser.parse_args()
