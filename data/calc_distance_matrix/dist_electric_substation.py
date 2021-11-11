'''
python see_sol_one.py malaga-city.ssv malaga-bike-neigh-pos.pdf da_aparcamientosBiciEMT-25830-MOD.csv facility_points.ssv malaga-p-median.sol N_padronbarrios_latlon_1.ssv
'''

import numpy as np
import gpxpy.geo
import csv

path = "/home/christian/2018/research/p-median/results/evolution/evo-sol/out/"
folders = ['e-1', 'e-w', 'e-p', 'r-1', 'r-w', 'r-p']


def load_map_ssv(filename):
    nodes = []
    arcs = []
    file = open(filename, "r")
    for line in file:
        data = line.split(" ")
        if data[0] == 'n':
            lat = float(data[2])
            lon = float(data[3].split('\n')[0])
            nodes.append({'id': int(data[1]), 'lat': lat, 'lon': lon})
        elif data[0] == 'a':
            node_from = int(data[2])
            node_to = int(data[3].split('\n')[0])
            arcs.append({'id': int(data[1]), 'from': node_from, 'to': node_to})
        else:
            print('line no valid')
    file.close()
    return nodes, arcs


# fp = np.genfromtxt('facility_points.ssv', delimiter=' ')

ss = np.genfromtxt('subestaciones.txt', delimiter=',')

print(len(ss))
print(ss[0])

nodes, arcs = load_map_ssv('malaga-city.ssv')

pertinencia = np.full((len(ss), len(arcs)), -1)

print(pertinencia)

def get_coordinates(arc, nodes):
    n_o = next(item for item in nodes if item["id"] == arc['from'])
    n_d = next(item for item in nodes if item["id"] == arc['to'])
    lat = (n_o['lat'] + n_d['lat']) / 2
    lon = (n_o['lon'] + n_d['lon']) / 2
    return lat, lon


for a in range(0, len(arcs)):
    lat, lon = get_coordinates(arcs[a], nodes)
    min_v = float("inf")
    index = -1
    for s in range(0, len(ss)):
        dist = gpxpy.geo.haversine_distance(lat, lon, ss[s][1], ss[s][2])
        if dist < min_v:
            index = s
            min_v = dist
    pertinencia[index][a] = min_v

np.savetxt("ExA.ssv", pertinencia, delimiter=" ")

print("=== END ===")
