import os
# Print the unweighted graph
import matplotlib.pyplot as plt
import networkx as nx


# Load the data
def load_map_ssv(filename='malaga-city.ssv'):
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


def add_stations(filename, G, col_map, val_map, shp_map):
    file = open(filename, "r")
    i = 0
    for line in file.readlines():
        data = line.split(" ")    # 20, 21
        idx = i
        lat = float(data[0])
        lon = float(data[1])
        col_map[idx] = 'blue'
        val_map[idx] = 300
        shp_map[idx] = 's'
        G.add_node(idx, pos=(lon, lat), color='blue', shape='s', size=300)
        i += 1
    file.close()


def read_facilities(filename):
    """Facilities (lat, lon) id is the same as the position in the list
    :param filename: Filename
    :return: list of (lat, lon)
    """
    out = []
    file = open(filename, "r")
    for line in file.readlines():
        data = line.split(" ")
        lat = float(data[1])
        lon = float(data[2])
        out.append((lat, lon))
    file.close()
    return out


def add_solution(G, col_map, val_map, facilities, indexes, color):
    cont = 0
    idn = 0
    for data in facilities:
        if idn in indexes:
            idx = "s" + str(idn)
            lat = float(data[0])
            lon = float(data[1])
            # print(lat, lon)
            col_map[idx] = color
            val_map[idx] = 700
            G.add_node(idx, pos=(lon, lat), color=color, shape='o', size=100)
            cont = cont + 1
            # print("Added", cont)
        idn = idn + 1


def print_graph(graph, filename, col_map, val_map, shp_map):
    fig = plt.figure()
    values = [col_map.get(n, 'black') for n in graph.nodes()]
    values_size = [val_map.get(n, 6) for n in graph.nodes()]
    values_shape = [shp_map.get(n, 'o') for n in graph.nodes()]
    #values_shape = ['v' for n in graph.nodes()]
    print(len(values_shape), len(values), len(values_size))
    # Rescale figure
    N = 8
    params = plt.gcf()
    plSize = params.get_size_inches()
    params.set_size_inches((plSize[0] * N, plSize[1] * N))
    # Remove axis
    plt.axis('off')
    pos = nx.get_node_attributes(graph, 'pos')
    colors = nx.get_node_attributes(graph, 'color')
    # Print
    for shape in set(values_shape):
        node_list = [node for node in graph.nodes() if graph.nodes[node]['shape'] == shape]
        values = [col_map.get(n, 'black') for n in graph.nodes() if graph.nodes[n]['shape'] == shape]
        values_size = [val_map.get(n, 6) for n in graph.nodes() if graph.nodes[n]['shape'] == shape]
        nx.draw_networkx_nodes(graph, pos, nodelist=node_list, node_size=values_size, node_color=values, node_shape=shape, alpha=0.7)
        #nx.draw_networkx_nodes(graph, pos)
    # nx.draw_networkx_labels(G,pos)
    nx.draw_networkx_edges(graph, pos, alpha=0.5)

    fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')#, pad_inches=-3)
    #plt.show()


def main(indexes, filename='map.pdf'):
    colors = ["orangered", "olivedrab", "darkgoldenrod",
              "lightseagreen", "darkorchid", "royalblue"]
    colors_algo = ["red", "green", "purple"]

    CITY = '/home/cintrano/PycharmProjects/location_problems/data/malaga-city.ssv'
    LOCATION_OF_FACILITIES = '/home/cintrano/PycharmProjects/location_problems/data/facility_points.ssv'
    SUBSTATIONS_LOCATIONS = '/home/cintrano/PycharmProjects/location_problems/data/substations_points.ssv'
    nodes, arcs = load_map_ssv(CITY)

    print(len(nodes))
    print(len(arcs))

    G = nx.Graph()  # G = nx.DiGraph()
    val_map = {}
    col_map = {}
    shp_map = {}

    add_stations(SUBSTATIONS_LOCATIONS, G, col_map, val_map, shp_map)

    facilities = read_facilities(LOCATION_OF_FACILITIES)

    add_solution(G, col_map, val_map, facilities, indexes, 'orange')

    for node in nodes:
        col_map[str(node['id'])] = 'black'
        val_map[str(node['id'])] = 1
        shp_map[str(node['id'])] = 'o'
        G.add_node('n' + str(node['id']), pos=(node['lon'], node['lat']), color='black', shape='.', size=1)

    for arc in arcs:
        # if next((item for item in nodes if item["id"] == arc["from"]), False):
        #     if next((item for item in nodes if item["id"] == arc["to"]), False):
        if arc['from'] != arc['to']:
            G.add_edge('n' + str(arc['from']), 'n' + str(arc['to']))

    print("G", G.number_of_nodes(), G.number_of_edges())

    print_graph(G, f'{os.path.dirname(__file__)}/maps/{filename}', col_map, val_map, shp_map)


def possible_locations(filename='locations.pdf'):
    colors = ["orangered", "olivedrab", "darkgoldenrod",
              "lightseagreen", "darkorchid", "royalblue"]
    colors_algo = ["red", "green", "purple"]

    CITY = '/home/cintrano/PycharmProjects/location_problems/data/malaga-city.ssv'
    LOCATION_OF_FACILITIES = '/home/cintrano/PycharmProjects/location_problems/data/facility_points.ssv'
    SUBSTATIONS_LOCATIONS = '/home/cintrano/PycharmProjects/location_problems/data/substations_points.ssv'
    nodes, arcs = load_map_ssv(CITY)

    print(len(nodes))
    print(len(arcs))

    G = nx.Graph()  # G = nx.DiGraph()
    val_map = {}
    col_map = {}
    shp_map = {}

    add_stations(SUBSTATIONS_LOCATIONS, G, col_map, val_map, shp_map)

    facilities = read_facilities(LOCATION_OF_FACILITIES)

    #add_solution(G, col_map, val_map, facilities, indexes, 'orange')

    for node in nodes:
        col_map[str(node['id'])] = 'black'
        val_map[str(node['id'])] = 1
        shp_map[str(node['id'])] = 'o'
        G.add_node('n' + str(node['id']), pos=(node['lon'], node['lat']), color='black', shape='.', size=1)

    for arc in arcs:
        # if next((item for item in nodes if item["id"] == arc["from"]), False):
        #     if next((item for item in nodes if item["id"] == arc["to"]), False):
        if arc['from'] != arc['to']:
            G.add_edge('n' + str(arc['from']), 'n' + str(arc['to']))
            n_from_lon = next((float(item['lon']) for item in nodes if item["id"] == arc["from"]), False)
            n_from_lat = next((float(item['lat']) for item in nodes if item["id"] == arc["from"]), False)
            n_to_lon = next((float(item['lon']) for item in nodes if item["id"] == arc["to"]), False)
            n_to_lat = next((float(item['lat']) for item in nodes if item["id"] == arc["to"]), False)
            if n_from_lat < 36.700872 and n_from_lon < -4.524534:
                pass
            else:
                lon = (n_from_lon + n_to_lon) / 2
                lat = (n_from_lat + n_to_lat) / 2
                col_map['pos' + str(arc['id'])] = 'green'
                val_map['pos' + str(arc['id'])] = 15
                shp_map['pos' + str(arc['id'])] = 'o'
                G.add_node('pos' + str(arc['id']), pos=(lon, lat), color='green', shape='o', size=15)

    print("G", G.number_of_nodes(), G.number_of_edges())

    print_graph(G, f'{os.path.dirname(__file__)}/{filename}', col_map, val_map, shp_map)  


if __name__ == '__main__':
    indexes = [764, 835, 1323, 2238, 2285, 3604, 3969, 4276, 4564, 4583, 4598, 4712, 4728, 4810, 6614, 7076, 7491, 8744,
               9287, 10255, 10867, 11294, 12394, 12956, 14111, 14510, 14661, 15164, 15302, 15337, 15463, 15644, 16270,
               16663, 18761, 18879, 20245, 20401, 20503, 21023, 24065, 24825, 24903, 25379, 26937, 27936, 28951, 30022,
               30741, 32951]
    # main(indexes)
    possible_locations()
'''
python see_sol2.py malaga-city.ssv mb2.eps da_aparcamientosBiciEMT-25830-MOD.csv facilities_points.ssv 
python see_sol_best.py malaga-city.ssv best-e-p.pdf da_aparcamientosBiciEMT-25830-MOD.csv facility_points.ssv e-p
'''
