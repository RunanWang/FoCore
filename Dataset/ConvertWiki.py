# from http://snap.stanford.edu/data/wiki-talk-temporal.html

import os
import datetime

base_path = os.path.dirname(os.path.realpath(__file__)) + '/'
raw_data = base_path + '/wiki-talk-temporal.txt'
result_path = base_path + 'result/'
# min_ts = 1003638700 2001-10-21 12:31:40
# max_ts = 1199614728 2008-01-06 18:18:48
min_ts = 1003638700
max_ts = 1199614728
ts_bound = [1023236302.8, 1042833905.6, 1062431508.4, 1082029111.2, 1101626714.0, 1121224316.8, 1140821919.6,
            1160419522.4, 1180017125.2, 1199614728.0]


def get_layer_interval():
    n_layer = 10
    interval = (max_ts - min_ts) / n_layer
    int_list = [min_ts + i * interval for i in range(1, n_layer + 1)]
    print(int_list)


def get_year(ts):
    a = datetime.datetime.utcfromtimestamp(ts)
    return a.year


def parse_dataset():
    n_layer = 10
    ne = 0
    ne_l = 0
    node_sets = [set() for _ in range(n_layer)]
    edge_sets = [set() for _ in range(n_layer)]
    node_to_nodeid = {}
    node_id = 0
    max_node = 0

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    graph_files = []
    for i in range(n_layer):
        graph_files.append(open(result_path + 'layer_' + str(i) + '.txt', 'w'))

    file_name = raw_data
    with open(file_name) as f:
        while True:
            line = f.readline()
            if not line:
                break

            u, v, ts = list(map(int, line.strip().split()))
            for i, b in enumerate(ts_bound):
                if ts < b:
                    year_id = i
                    break
            l = year_id

            if u != v and (u, v) not in edge_sets[l] and (v, u) not in edge_sets[l]:
                if u not in node_to_nodeid:
                    node_id += 1
                    node_to_nodeid[u] = node_id
                    max_node = max(max_node, u)
                if v not in node_to_nodeid:
                    node_id += 1
                    node_to_nodeid[v] = node_id
                    max_node = max(max_node, v)
                graph_files[l].write(f'{l + 1} {node_to_nodeid[u]} {node_to_nodeid[v]}\n')
                node_sets[l].add(u)
                node_sets[l].add(v)
                edge_sets[l].add((u, v))
                ne_l += 1
            ne += 1

    for i in range(n_layer):
        graph_files[i].close()

    with open(result_path + 'statistics.txt', 'w') as f:
        for i in range(n_layer):
            f.write('num_of_l' + str(i) + '_nodes:' + str(len(node_sets[i])) + '\n')
        for i in range(n_layer):
            f.write('num_of_l' + str(i) + '_edges:' + str(len(edge_sets[i])) + '\n')

    print(f"Total edge in graph: {ne}")
    print(f"Max node in graph: {max_node}")
    print(f"Max node-id in three graphs: {node_id}")
    print(f"Total edge in layered graphs: {ne_l}")


def combine_layer(combine_layers_n):
    w_file_name = result_path + f'wiki_{combine_layers_n}.txt'
    with open(w_file_name, "a") as w_f:
        w_f.write(f"{combine_layers_n} 1094018 1094018\n")
        for layer in range(combine_layers_n):
            read_file_name = result_path + 'layer_' + str(layer) + '.txt'
            with open(read_file_name, "r") as r_f:
                content = r_f.read()
                w_f.write(content)


# parse_dataset()
# get_layer_interval()

combine_layer(10)
