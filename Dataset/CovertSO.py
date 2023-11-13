# from http://snap.stanford.edu/data/sx-stackoverflow.html

import os
import datetime

base_path = os.path.dirname(os.path.realpath(__file__)) + '/'
raw_data_c2a = base_path + '/sx-stackoverflow-c2a.txt'
raw_data_c2q = base_path + '/sx-stackoverflow-c2q.txt'
raw_data_a2q = base_path + '/sx-stackoverflow-a2q.txt'
data_list = [raw_data_c2q, raw_data_c2a, raw_data_a2q]
raw_data = base_path + '/sx-stackoverflow.txt'
result_path = base_path + 'result/'
# 2009 - 2016
# min_ts = 1217567877 2008年8月1日星期五下午1点17分
# max_ts = 1457273428 2016年3月6日星期日晚上10点10分

year_to_layer = {
    2008: 0,
    2009: 0,
    2010: 1,
    2011: 2,
    2012: 3,
    2013: 4,
    2014: 5,
    2015: 6,
    2016: 7,
}


def get_timestamp_range():
    min_ts = 0
    max_ts = 0
    with open(raw_data) as f:
        while True:
            line = f.readline()
            if not line:
                break

            u, v, ts = list(map(int, line.strip().split()))
            if min_ts == 0 or max_ts == 0:
                max_ts = ts
                min_ts = ts

            else:
                max_ts = max(ts, max_ts)
                min_ts = min(ts, min_ts)

    print(f"Max time : {max_ts}")
    print(f"Min time : {min_ts}")


def get_year(ts):
    a = datetime.datetime.utcfromtimestamp(ts)
    return a.year


def parse_dataset():
    n_layer = 24
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

    # layer-num, node-num, max-node-num
    # graph_file.write(f"{24} {2601978} {2601978}\n")

    file_id = 0
    for file_name in data_list:
        with open(file_name) as f:
            while True:
                line = f.readline()
                if not line:
                    break

                u, v, ts = list(map(int, line.strip().split()))
                year_id = year_to_layer[int(get_year(ts))]
                l = file_id + year_id * 3

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
        file_id += 1

    for i in range(n_layer):
        graph_files[i].close()

    with open(result_path + 'statistics.txt', 'w') as f:
        for i in range(n_layer):
            f.write('num_of_l' + str(i) + '_nodes:' + str(len(node_sets[i])) + '\n')
        for i in range(n_layer):
            f.write('num_of_l' + str(i) + '_edges:' + str(len(edge_sets[i])) + '\n')

    print(f"Total edge in three graphs: {ne}")
    print(f"Max node in three graphs: {max_node}")
    print(f"Max node-id in three graphs: {node_id}")
    print(f"Total edge in layered graphs: {ne_l}")


def combine_layer(combine_layers_n):
    w_file_name = result_path + f'so_{combine_layers_n}.txt'
    with open(w_file_name, "a") as w_f:
        w_f.write(f"{combine_layers_n} 2601978 2601978\n")
        for layer in range(combine_layers_n):
            read_file_name = result_path + 'layer_' + str(layer) + '.txt'
            with open(read_file_name, "r") as r_f:
                content = r_f.read()
                w_f.write(content)


# parse_dataset()
combine_layer(5)
