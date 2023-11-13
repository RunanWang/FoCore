import copy
import heapq
import os.path
import pickle
from collections import deque
import constant as C
from os import stat
from Utils.log import Log
from Utils.Timer import Timer
from Utils.Metrics import density

L = Log(__name__).get_logger()


# Decomposition
def focore_decomposition(multilayer_graph, nodes_iterator, layers_iterator, dataset_name, save=False):
    """
    我们设计了众多不同的分解方法，但有着相同的输入与输出，这个函数包装了一层，用来选择具体的方法
    """
    focore_decomposition_vc(multilayer_graph, nodes_iterator, layers_iterator, dataset_name, save=save)


def check_diff(multilayer_graph, nodes_iterator, layers_iterator, dataset_name, save=False):
    """
    来检查不同方法的结果是否一致&比较各种方法的计算时间与peeling次数等
    """
    r0, t0 = focore_decomposition_vc(multilayer_graph, nodes_iterator, layers_iterator, dataset_name, False)
    fulfil_coreness(r0, nodes_iterator)
    r1, t1 = focore_decomposition_interleaved_peeling(multilayer_graph, nodes_iterator, layers_iterator, dataset_name,
                                                      False)
    fulfil_coreness(r1, nodes_iterator)
    L.info("Checking")
    flag = True
    for name, coreness in r1.items():
        try:
            coreness0 = r0[name]
        except:
            continue
        for i in range(0, len(coreness0)):
            if coreness[i] != coreness0[i]:
                L.debug(f"{name}: Node {i} not same. Baseline {coreness0[i]} vs. {coreness[i]}")
                flag = False
    if flag:
        L.info("Check Pass.")
    else:
        L.warn("Check Fail.")
    L.info(f"Baseline:")
    t0.print_timer()
    L.info(f"Pruning:")
    t1.print_timer()


# Decomposition-Comb-Utils
def gen_comb_name(comb):
    return f"focus{comb['focus_layers_comb']}-sup{comb['support_num']}"


def gen_c_name(focus_layer, sup_num):
    return f"focus{focus_layer}-sup{sup_num}"


def gen_possible_comb(layers_iterator, old_comb):
    """
    根据old-comb生成这个comb的所有枚举孩子节点(可对layer进行order)
    """
    focus_layers_comb_id = old_comb["focus_layers_comb_id"]
    support_num = old_comb["support_num"]
    combs = []
    layer_num = len(layers_iterator)
    if len(focus_layers_comb_id) == support_num:
        # max-comb
        if len(focus_layers_comb_id) == layer_num - 1 and focus_layers_comb_id[-1] + 1 == layer_num \
                and 0 not in focus_layers_comb_id:
            focus_layers_comb_new_id = copy.copy(focus_layers_comb_id)
            focus_layers_comb_new_id.append(0)
            focus_layers_comb_new = []
            for layer_no in focus_layers_comb_new_id:
                focus_layers_comb_new.append(layers_iterator[layer_no])
            focus_layers_comb_new.sort()
            combs.append({"focus_layers_comb_id": focus_layers_comb_new_id,
                          "focus_layers_comb": focus_layers_comb_new, "support_num": support_num + 1})
        # focus-layer数目等于sup数，没有可以枚举的组合了
        return combs
    if len(focus_layers_comb_id) == 0:
        # focus-layer为空，此时枚举sup+1和所有单个layer
        if support_num + 1 < layer_num:
            combs.append({"focus_layers_comb_id": [], "focus_layers_comb": [], "support_num": support_num + 1})
        for layer_id in range(layer_num):
            combs.append({"focus_layers_comb_id": [layer_id], "focus_layers_comb": [layers_iterator[layer_id]],
                          "support_num": support_num})
    else:
        # 其它情况，向focus中添加一个layer
        next_layer_start = focus_layers_comb_id[-1] + 1
        for next_layer_id in range(next_layer_start, layer_num):
            focus_layers_comb_new_id = copy.copy(focus_layers_comb_id)
            focus_layers_comb_new_id.append(next_layer_id)
            focus_layers_comb_new = []
            for layer_no in focus_layers_comb_new_id:
                focus_layers_comb_new.append(layers_iterator[layer_no])
            focus_layers_comb_new.sort()
            combs.append({"focus_layers_comb_id": focus_layers_comb_new_id,
                          "focus_layers_comb": focus_layers_comb_new, "support_num": support_num})
    return combs


def get_comb_parent(comb, layer_num):
    """
        返回comb的所有父亲节点
    """
    parent_comb = []
    focus_layers_comb = comb["focus_layers_comb"]
    support_num = comb["support_num"]
    if len(focus_layers_comb) == 0 and support_num == 1:
        return parent_comb
    elif len(focus_layers_comb) == layer_num:
        for mask_id in range(len(focus_layers_comb)):
            check_comb = []
            for i, k in enumerate(focus_layers_comb):
                if i != mask_id:
                    check_comb.append(k)
            check_name = gen_c_name(check_comb, support_num - 1)
            parent_comb.append(check_name)
        return parent_comb
    else:
        if len(focus_layers_comb) != support_num:
            check_support_num = support_num - 1
            check_name = gen_c_name(focus_layers_comb, check_support_num)
            parent_comb.append(check_name)
        for mask_id in range(len(focus_layers_comb)):
            check_comb = []
            for i, k in enumerate(focus_layers_comb):
                if i != mask_id:
                    check_comb.append(k)
            check_name = gen_c_name(check_comb, support_num)
            parent_comb.append(check_name)
        return parent_comb


def save_and_print(save, result, dataset_name, layers_iterator, node_iterator):
    index = covert_coreness_to_index(dataset_name, result, layers_iterator, save)
    distinct_core_num = 0
    for name, cv in index.items():
        distinct_core_num += len(cv)
    L.info(f"Comb number: {len(index)}")
    L.info(f"Distinct Core number: {distinct_core_num}")
    if save and C.INS_CORE_FULFIL:
        fulfil_coreness(result, node_iterator)
        out_path = C.OUTPUT_DIR / f"{dataset_name}_FoCore_decomposition.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        L.info(f"Size of coreness in file: {stat(out_path).st_size / 1024 / 1024} MB")


# Decomposition-by-comb (naive)
def check_comb_k_significant(nodes_iterator, skyline_degree, strict_skyline_degree):
    # 检查这个分解的结果是否有意义：
    # 如果对每个点，结果都和使用全部层得到的结果相同，那么则认为这个结果没有意义
    count = 0
    for node in nodes_iterator:
        if skyline_degree[node] != strict_skyline_degree[node]:
            count += 1
    if count > len(skyline_degree) * 0:
        return True
    else:
        return False


def check_comb_parent_significant(comb, result):
    """
    检查comb的父亲节点是否有意义，否则跳过计算
    """
    focus_layers_comb = comb["focus_layers_comb"]
    support_num = comb["support_num"]
    if len(focus_layers_comb) == 0 and support_num == 1:
        return True
    else:
        for mask_id in range(len(focus_layers_comb)):
            check_comb = []
            for i, k in enumerate(focus_layers_comb):
                if i != mask_id:
                    check_comb.append(k)
            check_name = gen_c_name(check_comb, support_num)
            if check_name not in result:
                return False
        if len(focus_layers_comb) != support_num:
            check_support_num = support_num - 1
            check_name = gen_c_name(focus_layers_comb, check_support_num)
            if check_name not in result:
                return False
        return True


def core_dec_for_given_comb(multilayer_graph, nodes_iterator, layers_iterator, focus_layers_comb, support_num):
    """
    给定了comb之后，计算这个comb下每个点的coreness值
    """
    L.info(f"-------------- focus = {focus_layers_comb}, background-num = {support_num} --------------")
    degree = [[0 for layer in layers_iterator] for node in nodes_iterator]  # 存子图上每层每个点的度数
    skyline_degree = [0 for node in nodes_iterator]  # 存子图top-lambda层上的度数
    layer_max_degree = [0 for layer in layers_iterator]  # 存每层的最大度数(建bucket用)
    node_peeled = [False for node in nodes_iterator]

    for node in nodes_iterator:
        for layer in layers_iterator:
            degree[node][layer] = len(multilayer_graph[node][layer])
            layer_max_degree[layer] = max(layer_max_degree[layer], degree[node][layer])
        skyline_degree[node] = heapq.nlargest(support_num, degree[node])[-1]

    # 确定peeling的起止点
    k_start = 0
    focus_k_max = 0
    for layer in layers_iterator:
        focus_k_max = max(focus_k_max, layer_max_degree[layer])
    skyline_k_max = max(skyline_degree)
    k_max = max(focus_k_max, skyline_k_max)
    del layer_max_degree

    # 建立peeling的bucket - skyline的和layer的
    skyline_bucket = [set() for i in range(k_max + 1)]
    layer_bucket = {}
    for layer in focus_layers_comb:
        layer_bucket[layer] = [set() for i in range(k_max + 1)]
    for node in nodes_iterator:
        skyline_bucket[skyline_degree[node]].add(node)
        for layer in focus_layers_comb:
            layer_bucket[layer][degree[node][layer]].add(node)

    # 进入peeling过程
    for k in range(k_start, k_max + 1):
        need_continue = True
        while need_continue:
            # 先在focus的指定层上peeling
            for focus_layer in focus_layers_comb:
                neighbors = set()
                # 在focus的每一层上peel去所有度数=k的点
                while layer_bucket[focus_layer][k]:
                    node = layer_bucket[focus_layer][k].pop()
                    if node_peeled[node]:
                        continue
                    node_peeled[node] = True
                    # 首先在每一层中移除node，先处理对邻居的影响
                    for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                        for neighbor in layer_neighbors:
                            # 由于node点被peel，导致node的度数>k的邻居 度数-1
                            if degree[neighbor][layer] > k:
                                # 在focus上的层需要额外维护bucket
                                if layer in focus_layers_comb:
                                    layer_bucket[layer][degree[neighbor][layer]].remove(neighbor)
                                    degree[neighbor][layer] -= 1
                                    layer_bucket[layer][degree[neighbor][layer]].add(neighbor)
                                else:
                                    degree[neighbor][layer] -= 1
                                # 由于度数变化，可能导致skyline-bucket需要维护，加入neighbors，之后统一检查
                                if degree[neighbor][layer] + 1 == skyline_degree[neighbor]:
                                    neighbors.add(neighbor)
                    # # node点已经不满足条件，并且在每层上都完成了peeling，需要在所有的bucket中移除
                    # for layer in focus_layers_comb:
                    #     # 排除的情况是这层上的点=k，被pop出，因而不用再remove
                    #     if layer != focus_layer:
                    #         layer_bucket[degree[node][layer]].remove(node)
                    # skyline_bucket[skyline_degree[node]].remove(node)
                    # 把所有层上node的度数赋为k，免得后续peel过程中再来维护
                    for layer in layers_iterator:
                        if degree[node][layer] > k:
                            degree[node][layer] = k
                    skyline_degree[node] = k
                # 完成后看看是否对skyline有影响
                for neighbor in neighbors:
                    try:
                        skyline_bucket[skyline_degree[neighbor]].remove(neighbor)
                    except KeyError:
                        # 排除的情况是这个node在focus-layer上被peel掉，已经移除出了skyline-bucket
                        continue
                    skyline_degree[neighbor] = heapq.nlargest(support_num, degree[neighbor])[-1]
                    skyline_bucket[max(skyline_degree[neighbor], k)].add(neighbor)

            # 然后在skyline上peeling
            while skyline_bucket[k]:
                node = skyline_bucket[k].pop()
                if node_peeled[node]:
                    continue
                node_peeled[node] = True
                skyline_degree[node] = k
                # 首先在每一层中移除node，先处理对邻居的影响
                neighbors = set()
                for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                    for neighbor in layer_neighbors:
                        # 由于node点被peel，导致node的度数>k的邻居 度数-1
                        if degree[neighbor][layer] > k:
                            if layer in focus_layers_comb:
                                # 在focus上的层需要额外维护bucket
                                layer_bucket[layer][degree[neighbor][layer]].remove(neighbor)
                                degree[neighbor][layer] -= 1
                                layer_bucket[layer][degree[neighbor][layer]].add(neighbor)
                            else:
                                degree[neighbor][layer] -= 1
                            # 由于度数变化，可能导致skyline-bucket需要维护，加入neighbors，之后统一检查
                            if degree[neighbor][layer] + 1 == skyline_degree[neighbor]:
                                neighbors.add(neighbor)
                # # node点已经不满足条件，并且在每层上都完成了peeling，需要在所有的bucket中移除
                # for focus_layer in focus_layers_comb:
                #     layer_bucket[focus_layer][degree[node][focus_layer]].remove(node)
                # 把所有层上node的度数赋为k，免得后续peel过程中再来维护
                for layer in layers_iterator:
                    if degree[node][layer] > k:
                        degree[node][layer] = k
                # 完成后看看是否对skyline有影响
                for neighbor in neighbors:
                    skyline_bucket[skyline_degree[neighbor]].remove(neighbor)
                    skyline_degree[neighbor] = heapq.nlargest(support_num, degree[neighbor])[-1]
                    skyline_bucket[max(skyline_degree[neighbor], k)].add(neighbor)

            need_continue = False
            for layer in focus_layers_comb:
                need_continue = need_continue or layer_bucket[layer][k]

    return skyline_degree


def focore_decomposition_enum_pruning(multilayer_graph, nodes_iterator, layers_iterator, dataset_name, save=False):
    """
        从初始comb开始，枚举有意义的comb，每个comb进行peeling
    """
    timer = Timer()
    timer.start()
    result = {}
    possible_comb = deque([{"focus_layers_comb_id": [], "focus_layers_comb": [], "support_num": 1}])
    layer_num = max(layers_iterator) + 1
    strict_skyline_degree = core_dec_for_given_comb(multilayer_graph, nodes_iterator, layers_iterator, [], layer_num)
    peel_num = 1

    while possible_comb:
        comb = possible_comb.popleft()
        # 首先检查这个comb的parent是否有意义，parent没有意义就没必要再进行分解了
        if check_comb_parent_significant(comb, result):
            focus_layers_comb = comb["focus_layers_comb"]
            support_num = comb["support_num"]
            skyline_degree = core_dec_for_given_comb(multilayer_graph, nodes_iterator, layers_iterator,
                                                     focus_layers_comb, support_num)
            peel_num += 1
            if check_comb_k_significant(nodes_iterator, skyline_degree, strict_skyline_degree):
                # 这个组合有意义，才存下来，并枚举基于这个组合的更多可能组合
                result_name = gen_c_name(focus_layers_comb, support_num)
                result[result_name] = copy.copy(skyline_degree)
                new_combs = gen_possible_comb(layers_iterator, comb)
                for new_comb in new_combs:
                    possible_comb.append(new_comb)
    result[gen_c_name([i for i in range(0, layer_num)], layer_num)] = copy.copy(strict_skyline_degree)

    L.info("FoCore Decomposition (EP) Phase: ")
    timer.stop()
    timer.print_timer()
    L.info(f"Peel number: {peel_num}")
    save_and_print(save, result, dataset_name, layers_iterator, nodes_iterator)
    return result, timer


# Decomposition-by-k (inter_core_decomposition_by_k_bfs_mem)
def peel_nodes_by_batch(to_peel_nodes, comb_degree, comb_skyline, multilayer_graph, sup_num, peeling_k, result=None):
    neighbors = set()
    for node in to_peel_nodes:
        if node in comb_degree:
            # 这个点已经被peel过了
            del comb_degree[node]
            del comb_skyline[node]
        if result is not None:
            result[node] = peeling_k
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in layer_neighbors:
                if neighbor not in comb_degree:
                    continue
                if comb_degree[neighbor][layer] > peeling_k:
                    comb_degree[neighbor][layer] -= 1
                    # 由于度数变化，可能导致skyline-bucket需要维护，加入neighbors，之后统一检查
                    if comb_degree[neighbor][layer] + 1 == comb_skyline[neighbor]:
                        neighbors.add(neighbor)
    for neighbor in neighbors:
        if neighbor not in comb_degree:
            continue
        comb_skyline[neighbor] = heapq.nlargest(sup_num, comb_degree[neighbor])[-1]


def peel_certain_k(peeling_k, comb_info, peeling_comb_name, degree, skyline_degree, multilayer_graph, layer_iter,
                   result):
    sup_num = comb_info[peeling_comb_name]['comb_sup']
    inter_layers = comb_info[peeling_comb_name]['comb_inter']
    father_combs = comb_info[peeling_comb_name]["father"]

    comb_degree = degree[peeling_comb_name]
    comb_skyline = skyline_degree[peeling_comb_name]

    # 第0轮的时候，需要建立degree和skyline
    if comb_degree is None and comb_skyline is None:
        degree[peeling_comb_name] = {}
        comb_degree = degree[peeling_comb_name]
        skyline_degree[peeling_comb_name] = {}
        comb_skyline = skyline_degree[peeling_comb_name]

        # 从所有father-comb中选择剩余节点最少的一个作为base
        base_comb_name = father_combs[-1]
        base_comb_remain_nodes = len(degree[base_comb_name])
        for father_comb_name in father_combs:
            if comb_info[father_comb_name]["comb_sup"] != sup_num:
                continue
            if len(degree[father_comb_name]) < base_comb_remain_nodes:
                base_comb_name = father_comb_name
                base_comb_remain_nodes = len(degree[father_comb_name])

        # 根据base复制一份comb自己的degree和skyline
        # print(f"{peeling_comb_name} Using skyline of {base_comb_name}")
        father_peeled_nodes = set()
        for node in degree[base_comb_name]:
            flag = False
            for father_comb_name in father_combs:
                if node not in degree[father_comb_name]:
                    father_peeled_nodes.add(node)
                    flag = True
                    break
            if flag:
                continue
            comb_degree[node] = [0 for layer in layer_iter]
            for layer in layer_iter:
                comb_degree[node][layer] = degree[base_comb_name][node][layer]

            if len(inter_layers) == 0:
                # 是firmCore，需要重新计算skyline
                comb_skyline[node] = heapq.nlargest(sup_num, comb_degree[node])[-1]
            else:
                comb_skyline[node] = skyline_degree[base_comb_name][node]

        if len(comb_degree) == 0:
            comb_name_queue = deque([peeling_comb_name])
            while comb_name_queue:
                comb_name = comb_name_queue.popleft()
                if not comb_info[comb_name]["finish"]:
                    comb_info[comb_name]["finish"] = True
                    if comb_name in result:
                        del result[comb_name]
                    for child_name in comb_info[comb_name]["sons"]:
                        comb_name_queue.append(child_name)

        result[peeling_comb_name] = {}
        # 在degree中peel相比base移除的点
        peel_nodes_by_batch(father_peeled_nodes, comb_degree, comb_skyline, multilayer_graph, sup_num, peeling_k)

    # 先peel祖先本轮k peel的那些点(if判断第一轮除外、root节点除外)
    if peeling_k != 0 and (sup_num != 1 or len(inter_layers) != 0):
        father_peeled_nodes = set()
        for node in comb_degree:
            for father_comb_name in father_combs:
                if node not in degree[father_comb_name]:
                    father_peeled_nodes.add(node)
                    break
        peel_nodes_by_batch(father_peeled_nodes, comb_degree, comb_skyline, multilayer_graph, sup_num, peeling_k,
                            result[peeling_comb_name])

    # 寻找初始阶段哪些点需要peel
    to_peel_nodes = set()
    # 建立peeling结构
    for node in comb_degree:
        # 不满足sup，需要peel
        if comb_skyline[node] <= peeling_k:
            to_peel_nodes.add(node)
            continue
        # 不满足inter，需要peel
        for layer in inter_layers:
            if comb_degree[node][layer] <= peeling_k:
                to_peel_nodes.add(node)
            continue

    # 进行peeling
    neighbors = set()  # 这个堆用来存skyline发生变化的点
    # 是因为peeling过程中可能需要频繁更新一个点的skyline，做nlargest消耗很大，所以我们不每次都更新skyline，而是先peel
    # 把可能受影响的点聚集在一起统一更新skyline
    while to_peel_nodes:
        # node 是本轮要处理的点，先标记结果
        node = to_peel_nodes.pop()
        del comb_degree[node]
        del comb_skyline[node]
        # node是本轮peel掉的，一定结果是k
        if peeling_k != 0:
            result[peeling_comb_name][node] = peeling_k
        # 首先在（focus指定的）每一层中移除node，并处理对邻居的影响
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in layer_neighbors:
                if neighbor not in comb_degree:
                    continue
                if comb_degree[neighbor][layer] > peeling_k:
                    # 由于node点被peel，导致node的度数>k的邻居 度数-1
                    comb_degree[neighbor][layer] -= 1
                    # 在focus上的层需要额外检查其是否满足对应层上的要求
                    if comb_degree[neighbor][layer] == peeling_k and layer in inter_layers:
                        to_peel_nodes.add(neighbor)
                    # 由于度数变化，可能导致skyline-bucket需要维护，加入neighbors，之后统一检查
                    if comb_degree[neighbor][layer] + 1 == comb_skyline[neighbor]:
                        neighbors.add(neighbor)
        # focus的点移除干净了，再来检查skyline
        if not to_peel_nodes:
            for neighbor in neighbors:
                if neighbor not in comb_degree:
                    # 已经被移除的neighbor不必再检查（可能是加入时还没peel，但之后peel掉了）
                    continue
                # 这步耗时最多，重新计算skyline，并把不满足要求的加入待peel列表
                new_skyline = heapq.nlargest(sup_num, comb_degree[neighbor])[-1]
                comb_skyline[neighbor] = new_skyline
                if new_skyline <= peeling_k:
                    to_peel_nodes.add(neighbor)
            neighbors = set()

    # 如果子图没peel光，进行孩子的peeling
    if len(comb_degree) == 0:
        comb_info[peeling_comb_name]["finish"] = True
        if peeling_k == 0:
            comb_name_queue = deque([peeling_comb_name])
            while comb_name_queue:
                comb_name = comb_name_queue.popleft()
                if not comb_info[comb_name]["finish"]:
                    comb_info[comb_name]["finish"] = True
                    if comb_name in result:
                        del result[comb_name]
                    for child_name in comb_info[comb_name]["sons"]:
                        comb_name_queue.append(child_name)


def focore_decomposition_interleaved_peeling(multilayer_graph, nodes_iterator, layers_iterator, dataset_name,
                                             save=False):
    """
    这个方法首先构建整个树，然后对每一个k值，从根结点到每一个叶节点一次剥掉点
    根据core的性质，若是父亲结点不满足k，那么孩子节点一定也不满足k值条件，因此这样似乎可以减少一些重复计算
    :return:
    """
    timer = Timer()
    timer.start()
    L.info(f"Layer Iter Order: {list(layers_iterator)}")
    layer_num = len(layers_iterator)
    degree = {node: [0 for layer in layers_iterator] for node in nodes_iterator}  # 存子图上每层每个点的度数
    skyline_degree = {node: 0 for node in nodes_iterator}  # 存子图top-lambda层上的度数
    max_degree = 0

    # 初始化每个点的邻居点数目
    for node in nodes_iterator:
        for layer in layers_iterator:
            degree[node][layer] = len(multilayer_graph[node][layer])
        skyline_degree[node] = max(degree[node])
        max_degree = max(max_degree, skyline_degree[node])

    # O(V 2^L)的空间，key为comb-name
    result = {}  # 在comb下的coreness
    comb_info = {}  # 父子信息

    # 生成全部可能的comb
    root_comb = {"focus_layers_comb_id": [], "focus_layers_comb": [], "support_num": 1}
    root_comb_name = gen_comb_name(root_comb)
    possible_comb = deque([root_comb])  # 暂存还没有扩展的comb
    comb_order = deque([root_comb_name])
    comb_info[root_comb_name] = {"sons": [], "comb_sup": root_comb['support_num'], "father": [],
                                 "comb_inter": root_comb['focus_layers_comb'], "finish": False}
    degree = {root_comb_name: degree}
    skyline_degree = {root_comb_name: skyline_degree}
    result[root_comb_name] = {}
    while possible_comb:
        comb = possible_comb.popleft()
        comb_name = gen_comb_name(comb)
        # 枚举comb下的新comb
        new_combs = gen_possible_comb(layers_iterator, comb)
        for new_comb in new_combs:
            new_comb_name = gen_comb_name(new_comb)
            result[new_comb_name] = []
            degree[new_comb_name] = None
            skyline_degree[new_comb_name] = None
            comb_info[new_comb_name] = {"sons": [], "father": get_comb_parent(new_comb, layer_num),
                                        "comb_sup": new_comb['support_num'],
                                        "comb_inter": new_comb['focus_layers_comb'], "finish": False}
            possible_comb.append(new_comb)
            comb_order.append(new_comb_name)
            for father_comb_name in comb_info[new_comb_name]["father"]:
                comb_info[father_comb_name]["sons"].append(new_comb_name)

    # 进入peeling
    for peeling_k in range(0, max_degree + 1):
        L.info(f"Peeling for k={peeling_k}")
        if comb_info[root_comb_name]["finish"]:
            break
        for comb_name in comb_order:
            if comb_info[comb_name]["finish"]:
                continue
            peel_certain_k(peeling_k, comb_info, comb_name, degree, skyline_degree, multilayer_graph, layers_iterator,
                           result)

    L.info("FoCore Decomposition (IP) Phase: ")
    timer.stop()
    timer.print_timer()
    # fulfil_coreness(result, nodes_iterator)
    save_and_print(save, result, dataset_name, layers_iterator, nodes_iterator)
    return result, timer


# Decomposition-vertex-centric-mem (inter_core_decomposition_vc_mem)
def core_dec_for_root(multilayer_graph, nodes_iterator, layers_iterator):
    """
    给定了comb之后，计算这个comb下每个点的coreness值
    """
    L.info("Peeling for root comb focus[]-sup1")
    degree = [[0 for layer in layers_iterator] for node in nodes_iterator]  # 存子图上每层每个点的度数
    skyline_degree = [0 for node in nodes_iterator]  # 存子图top-lambda层上的度数
    layer_max_degree = [0 for layer in layers_iterator]  # 存每层的最大度数(建bucket用)
    node_peeled = [False for node in nodes_iterator]

    for node in nodes_iterator:
        for layer in layers_iterator:
            degree[node][layer] = len(multilayer_graph[node][layer])
            layer_max_degree[layer] = max(layer_max_degree[layer], degree[node][layer])
        skyline_degree[node] = max(degree[node])

    # 确定peeling的起止点
    k_start = 0
    focus_k_max = 0
    for layer in layers_iterator:
        focus_k_max = max(focus_k_max, layer_max_degree[layer])
    skyline_k_max = max(skyline_degree)
    k_max = max(focus_k_max, skyline_k_max)
    del layer_max_degree

    # 建立peeling的bucket - skyline的和layer的
    skyline_bucket = [set() for i in range(k_max + 1)]
    for node in nodes_iterator:
        skyline_bucket[skyline_degree[node]].add(node)

    # 进入peeling过程
    for k in range(k_start, k_max + 1):
        # 然后在skyline上peeling
        while skyline_bucket[k]:
            node = skyline_bucket[k].pop()
            if node_peeled[node]:
                continue
            node_peeled[node] = True
            skyline_degree[node] = k
            # 首先在每一层中移除node，先处理对邻居的影响
            neighbors = set()
            for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                for neighbor in layer_neighbors:
                    # 由于node点被peel，导致node的度数>k的邻居 度数-1
                    if degree[neighbor][layer] > k:
                        degree[neighbor][layer] -= 1
                        # 由于度数变化，可能导致skyline-bucket需要维护，加入neighbors，之后统一检查
                        if degree[neighbor][layer] + 1 == skyline_degree[neighbor]:
                            neighbors.add(neighbor)
            # 把所有层上node的度数赋为k，免得后续peel过程中再来维护
            for layer in layers_iterator:
                if degree[node][layer] > k:
                    degree[node][layer] = k
            # 完成后看看是否对skyline有影响
            for neighbor in neighbors:
                skyline_bucket[skyline_degree[neighbor]].remove(neighbor)
                skyline_degree[neighbor] = max(degree[neighbor])
                skyline_bucket[max(skyline_degree[neighbor], k)].add(neighbor)
    return skyline_degree


def peel_for_certain_comb_vc(multilayer_graph, layers_iterator, coreness, comb, sup_buf):
    L.info(f"Calculating for {gen_comb_name(comb)}")

    focus_layers_comb = comb["focus_layers_comb"]
    extra_support_layer_num = comb["support_num"] - len(focus_layers_comb)
    extra_layers = []
    for layer in layers_iterator:
        if layer not in focus_layers_comb:
            extra_layers.append(layer)
    layers = focus_layers_comb
    if extra_support_layer_num > 0:
        layers = layers_iterator
    active_nodes = set()
    # 初始化
    for node, node_coreness in coreness.items():
        if node_coreness == 0:
            continue
        for layer in layers:
            sup_buf[node][layer][node_coreness] = 0
            iter_c = node_coreness
            while iter_c >= 0:
                sup_buf[node][layer][iter_c] = 0
                iter_c -= 1
            # 这是一个类似bucket的结构，每个点每层上存着满足sup的点，和不满足sup的点对应的当前coreness
            for neighbor in multilayer_graph[node][layer]:
                if neighbor not in coreness:
                    continue
                neighbor_coreness = coreness[neighbor]
                if neighbor_coreness == 0:
                    continue
                if neighbor_coreness >= node_coreness:
                    sup_buf[node][layer][node_coreness] += 1
                else:
                    sup_buf[node][layer][neighbor_coreness] += 1

        # 先处理在focus-layer中的层，这些层上要求都是sup>k,否则要标记为active
        for layer in focus_layers_comb:
            if sup_buf[node][layer][node_coreness] < node_coreness:
                active_nodes.add(node)
                break

        # 统计sup-layer是否满足要求，否则要标记为active
        if extra_support_layer_num > 0 and node not in active_nodes:
            satisfy_layer_num = 0
            for layer in layers_iterator:
                if sup_buf[node][layer][node_coreness] >= node_coreness:
                    satisfy_layer_num += 1
            if satisfy_layer_num < comb["support_num"]:
                active_nodes.add(node)

    # 迭代至收敛
    while active_nodes:
        # 计算新值并更新数据结构
        n_set = set()
        while active_nodes:
            node = active_nodes.pop()
            old_node_coreness = coreness[node]
            # meter["t_inter"].start()
            # 检查这些active点在focus-layer上是否满足要求
            for layer in focus_layers_comb:
                node_coreness = coreness[node]
                while sup_buf[node][layer][node_coreness] < node_coreness:
                    node_sup_num = sup_buf[node][layer][node_coreness]
                    node_coreness -= 1
                    # print(f"{node_coreness}:{sup_buf[node][layer][node_coreness]}")
                    sup_buf[node][layer][node_coreness] += node_sup_num
                # 把剩余层上的数据结构也进行更新
                for other_layer in layers:
                    if other_layer != layer:
                        iter_c = coreness[node]
                        while iter_c != node_coreness:
                            node_sup_num = sup_buf[node][other_layer][iter_c]
                            iter_c -= 1
                            sup_buf[node][other_layer][iter_c] += node_sup_num
                coreness[node] = node_coreness
            # meter["t_inter"].stop()

            # meter["t_firm"].start()
            # 检查这些active点是否有extra_support_layer_num层上满足要求
            node_coreness = coreness[node]
            if extra_support_layer_num > 0:
                # 检查是否满足sup-layer条件
                satisfy_layer_num = 0
                for layer in extra_layers:
                    if sup_buf[node][layer][node_coreness] >= node_coreness:
                        satisfy_layer_num += 1
                # 若不满足，则coreness--，更新数据结构，并再次检查
                node_coreness = coreness[node]
                while satisfy_layer_num < extra_support_layer_num:
                    node_coreness -= 1
                    for layer in layers_iterator:
                        sup_buf[node][layer][node_coreness] += sup_buf[node][layer][node_coreness + 1]
                    satisfy_layer_num = 0
                    # 再次检查是否满足sup-layer条件
                    for layer in extra_layers:
                        if sup_buf[node][layer][node_coreness] >= node_coreness:
                            satisfy_layer_num += 1
                coreness[node] = node_coreness
            # meter["t_firm"].stop()

            # 这个点更新了，需要更新邻居的数据结构，并检查是否标记邻居为active
            # meter["t_buf"].start()
            if old_node_coreness != coreness[node]:
                now_node_coreness = coreness[node]
                # 更新邻居的数据结构
                for layer in layers:
                    for neighbor in multilayer_graph[node][layer]:
                        if neighbor not in coreness:
                            continue
                        neighbor_coreness = coreness[neighbor]
                        old_neighbor_c = neighbor_coreness if neighbor_coreness <= old_node_coreness else old_node_coreness
                        now_neighbor_c = neighbor_coreness if neighbor_coreness <= now_node_coreness else now_node_coreness
                        sup_buf[neighbor][layer][old_neighbor_c] -= 1
                        sup_buf[neighbor][layer][now_neighbor_c] += 1
                        n_set.add(neighbor)
            # meter["t_buf"].stop()
            if coreness[node] == 0:
                del coreness[node]

        # 更新active信息
        for neighbor in n_set:
            if neighbor in active_nodes:
                continue
            if neighbor not in coreness:
                continue
            neighbor_coreness = coreness[neighbor]
            if neighbor_coreness == 0:
                continue
            # 先处理在focus-layer中的层，这些层上要求都是sup>k,否则要标记为active
            for layer in focus_layers_comb:
                if sup_buf[neighbor][layer][neighbor_coreness] < neighbor_coreness:
                    active_nodes.add(neighbor)
                    continue
            # 统计sup-layer是否满足要求，否则要标记为active
            if extra_support_layer_num > 0:
                satisfy_layer_num = 0
                for layer in extra_layers:
                    if sup_buf[neighbor][layer][neighbor_coreness] >= neighbor_coreness:
                        satisfy_layer_num += 1
                if satisfy_layer_num < extra_support_layer_num:
                    active_nodes.add(neighbor)

    return


def focore_decomposition_vc(multilayer_graph, nodes_iterator, layers_iterator, dataset_name, save=False):
    timer = Timer()
    # meter = {"t_prepare": AccumulateTimer(), "t_inter": AccumulateTimer(), "t_firm": AccumulateTimer(),
    #          "t_active": AccumulateTimer(), "t_up": AccumulateTimer(), "t_buf": AccumulateTimer(),
    #          "t_copy": AccumulateTimer()}
    timer.start()
    L.info(f"Layer Iter Order: {list(layers_iterator)}")

    # 下面主要是生成skyline作为root-comb的初始coreness值
    degree = [[0 for layer in layers_iterator] for node in nodes_iterator]  # 存子图上每层每个点的度数
    skyline_degree = [0 for node in nodes_iterator]  # 存子图top-lambda层上的度数
    for node in nodes_iterator:
        for layer in layers_iterator:
            degree[node][layer] = len(multilayer_graph[node][layer])
        skyline_degree[node] = max(degree[node])

    # O(V 2^L)的空间，key为comb-name
    result = {}  # 在comb下的coreness
    root_comb = {"focus_layers_comb_id": [], "focus_layers_comb": [], "support_num": 1}
    possible_comb = deque([root_comb])  # 暂存还没有扩展的comb
    sup_buf = []

    while possible_comb:
        comb = possible_comb.popleft()
        comb_name = gen_comb_name(comb)
        # meter["t_up"].start()
        # 下面确定vertex-centric的起始coreness值
        comb_fathers = get_comb_parent(comb, len(layers_iterator))
        to_del = set()
        if len(comb_fathers) != 0:
            # 说明非root-comb，起始coreness为所有父亲coreness（已迭代收敛）中的最小值
            for father_name in comb_fathers:
                if father_name not in result:
                    # 这个特殊情况出现在一些comb没被生成过
                    continue
                if comb_name not in result:
                    # 初始化copy第一个父亲的coreness值
                    result[comb_name] = copy.copy(result[father_name])
                else:
                    # 后面求min
                    # meter["t_copy"].start()
                    for node, node_c in result[comb_name].items():
                        if node not in result[father_name] or result[father_name][node] == 0:
                            to_del.add(node)
                        else:
                            result[comb_name][node] = min(node_c, result[father_name][node])
                    # meter["t_copy"].stop()
            for to_del_node in to_del:
                del result[comb_name][to_del_node]
            # vertex-centric 来计算这个comb的准确coreness
            if len(result[comb_name]) != 0:
                peel_for_certain_comb_vc(multilayer_graph, layers_iterator, result[comb_name], comb, sup_buf)
        else:
            # root-comb使用skyline作为起始coreness值
            root_coreness = core_dec_for_root(multilayer_graph, nodes_iterator, layers_iterator)
            sup_buf = [[[0 for c in range(0, root_coreness[node] + 1)] for layer in layers_iterator] for node in
                       nodes_iterator]
            result[comb_name] = {}
            for node in nodes_iterator:
                if root_coreness[node] != 0:
                    result[comb_name][node] = root_coreness[node]

        if len(result[comb_name]) == 0:
            continue

        # 枚举comb下的新comb
        new_combs = gen_possible_comb(layers_iterator, comb)
        for new_comb in new_combs:
            possible_comb.append(new_comb)
        # meter["t_up"].stop()

    L.info("FoCore Decomposition (VC) Phase: ")
    timer.stop()
    timer.print_timer()
    # fulfil_coreness(result, nodes_iterator)
    save_and_print(save, result, dataset_name, layers_iterator, nodes_iterator)
    # for name, certain_meter in meter.items():
    #     L.info(f"Meter Name: {name}, {certain_meter.to_str()}.")
    return result, timer


# Processing-Decomposition-Result
def fulfil_coreness(coreness, node_iterator):
    for comb_name, comb_coreness in coreness.items():
        list_coreness = [0 for node in node_iterator]
        for node, c in comb_coreness.items():
            list_coreness[node] = c
        coreness[comb_name] = list_coreness


# index
def covert_coreness_to_index(dataset_name, coreness, layers_iterator, save=False):
    index = {}
    L.info("Start converting coreness into index.")
    # L.info(f"Size of full coreness: {asizeof(coreness) / 1024 / 1024} MB")

    # 删除coreness全为0的comb
    to_del = []
    for comb_name, c in coreness.items():
        if len(c) == 0:
            to_del.append(comb_name)
    for comb_name in to_del:
        del coreness[comb_name]

    root_comb = {"focus_layers_comb_id": [], "focus_layers_comb": [], "support_num": 1}
    root_comb_name = gen_comb_name(root_comb)
    possible_comb = deque([root_comb])  # 暂存还没有扩展的comb

    index[root_comb_name] = {}

    to_del = []
    while possible_comb:
        comb = possible_comb.popleft()
        comb_name = gen_comb_name(comb)
        # 枚举comb下的新comb
        new_combs = gen_possible_comb(layers_iterator, comb)
        for new_comb in new_combs:
            new_comb_name = gen_comb_name(new_comb)
            if new_comb_name not in coreness:
                continue
            index[new_comb_name] = {}
            possible_comb.append(new_comb)

        # 形成index
        cores_to_nodes = index[comb_name]
        # 是list存的（—naive方法或fulfil过-）
        if type(coreness[comb_name]) is list:
            for node, c in enumerate(coreness[comb_name]):
                if c == 0:
                    continue
                # 实测并没有很大的改进，并且会严重增加查询时的时间，故放弃
                # for son_comb in index[comb_name]["sons"]:
                #     if coreness[son_comb][node] == c:
                #         continue
                if c not in cores_to_nodes:
                    cores_to_nodes[c] = [node]
                else:
                    cores_to_nodes[c].append(node)
        # 是dict存的（byk or vc）
        elif type(coreness[comb_name]) is dict:
            for node, c in coreness[comb_name].items():
                if c == 0:
                    continue
                # 实测并没有很大的改进，并且会严重增加查询时的时间，故放弃
                # for son_comb in index[comb_name]["sons"]:
                #     if coreness[son_comb][node] == c:
                #         continue
                if c not in cores_to_nodes:
                    cores_to_nodes[c] = [node]
                else:
                    cores_to_nodes[c].append(node)

        if len(cores_to_nodes) == 0:
            to_del.append(comb_name)
        for c, node_list in cores_to_nodes.items():
            node_list.sort()

    for zero_comb in to_del:
        del index[zero_comb]
    # L.info(f"Size of index in mem: {asizeof(index) / 1024 / 1024} MB")

    if save:
        out_path = C.OUTPUT_DIR / f"{dataset_name}_FoCore_index.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(index, f)
        L.info(f"Size of index in file: {stat(out_path).st_size / 1024 / 1024} MB")
        out_path = C.OUTPUT_DIR / f"{dataset_name}_FoCore_index.txt"
        with open(out_path, "w") as f:
            for name in index:
                core = index[name]
                f.write(f"{name}\n")
                for c in sorted(core):
                    nodes = core[c]
                    f.write(f"coreness={c}, nodes: {nodes}\n")
                f.write("\n")
    return index


# analyze
def analyze_core(multilayer_graph, dataset_name):
    index_file_path = C.OUTPUT_DIR / f"{dataset_name}_FoCore_index.pkl"
    if not os.path.exists(index_file_path):
        focore_decomposition(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator,
                             multilayer_graph.layers_iterator, dataset_name, save=True)
    with open(index_file_path, 'rb') as f:
        index = pickle.load(f)

    core_num_stat = []
    for name, core in index.items():
        sup_num = int(name.split("p")[-1])
        inter_layer_str_list = name.split("[")[-1].split("]")[0].split(", ")
        inter_layer = []
        for s in inter_layer_str_list:
            if s != "":
                inter_layer.append(int(s))

        inter_num = len(inter_layer)
        c_to_num = {}
        max_c = 0
        for c in sorted(core, reverse=True):
            nodes = core[c]
            c_to_num[c] = len(nodes)
            # L.info(f"core-size = {len(nodes)}, c = {c}")
            if c > max_c:
                max_c = c

        max_size = c_to_num[max_c]

        i = max_c
        while i > 0:
            if i - 1 not in c_to_num:
                c_to_num[i - 1] = c_to_num[i]
            else:
                c_to_num[i - 1] = c_to_num[i] + c_to_num[i - 1]
            i -= 1
        # L.info(f"sum = {sum(c_to_num)}, c = {max_c}")
        total_core_size = 0
        for c, core_size in c_to_num.items():
            total_core_size += core_size
        avg_size = total_core_size / max_c

        subgraph_node_num = 0
        subgraph_edge_num = {}
        best_density = 0
        in_graph_nodes = set()
        for layer in multilayer_graph.layers_iterator:
            subgraph_edge_num[layer] = 0
        for c in sorted(core, reverse=True):
            nodes = core[c]
            subgraph_node_num += len(nodes)
            for node in nodes:
                in_graph_nodes.add(node)
                for layer, layer_neighbors in enumerate(multilayer_graph.adjacency_list[node]):
                    for neighbor in layer_neighbors:
                        if neighbor in in_graph_nodes:
                            subgraph_edge_num[layer] += 1
            k_density = density(subgraph_node_num, subgraph_edge_num, 0)[0]
            if k_density > best_density:
                best_density = k_density

        cat_name = f"s{sup_num}-i{inter_num}"
        info = {"sup_num": sup_num, "inter_num": inter_num, "name": name, "cat": cat_name,
                "core_num": max_c, "max_core_size": max_size, "avg_core_size": avg_size, "density": best_density}
        L.info(info)
        core_num_stat.append(info)
    print(core_num_stat)


# denest graph
def focore_denest_graph(multilayer_graph, dataset_name):
    index_file_path = C.OUTPUT_DIR / f"{dataset_name}_FoCore_index.pkl"
    if not os.path.exists(index_file_path):
        focore_decomposition(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator,
                             multilayer_graph.layers_iterator, dataset_name, save=True)
    with open(index_file_path, 'rb') as f:
        index = pickle.load(f)

    beta_list = [0, 0.5, 1, 2, 3, 4, 5, 6]
    best_name = {}
    best_density = {}
    best_subgraph_size = {}
    best_selected_layer_num = {}
    best_min_avg_degree = {}
    firm_best_name = {}
    firm_best_density = {}
    cc_best_name = {}
    cc_best_density = {}
    for beta in beta_list:
        best_name[beta] = ""
        best_density[beta] = -1
        firm_best_name[beta] = ""
        firm_best_density[beta] = -1
        cc_best_name[beta] = ""
        cc_best_density[beta] = -1
    for name, core in index.items():
        subgraph_node_num = 0
        subgraph_edge_num = {}
        in_graph_nodes = set()
        for layer in multilayer_graph.layers_iterator:
            subgraph_edge_num[layer] = 0
        for c in sorted(core, reverse=True):
            nodes = core[c]
            subgraph_node_num += len(nodes)
            for node in nodes:
                in_graph_nodes.add(node)
                for layer, layer_neighbors in enumerate(multilayer_graph.adjacency_list[node]):
                    for neighbor in layer_neighbors:
                        if neighbor in in_graph_nodes:
                            subgraph_edge_num[layer] += 1
            for beta in beta_list:
                k_density, selected_layers, avg_degrees_per_layer = density(subgraph_node_num, subgraph_edge_num, beta)
                selected_layer_num = len(selected_layers)
                min_avg_degree = max(avg_degrees_per_layer)
                for layer in selected_layers:
                    if avg_degrees_per_layer[layer] < min_avg_degree:
                        min_avg_degree = avg_degrees_per_layer[layer]

                if name.split("]")[0].split("[")[1] == "":
                    inter_num = 0
                else:
                    inter_num = len(name.split("]")[0].split("[")[1].split(","))
                sup_num = int(name.split("sup")[1])
                if inter_num == 0:
                    if k_density > firm_best_density[beta]:
                        firm_best_density[beta] = k_density
                        firm_best_name[beta] = name + f"_k={c}"
                if inter_num == sup_num:
                    if k_density > cc_best_density[beta]:
                        cc_best_density[beta] = k_density
                        cc_best_name[beta] = name + f"_k={c}"
                if k_density > best_density[beta]:
                    best_density[beta] = k_density
                    best_min_avg_degree[beta] = min_avg_degree
                    best_selected_layer_num[beta] = selected_layer_num
                    best_name[beta] = name + f"_k={c}"
                    best_subgraph_size[beta] = subgraph_node_num

    L.info("FoCore:")
    L.info(best_density)
    L.info(best_name)
    L.info(best_subgraph_size)
    L.info(f"best_min_avg_degree: {best_min_avg_degree}")
    L.info(f"best_selected_layer_num: {best_selected_layer_num}")
    L.info("CCore:")
    L.info(cc_best_density)
    L.info(cc_best_name)
    L.info("FirmCore:")
    L.info(firm_best_density)
    L.info(firm_best_name)


# core-den
def core_den(multilayer_graph, dataset_name):
    index_file_path = C.OUTPUT_DIR / f"{dataset_name}_FoCore_index.pkl"
    if not os.path.exists(index_file_path):
        focore_decomposition(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator,
                             multilayer_graph.layers_iterator, dataset_name, save=True)
    with open(index_file_path, 'rb') as f:
        index = pickle.load(f)

    core_num_stat = []
    for name, core in index.items():
        subgraph_node_num = 0
        subgraph_edge_num = {}
        in_graph_nodes = set()
        for layer in multilayer_graph.layers_iterator:
            subgraph_edge_num[layer] = 0
        for c in sorted(core, reverse=True):
            nodes = core[c]
            subgraph_node_num += len(nodes)
            for node in nodes:
                in_graph_nodes.add(node)
                for layer, layer_neighbors in enumerate(multilayer_graph.adjacency_list[node]):
                    for neighbor in layer_neighbors:
                        if neighbor in in_graph_nodes:
                            subgraph_edge_num[layer] += 1
            density_by_layer = []
            for layer in multilayer_graph.layers_iterator:
                density_by_layer.append(round(subgraph_edge_num[layer] / subgraph_node_num, 2))
            if c == 6:
                avg = round(sum(density_by_layer) / len(multilayer_graph.layers_iterator), 2)
                print(f"Density of {name}-{c}-FoCore: {density_by_layer}, {avg}")
