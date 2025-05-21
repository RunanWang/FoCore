import copy
import heapq
import os.path
import pickle
from collections import deque, Counter
import math as math
from math import degrees

import constant as C
from Utils.log import Log
from Utils.Timer import Timer
from Algorithm.FoCore import focore_decomposition_interleaved_peeling

L = Log(__name__).get_logger()


class Combo(object):
    def __init__(self, focus=None, lamb: int = None, k: int = None, to_copy=None, name: str = None):
        self.f = []
        self.l = 1
        self.k = 1
        if focus is not None:
            self.f = focus
            self.l = lamb
            self.k = k
        elif to_copy is not None:
            self.f = [ff for ff in to_copy.f]
            self.l = to_copy.l
            self.k = to_copy.k
        elif name is not None:
            sp = name.split("-")
            fs = sp[0].split("&")
            if len(fs) == 1 and fs[0] == "":
                self.f = []
            else:
                self.f = [int(ff) for ff in fs]
            self.l = int(sp[1])
            self.k = int(sp[2])

    def __str__(self):
        sort_f = copy.copy(self.f)
        sort_f.sort()
        return f"{'&'.join([str(ff) for ff in sort_f])}-{self.l}-{self.k}"

    def get_lattice(self):
        return self.k + self.l - 2 + len(self.f)

    def get_intersect(self, other):
        ans = Combo(focus=[], lamb=1, k=1)
        ans.l = max(self.l, other.l)
        ans.k = max(self.k, other.k)
        f = set()
        for l in self.f:
            f.add(l)
        for l in other.f:
            f.add(l)
        ans.f = [l for l in f]
        return str(ans)

    def fl_str(self):
        sort_f = copy.copy(self.f)
        sort_f.sort()
        return f"{'&'.join([str(ff) for ff in sort_f])}-{self.l}"

    def f_str(self):
        sort_f = copy.copy(self.f)
        sort_f.sort()
        return f"{'&'.join([str(ff) for ff in sort_f])}"

    def is_son(self, father_combo):
        if father_combo.l < self.l and father_combo.k < self.k:
            for l in father_combo.f:
                if l not in self.f:
                    return False
            return True
        else:
            return False

    def get_one_hop_father_names(self):
        fathers = []
        if self.k > 1:
            c = Combo(to_copy=self)
            c.k = c.k - 1
            fathers.append(str(c))
        if self.l > len(self.f) and self.l > 1:
            c = Combo(to_copy=self)
            c.l = c.l - 1
            fathers.append(str(c))
        if len(self.f) > 0:
            for i in range(len(self.f)):
                c = Combo(to_copy=self)
                c.f = self.f[:i] + self.f[i + 1:]
                fathers.append(str(c))
        return fathers

    def get_one_hop_sons_name(self, layers_iter):
        sons = []
        c = Combo(to_copy=self)
        c.k = c.k + 1
        sons.append(str(c))
        if self.l < len(layers_iter) - 1:
            c = Combo(to_copy=self)
            c.l = c.l + 1
            sons.append(str(c))
        if len(self.f) < self.l:
            possible_layer = []
            for layer in layers_iter:
                if layer not in self.f:
                    possible_layer.append(layer)
            for li in possible_layer:
                c = Combo(to_copy=self)
                c.f = c.f + [li]
                sons.append(str(c))
        return sons

    def enum_one_hop_son_combos(self, layers_iter):
        sons = []
        if len(self.f) == 0:
            if self.l == 1:
                c = Combo(to_copy=self)
                c.k = c.k + 1
                sons.append(c)
            if self.l < len(layers_iter) - 1:
                c = Combo(to_copy=self)
                c.l = c.l + 1
                sons.append(c)
        if len(self.f) < self.l:
            max_index = -1
            if len(self.f) != 0:
                max_index = max(layers_iter.index(layer) for layer in self.f)
            for li in range(max_index + 1, len(layers_iter)):
                c = Combo(to_copy=self)
                c.f = c.f + [layers_iter[li]]
                sons.append(c)
        return sons

    def est_one_hop_sons_num(self, layers_iter):
        son_count = 2  # 自己和k递增各一个
        if self.l < len(layers_iter) - 1:
            son_count += 1
        if len(self.f) < self.l:
            son_count += len(layers_iter) - len(self.f)
        return son_count

    def est_all_sons_num(self, layers_iter, max_k):
        son_count = 1  # 自己一个
        f_l_pair = 0
        for i in range(self.l, len(layers_iter)):
            for j in range(len(self.f), i + 1):
                f_l_pair += math.comb(len(layers_iter) - len(self.f), j)
        son_count += f_l_pair * max(0, (max_k - self.k))
        return son_count

    def est_all_fl_sons_num(self, layers_iter):
        son_count = 1  # 自己一个
        f_l_pair = 0
        for i in range(self.l, len(layers_iter)):
            for j in range(len(self.f), i + 1):
                f_l_pair += math.comb(len(layers_iter) - len(self.f), j)
        son_count += f_l_pair
        return son_count


def get_context(multilayer_graph, c: Combo, deg):
    """
    从一个子图计算一个focore
    """
    nodes = set(deg.keys())
    degree = {v: copy.copy(deg[v]) for v in nodes}
    coreness = {v: heapq.nlargest(c.l, degree[v])[-1] for v in nodes}
    to_peel = set()
    for v in nodes:
        if coreness[v] < c.k:
            to_peel.add(v)
        else:
            for l in c.f:
                if degree[v][l] < c.k:
                    to_peel.add(v)

    neighbors = set()
    while to_peel:
        # node 是本轮要处理的点，先标记结果
        node = to_peel.pop()
        del degree[node]
        del coreness[node]
        # 首先在（focus指定的）每一层中移除node，并处理对邻居的影响
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in layer_neighbors:
                if neighbor not in degree:
                    continue
                if degree[neighbor][layer] >= c.k:
                    # 由于node点被peel，导致node的度数>k的邻居 度数-1
                    degree[neighbor][layer] -= 1
                    # 在focus上的层需要额外检查其是否满足对应层上的要求
                    if degree[neighbor][layer] < c.k and layer in c.f:
                        to_peel.add(neighbor)
                    # 由于度数变化，可能导致skyline-bucket需要维护，加入neighbors，之后统一检查
                    if degree[neighbor][layer] + 1 == coreness[neighbor]:
                        neighbors.add(neighbor)
        # focus的点移除干净了，再来检查skyline
        if not to_peel:
            for neighbor in neighbors:
                if neighbor not in degree:
                    # 已经被移除的neighbor不必再检查（可能是加入时还没peel，但之后peel掉了）
                    continue
                # 这步耗时最多，重新计算skyline，并把不满足要求的加入待peel列表
                new_l = heapq.nlargest(c.l, degree[neighbor])[-1]
                coreness[neighbor] = new_l
                if new_l < c.k:
                    to_peel.add(neighbor)
            neighbors = set()

    if len(coreness) == 0:
        max_k = c.k - 1
    else:
        max_k = max(coreness.values())
    return max_k, degree


class FoCache(object):
    def __init__(self, node_iter, layer_iter, multilayer_graph, budget_explore, budget_cache):
        self.nodes_num = len(node_iter)
        self.node_iter = node_iter
        self.layer_iter = layer_iter
        self.mlg = multilayer_graph

        self.comboList = []
        self.comboNameToComboId = {}
        self.sons = {}
        self.fathers = {}
        self.context_reference_cid = {}
        self.bound_k = {"-1": self.nodes_num}

        self.frontierComboIdList = set()  # 当前前线（被枚举，未被计算）
        self.comboStatus = {}  # 0: Enumerated, 1: Frontier, 2:Calculated, 4: Empty, 5: Selected
        self.empty_list = set()

        self.budget_explore = budget_explore
        self.budget_cache = budget_cache
        self.explored = 0
        self.explored_footprint = []
        self.selected_footprint = []
        self.gain_dict = {}

        self.son_combo_num = {}
        self.degrees = {}
        self.cache_degrees = {}
        self.base_degree = {v: [len(layer_neighbors) for layer_neighbors in multilayer_graph[v]] for v in node_iter}

    def est_son_num(self, coid):
        max_son_num = self.comboList[coid].est_all_sons_num(self.layer_iter, self.get_max_k(coid))
        return max_son_num

    def get_max_k(self, cid):
        combo_fl_name = self.comboList[cid].fl_str()
        if combo_fl_name in self.bound_k:
            return self.bound_k[combo_fl_name]
        else:
            return self.bound_k[self.comboList[0].fl_str()]

    def update_k_bound(self, cid, mk):
        fl_name = self.comboList[cid].fl_str()
        if fl_name in self.bound_k:
            self.bound_k[fl_name] = min(self.bound_k[fl_name], mk)
        else:
            self.bound_k[fl_name] = mk

    def get_context(self, cid):
        context_cid = self.context_reference_cid[cid]
        if context_cid == -1:
            return context_cid, self.base_degree
        if self.comboStatus[context_cid] == 2:
            return context_cid, self.degrees[context_cid]
        elif self.comboStatus[context_cid] == 5:
            return context_cid, self.degrees[context_cid]
        else:
            return context_cid, self.base_degree

    def update_context(self, cid, ctx_id):
        old_ctx_id = self.context_reference_cid[cid]
        if old_ctx_id == -1:
            old_peel_num = 0
        else:
            old_peel_num = self.nodes_num - len(self.degrees[old_ctx_id])
        new_peel_num = self.nodes_num - len(self.degrees[ctx_id])
        if new_peel_num > old_peel_num:
            self.context_reference_cid[cid] = ctx_id
            return True
        return False

    def enum_combo(self, combo_name: str = None):
        # 无输入时枚举root-combo
        if combo_name is None:
            c = Combo(focus=[], lamb=1, k=1)
            combo_name = str(c)
        # 若曾被枚举，返回cid
        if combo_name in self.comboNameToComboId:
            cid = self.comboNameToComboId[combo_name]
            return cid
        # 这个点被枚举，状态变化，首先把这个combo记录下来
        c = Combo(name=combo_name)
        cid = len(self.comboList)
        self.comboNameToComboId[combo_name] = len(self.comboList)
        self.comboList.append(c)
        # 父子关系交给外边建立（枚举顺序）
        self.context_reference_cid[cid] = -1
        self.frontierComboIdList.add(cid)
        self.comboStatus[cid] = 1
        self.sons[cid] = set()
        self.fathers[cid] = set()
        father_names = c.get_one_hop_father_names()
        for f_name in father_names:
            if f_name in self.comboNameToComboId:
                f_id = self.comboNameToComboId[f_name]
                self.fathers[cid].add(f_id)
                self.sons[f_id].add(cid)
        self.son_combo_num[cid] = self.est_son_num(cid)
        return cid

    def enum_combo_lattice(self, combo_name: str = None):
        # 若曾被枚举，返回cid
        if combo_name in self.comboNameToComboId:
            cid = self.comboNameToComboId[combo_name]
            return cid
        # 这个点被枚举，状态变化，首先把这个combo记录下来
        c = Combo(name=combo_name)
        for empty_id in self.empty_list:
            if c.is_son(self.comboList[empty_id]):
                return -1
        cid = len(self.comboList)
        self.comboNameToComboId[combo_name] = len(self.comboList)
        self.comboList.append(c)
        # 父子关系交给外边建立（枚举顺序）
        self.context_reference_cid[cid] = -1
        self.comboStatus[cid] = 1
        self.sons[cid] = set()
        self.fathers[cid] = set()
        father_names = c.get_one_hop_father_names()
        for f_name in father_names:
            if f_name in self.comboNameToComboId:
                f_id = self.comboNameToComboId[f_name]
                if self.comboStatus[f_id] == 4:
                    self.comboStatus[cid] = 4

                self.fathers[cid].add(f_id)
                self.sons[f_id].add(cid)
        self.son_combo_num[cid] = self.est_son_num(cid)
        return cid

    def explore_combo(self, cid):
        ctxid, context = self.get_context(cid)
        # L.info(f"Getting {self.comboList[cid]} from context of {self.comboList[ctxid]}: length {len(context)}")
        mk, deg = get_context(self.mlg, self.comboList[cid], context)
        self.update_k_bound(cid, mk)
        if len(deg) == 0:
            self.comboStatus[cid] = 4
            to_mark_son_id = deque([cid])
            while to_mark_son_id:
                sid = to_mark_son_id.popleft()
                for son_id in self.sons[sid]:
                    if self.comboStatus[son_id] == 1:
                        self.comboStatus[son_id] = 4
                        self.frontierComboIdList.remove(son_id)
                        to_mark_son_id.append(son_id)
            to_prune_combo_num = self.son_combo_num[cid]
            to_update_father_id = deque([cid])
            while to_update_father_id:
                fid = to_update_father_id.popleft()
                for father_id in self.fathers[fid]:
                    if self.comboStatus[father_id] == 1:
                        to_update_father_id.append(father_id)
                self.son_combo_num[fid] -= to_prune_combo_num

        else:
            self.degrees[cid] = deg
            self.comboStatus[cid] = 2
            # 枚举自己的全部孩子进入frontier
            son_name_list = self.comboList[cid].get_one_hop_sons_name(self.layer_iter)
            for son_name in son_name_list:
                # 未被枚举的combo，插入
                son_id = self.enum_combo(son_name)
                self.sons[cid].add(son_id)
                self.fathers[son_id].add(cid)
                self.update_context(son_id, cid)
            to_update_father_id = deque([cid])
            fl_sons_num = self.comboList[cid].est_all_fl_sons_num(self.layer_iter)
            while to_update_father_id:
                fid = to_update_father_id.popleft()
                for father_id in self.fathers[fid]:
                    if self.comboStatus[father_id] == 1:
                        to_update_father_id.append(father_id)
                self.son_combo_num[fid] -= fl_sons_num * (self.get_max_k(fid) - mk)
        return 1

    def select_explore_id(self):
        ubd = {}
        for cid in self.frontierComboIdList:
            ubc = self.son_combo_num[cid]
            if self.context_reference_cid[cid] == -1:
                ubv = self.nodes_num
            else:
                ubv = len(self.degrees[self.context_reference_cid[cid]])
            ub = ubc * ubv
            ubd[cid] = ub
        scid, max_value = max(ubd.items(), key=lambda x: x[1])
        self.frontierComboIdList.remove(scid)
        return scid

    def select_build_id(self, candidates):
        scid, max_value = max(self.gain_dict.items(), key=lambda x: x[1])
        candidates.remove(scid)
        self.gain_dict.pop(scid)
        return scid

    def explore(self):
        self.enum_combo()
        while self.frontierComboIdList:
            cid = self.select_explore_id()
            self.explored += self.explore_combo(cid)
            self.explored_footprint.append(str(self.comboList[cid]))
            if self.explored >= self.budget_explore:
                break
        q = deque(self.frontierComboIdList)
        self.empty_list = {cid for cid, s in self.comboStatus.items() if s == 4}
        max_lattice = 0
        for cid in self.frontierComboIdList:
            max_lattice = max(self.comboList[cid].get_lattice(), max_lattice)
        while q:
            cid = q.popleft()
            if self.comboList[cid].get_lattice() < max_lattice:
                if self.comboStatus[cid] != 4:
                    son_name_list = self.comboList[cid].get_one_hop_sons_name(self.layer_iter)
                    for son_name in son_name_list:
                        # 未被枚举的combo，插入
                        if son_name not in self.comboNameToComboId:
                            son_id = self.enum_combo_lattice(son_name)
                            if son_id == -1:
                                continue
                            self.sons[cid].add(son_id)
                            self.fathers[son_id].add(cid)
                            q.append(son_id)

        value_counts = Counter(self.comboStatus.values())
        L.info(
            f"Explore num={self.explored}, Calculated-Non-empty={value_counts[2]}, Calculated_Empty={self.explored - value_counts[2]}, Marked-empty={value_counts[4] - (self.explored - value_counts[2])}")

    def cal_gain(self, cid):
        gain = 0
        all_possible = deque([cid])
        caled = set()
        while all_possible:
            ccid = all_possible.popleft()
            if ccid in caled:
                continue
            caled.add(ccid)
            ctx_id_now = self.context_reference_cid[ccid]
            cc_gain = 0
            if ctx_id_now != -1:
                cc_gain = self.nodes_num - len(self.degrees[ctx_id_now])
            c_gain = self.nodes_num - len(self.degrees[cid])
            if cc_gain < c_gain:
                gain += c_gain - cc_gain
            if self.comboStatus[ccid] == 4:
                continue
            for sid in self.sons[ccid]:
                if sid not in caled:
                    all_possible.append(sid)
        return gain

    def build(self):
        L.info("Exploring...")
        timer = Timer()
        timer.start()
        self.explore()
        timer.stop()
        L.info("Exploring phase:")
        timer.print_timer()
        L.info("Selecting...")
        timer = Timer()
        timer.start()
        for key in self.context_reference_cid:
            self.context_reference_cid[key] = -1
        candidates = [cid for cid, status in self.comboStatus.items() if status == 2]
        for cid in candidates:
            self.gain_dict[cid] = self.cal_gain(cid)

        while len(self.cache_degrees) < self.budget_cache and len(candidates) > 0:
            if len(self.cache_degrees) % 10 == 0:
                L.info(f"{len(self.cache_degrees)}/{self.budget_cache}")
            cid = self.select_build_id(candidates)
            for upcid in candidates:
                self.gain_dict[upcid] = self.cal_gain(upcid)
            self.selected_footprint.append(cid)
            self.comboStatus[cid] = 5
            # L.info(f"Selected: {self.comboList[cid]}")
            to_update_cids = deque([cid])
            while to_update_cids:
                ucid = to_update_cids.popleft()
                for son_id in self.sons[ucid]:
                    if self.comboStatus[son_id] == 4:
                        continue
                    if self.comboStatus[son_id] == 2 or self.comboStatus[son_id] == 5:
                        to_update_cids.append(son_id)
                is_updated = self.update_context(ucid, cid)
            self.cache_degrees[cid] = self.degrees[cid]
        timer.stop()
        L.info("Selecting phase:")
        timer.print_timer()

    def get_focore_using_cache(self, query_name: str):
        if query_name in self.comboNameToComboId:
            cid = self.comboNameToComboId[query_name]
            if cid in self.cache_degrees:
                # L.info(f"Cid {self.comboList[cid]} in cache, return cached result:")
                return self.cache_degrees[cid].keys()
        q = deque([query_name])
        infer_cids = []
        while q:
            combo_name = q.popleft()
            c = Combo(name=combo_name)
            f_names = c.get_one_hop_father_names()
            for f_name in f_names:
                if f_name in self.comboNameToComboId:
                    cid = self.comboNameToComboId[f_name]
                    if self.comboStatus[cid] == 5:
                        infer_cids.append(cid)
                    elif self.comboStatus[cid] == 4:
                        # L.info(f"Getting {query_name} empty Core:")
                        return set()
                    else:
                        q.append(f_name)
                else:
                    q.append(f_name)
        final_infer_cid = -1
        infer_size = self.nodes_num
        for infer_cid in infer_cids:
            if len(self.cache_degrees[infer_cid]) < infer_size:
                final_infer_cid = infer_cid
                infer_size = len(self.cache_degrees[infer_cid])
        if final_infer_cid != -1 and self.comboStatus[final_infer_cid] == 4:
            return set()
        else:
            if final_infer_cid == -1:
                # L.info(f"Getting {query_name} using base:")
                return get_context(self.mlg, Combo(name=query_name), self.base_degree)[1].keys()
            else:
                # L.info(f"Getting {query_name} using context of {self.comboList[final_infer_cid]}:")
                return get_context(self.mlg, Combo(name=query_name), self.cache_degrees[final_infer_cid])[1].keys()

    def print_info(self):
        value_counts = Counter(self.comboStatus.values())
        L.info(f"Combo Status: {value_counts} (1: Frontier, 2:Calculated, 4: Empty, 5:Selected)")
        L.info(f"Explored Combos: {self.explored_footprint}")
        footprint = [str(self.comboList[cid]) for cid in self.selected_footprint]
        L.info(f"Selected Combos: {footprint}")


def enum_prune(multilayer_graph, nodes_iterator, layers_iterator, be, bs):
    # be = len(layers_iterator) * 10
    # bs = len(layers_iterator) * 10
    L.info(f"Explore Budget = {be}, Size Budget = {bs}")
    fc = FoCache(nodes_iterator, layers_iterator, multilayer_graph, be, bs)
    fc.build()
    # fc.bfs_build()
    fc.print_info()
    L.info("Query On Cache...")
    timer = Timer()

    total_combo_size = 0
    to_explore = deque([Combo(focus=[], lamb=1, k=1)])
    while to_explore:
        combo = to_explore.popleft()
        # L.info(f"Getting Core: {combo}")
        timer.start()
        core = fc.get_focore_using_cache(str(combo))
        total_combo_size += 1
        timer.stop()
        if len(core) != 0:
            sons = combo.enum_one_hop_son_combos(layers_iterator)
            to_explore += sons
    L.info(f"Query Phase ({total_combo_size} Combos): ")
    timer.print_timer()


def check(multilayer_graph, nodes_iterator, layers_iterator, dataset_name):
    for be in [100]:
        for bc in [10]:
            enum_prune(multilayer_graph, nodes_iterator, layers_iterator, len(layers_iterator) * be,
                       len(layers_iterator) * bc)
