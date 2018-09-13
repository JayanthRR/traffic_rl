from traffic_model import *
import numpy as np
# import matplotlib.pyplot as plt
import scipy
import math
from math import inf, log
import queue as Q
from tqdm import tqdm
import heapq

# np.random.seed = 0


class Edge:
    def __init__(self):
        self.distance = math.inf
        self.parent = None


def cost_function(x):
    # think of a better cost function that is smooth and monotone increasing
    #     return log(1+x)
    # return log(1+x)
    return np.power(x, 2, dtype=np.float64)
    # return x


def compute_cost(xt, current_link, destination):
    # return xt[current_link]
    if current_link == destination:
        return 0
    else:
        return cost_function(xt[current_link])


def get_successor_edges(A, current_edge):
    adj = []
    for i in range(A.shape[0]):
        if A[i, current_edge] > 0:
            adj.append(i)
    return adj


def get_predecessor_edges(A, current_edge):
    adj = []
    for i in range(A.shape[0]):
        if A[current_edge, i] > 0:
            adj.append(i)
    return adj


def is_exists_path(A, source, destination):

    queue = Q.Queue()
    queue.put(source)
    covered_edges = []

    while not queue.empty():
        current_edge = queue.get()
        covered_edges.append(current_edge)
        for edge in get_successor_edges(A, current_edge):
            if edge == destination:
                return True

            if edge not in covered_edges:
                queue.put(edge)

    return False


def greedy(A, xt, source, destination, term_flag=False, signal_flag=False):
    # flag determines what to do when no decision is taken. (Should it stay on the same edge or terminate? )
    # source is the current link and the destination is the destination link

    # if signal_flag and st[source] == 1:
    #     # print("greedy signal")
    #     return source

    next_node = None
    min_cost = inf
    for j in get_successor_edges(A, source):
        if j != source:  # and (A[j, source] > 0):
            if j == destination:
                next_node = j

                break
            if min_cost > compute_cost(xt, j, destination):
                next_node = j
                min_cost = compute_cost(xt, j, destination)
    # this part deals with when there is no way out from the current link (Have to deal with this separately
    # in the signalized case.)
    if not term_flag:
        if next_node is None:
            next_node = source

    return next_node


"""Variant of dijkstra implementation. Here A is not adjacency matrix wrt nodes, but wrt edges. so A(i,j) doesn't
add any significance apart from determining the adjacent edges. The cost computation is performed using xt. 
"""


def dijkstra(A, xt, source, destination):
    edges = [Edge() for _ in range(len(xt))]
    edges[source].distance = 0

    colored_edges = []
    q = [(edges[j].distance, j) for j in range(len(edges))]
    # q = Q.PriorityQueue()
    # for j in range(len(xt)):
    #     q.put((edges[j].distance, j))
    heapq.heapify(q)

    while q:
        _, current_edge = heapq.heappop(q)
        if current_edge not in colored_edges:
            colored_edges.append(current_edge)

        adj = get_successor_edges(A, current_edge)

        for j in adj:
            if edges[j].distance > edges[current_edge].distance + compute_cost(xt, j, destination):
                temp = (edges[j].distance, j)
                q = list(q)
                q.remove(temp)
                edges[j].distance = edges[current_edge].distance + compute_cost(xt, j, destination)
                edges[j].parent = current_edge
                q.append((edges[j].distance, j))
                heapq.heapify(q)

            if j == destination:
                break

    if (edges[destination].distance is inf) and is_exists_path(A, source, destination):
        print("path not found, how?")
        # dijkstra(A, xt, source, destination)

    return edges, colored_edges


def dijkstra_policy(A, xt, source, destination, term_flag=False, signal_flag=False):
    # chooses just the next edge to traverse
    #
    # if signal_flag and st[source] == 1:
    #     # print("signal")
    #     return source

    decision = None

    path = []
    edges, colored_edges = dijkstra(A, xt, source, destination)

    if edges[destination].distance is inf:
        if not term_flag:
            return source
        else:
            return None

    prev = destination
    while prev is not None:
        temp = edges[prev].parent
        path.append(temp)
        if temp == source:
            decision = prev
            break
        prev = temp

    return decision


def const_dijkstra_policy(A, xt, source, destination, term_flag=False, signal_flag=False):
    # chooses just the next edge to traverse
    #
    # if signal_flag and st[source] == 1:
    #     # print("signal")
    #     return source

    decision = None

    path = []
    edges, colored_edges = dijkstra(A, xt, source, destination)

    if edges[destination].distance is inf:
        if not term_flag:
            return source
        else:
            return None

    prev = destination
    while prev is not None:
        temp = edges[prev].parent
        if temp == source:
            decision = prev
            break
        path.append(temp)
        prev = temp

    return path


def policy_fn(A, xt, source, destination, policy="dijkstra", term_flag=False, signal_flag=False, lookahead=0):

    if policy == "dijkstra":
        for i in range(lookahead):
            xt = state_transition(A, xt, wt=np.zeros_like(xt))
            # xt = state_transition(A, xt, wt=gaussian(0, 0.1, A.shape[0]))
        return dijkstra_policy(A, xt, source, destination, term_flag=term_flag, signal_flag=signal_flag)

    elif policy == "greedy":
        for i in range(lookahead):
            xt = state_transition(A, xt, wt=np.zeros_like(xt))
            # xt = state_transition(A, xt, wt=gaussian(0, 0.1, A.shape[0]))

        return greedy(A, xt, source, destination, term_flag=term_flag, signal_flag=signal_flag)
