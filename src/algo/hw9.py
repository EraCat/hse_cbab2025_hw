from collections import deque
from queue import PriorityQueue
from typing import Any, NamedTuple


class Edge(NamedTuple):
    to: int
    weight: float


def find_sum_between(graph):
    def bfs(u, graph):
        dist = [-1] * len(graph)
        dist[u] = 0
        queue = deque([u])
        while queue:
            v = queue.popleft()
            for neighbor in graph[v]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[v] + 1
                    queue.append(neighbor)

        return dist

    total = 0
    for v in range(len(graph)):
        total += sum(bfs(v, graph))

    total //= 2
    return total


def run_find_sum_between():
    adjacency_list = read_undirected_graph_as_adjacency_list()

    print(find_sum_between(adjacency_list))


def read_undirected_graph_as_adjacency_list() -> list[set[Any]]:
    n, m = map(int, input().split())
    graph = [set() for _ in range(n)]

    for _ in range(m):
        a, b = map(int, input().split())
        graph[a - 1].add(b - 1)
        graph[b - 1].add(a - 1)
    return graph


def read_undirected_graph_as_adjacency_list_with_weights():
    n, m = map(int, input().split())
    s, t = map(int, input().split())
    graph = [set() for _ in range(n)]

    for _ in range(m):
        a, b, w = map(int, input().split())
        graph[a - 1].add(Edge(b - 1, w))
        graph[b - 1].add(Edge(a - 1, w))

    return graph, s, t


def find_distance_between(graph, s, t):
    dist = [10 ** 6] * len(graph)
    dist[s] = 0

    pq = PriorityQueue()
    pq.put((0, s))
    while pq.qsize() > 0:
        cur_d, v = pq.get_nowait()
        if cur_d > dist[v]:
            continue
        for edge in graph[v]:
            if dist[edge.to] > dist[v] + edge.weight:
                dist[edge.to] = dist[v] + edge.weight
                pq.put((dist[edge.to], edge.to))

    return -1 if dist[t] == 10 ** 6 else dist[t]


def find_fee_amount(graph, s):
    cost = [-1] * len(graph)
    cost[s] = 0
    queue = deque([s])
    while queue:
        v = queue.popleft()
        for neighbor in graph[v]:
            if cost[neighbor] == -1:
                cost[neighbor] = cost[v] + 1
                queue.append(neighbor)

    print(*cost)



def run_find_fee_amount():
    def read_data() -> tuple[list[set[Any]], int]:
        n, s, m = map(int, input().split())
        graph = [set() for _ in range(n)]

        for _ in range(m):
            a, b = map(int, input().split())
            # graph[a - 1].add(b - 1)
            graph[b - 1].add(a - 1)
        return graph, s

    adjacency_list, s = read_data()

    find_fee_amount(adjacency_list, s - 1)


def run_find_distance_between():
    graph, s, t = read_undirected_graph_as_adjacency_list_with_weights()

    print(find_distance_between(graph, s - 1, t - 1))


def find_longest_path(graph):
    ordered_v = topo_sort(graph)

    dist = [0] * len(graph)
    for v in ordered_v:
        if dist[v] == -1:
            continue
        for w in graph[v]:
            dist[w] = max(dist[w], dist[v] + 1)

    return max(dist)

def run_find_longest_path():
    n, m = map(int, input().split())
    graph = [set() for _ in range(n)]
    for _ in range(m):
        a, b = map(int, input().split())
        graph[a - 1].add(b - 1)
    print(find_longest_path(graph))


def topo_sort(graph):
    in_degree = [0] * len(graph)
    for v in range(len(graph)):
        for w in graph[v]:
            in_degree[w] = in_degree[w] + 1

    q = deque([v for v in range(len(graph)) if in_degree[v] == 0])
    order = []

    while q:
        v = q.popleft()
        order.append(v)
        for w in graph[v]:
            in_degree[w] -= 1
            if in_degree[w] == 0:
                q.append(w)

    return order


def find_shortest_path(graph, s,t):
    dist = [10 ** 6] * len(graph)
    dist[s] = 0
    parent = [-1] * len(graph)

    pq = PriorityQueue()
    pq.put((0, s))

    while pq.qsize() > 0:
        cur_d, v = pq.get_nowait()
        if cur_d > dist[v]:
            continue
        for edge in graph[v]:
            if dist[edge.to] > dist[v] + edge.weight:
                parent[edge.to] = v
                dist[edge.to] = dist[v] + edge.weight
                pq.put((dist[edge.to], edge.to))

    if dist[t] == 10 ** 6:
        print(-1)
        return

    path = []
    cur = t
    while cur != -1:
        path.append(cur + 1)
        cur = parent[cur]

    path.reverse()

    print(dist[t])
    print(*path)


def run_find_shortest_path():
    graph, s, t = read_undirected_graph_as_adjacency_list_with_weights()

    find_shortest_path(graph, s - 1, t - 1)


def main():
    # run_find_sum_between()
    # run_find_distance_between()
    # run_find_fee_amount()
    # run_find_longest_path()
    run_find_shortest_path()


if __name__ == '__main__':
    main()
