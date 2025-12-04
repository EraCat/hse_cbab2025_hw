import sys
from collections import deque
from typing import Any


def cat_lovers(graph):
    triangles = 0

    for v in range(len(graph)):
        neighbors = list(graph[v])
        if len(neighbors) < 2:
            continue

        for i in range(len(neighbors)):
            u = neighbors[i]
            if v > u:
                continue
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if v > w:
                    continue

                if w in graph[u]:
                    triangles += 1

    return triangles


def run_cat_lovers():
    graph = read_undirected_graph_as_adjacency_list()

    print(cat_lovers(graph))


def read_undirected_graph_as_adjacency_list() -> list[set[Any]]:
    n, m = map(int, input().split())
    graph = [set() for _ in range(n)]

    for _ in range(m):
        a, b = map(int, input().split())
        graph[a - 1].add(b - 1)
        graph[b - 1].add(a - 1)
    return graph


def read_directed_graph_as_adjacency_list(n, m) -> list[set[Any]]:
    graph = [set() for _ in range(n)]

    for _ in range(m):
        a, b = map(int, input().split())
        graph[a - 1].add(b - 1)
    return graph


def connected_components(graph):
    visited = [0] * len(graph)
    color = 0
    for v in range(len(graph)):
        if visited[v] == 0:
            color += 1
            dfs_iter_with_color(graph, v, visited, color)

    print(color)
    print(*visited)


def dfs_iter_with_color(graph, v, visited: list[int], color: int):
    stack = [v]
    while len(stack) > 0:
        u = stack.pop()
        if visited[u] == 0:
            visited[u] = color
            for w in graph[u]:
                stack.append(w)


def dfs_with_color(graph, v, visited, color):
    visited[v] = color
    for w in graph[v]:
        if visited[w] == 0:
            dfs_with_color(graph, w, visited, color)


def run_connected_components():
    graph = read_undirected_graph_as_adjacency_list()
    connected_components(graph)


def find_path(graph, start, end):
    visited = [False] * len(graph)
    parent = [-1] * len(graph)
    visited[start] = True
    stack = [start]
    while len(stack) > 0:
        u = stack.pop()
        if u == end:
            break
        for w in graph[u]:
            if visited[w] == 0:
                visited[w] = True
                parent[w] = u
                stack.append(w)

    if parent[end] == -1:
        return [-1]

    path = deque()
    cur = end
    while cur != -1:
        path.appendleft(cur + 1)
        cur = parent[cur]

    return path


def run_find_path():
    n, m, s, e = map(int, input().split())

    graph = read_directed_graph_as_adjacency_list(n, m)
    print(*find_path(graph, s - 1, e - 1))


def is_tree(graph):
    visited = [False] * len(graph)

    stack = [(0, -1)]
    visited[0] = True
    while stack:
        v, parent = stack.pop()
        for u in graph[v]:
            if u == parent:
                continue
            if visited[u]:
                return False
            visited[u] = True
            stack.append((u, v))

    if all(visited):
        return True

    return False


def run_is_tree():
    graph = read_undirected_graph_as_adjacency_list()
    print('YES' if is_tree(graph) else 'NO')


def find_amount_of_connected_components(graph):
    visited = [False] * len(graph)
    color = 0
    vertices = []

    for v in range(len(graph)):
        if visited[v] == 0:
            color += 1
            vertices.append(v)
        dfs_iter_with_color(graph, v, visited, color)

    edges = []
    for i in range(len(vertices) - 1):
        edges.append((vertices[i] + 1, vertices[i + 1] + 1))

    print(color - 1)
    for a, b in edges:
        print(a, b)


def run_find_amount_of_connected_components():
    graph = read_undirected_graph_as_adjacency_list()
    find_amount_of_connected_components(graph)


def find_cycle(graph):
    n = len(graph)
    color = [0] * n
    parent = [-1] * n
    cycle = None

    def build_cycle(v, w):
        path = []
        cur = v
        path.append(cur + 1)
        while cur != w:
            cur = parent[cur]
            path.append(cur + 1)
        path.reverse()
        return path

    for s in range(n):
        if color[s] != 0:
            continue

        color[s] = 1
        parent[s] = -1
        stack = [(s, iter(graph[s]))]

        while stack:
            v, it = stack[-1]
            w = next(it, None)

            if w is None:
                color[v] = 2
                stack.pop()
                continue

            if color[w] == 0:
                parent[w] = v
                color[w] = 1
                stack.append((w, iter(graph[w])))
            elif color[w] == 1:
                cycle = build_cycle(v, w)
                print("YES")
                print(*cycle)
                return

    print("NO")


def run_find_cycle():
    n, m = map(int, input().split())
    graph = read_directed_graph_as_adjacency_list(n, m)

    find_cycle(graph)


def main():
    sys.setrecursionlimit(10 ** 6)
    # run_cat_lovers()
    # run_connected_components()
    # run_find_path()
    # run_is_tree()
    # run_find_amount_of_connected_components()
    run_find_cycle()


if __name__ == '__main__':
    main()
