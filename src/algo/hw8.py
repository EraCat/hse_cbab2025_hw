import sys
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


def read_directed_graph_as_adjacency_list() -> list[set[Any]]:
    n, m = map(int, input().split())
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

def main():
    # run_cat_lovers()
    run_connected_components()


if __name__ == '__main__':
    main()
