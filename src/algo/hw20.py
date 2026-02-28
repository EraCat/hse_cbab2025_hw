from dataclasses import dataclass


@dataclass
class Edge:
    to: int
    reverse: int
    cap: int
    orig: int


def task_1():
    n, m, s, t = map(int, input().split())
    s -= 1
    t -= 1

    g = [[] for _ in range(n)]

    def add_edge(v, u):
        g[v].append(Edge(u, len(g[u]), 1, 1))
        g[u].append(Edge(v, len(g[v]) - 1, 0, 0))

    for _ in range(m):
        i, j = map(int, input().split())
        i -= 1
        j -= 1
        add_edge(i, j)

    def dfs(s, t):
        parent_v = [-1] * n
        parent_e = [-1] * n
        visited = [False] * n

        stack = [(s, 0)]
        visited[s] = True
        parent_v[s] = s

        while stack:
            v, i = stack[-1]
            if v == t:
                return parent_v, parent_e

            if i >= len(g[v]):
                stack.pop()
                continue

            stack[-1] = (v, i + 1)
            e = g[v][i]
            if e.cap <= 0:
                continue
            u = e.to
            if visited[u]:
                continue

            visited[u] = True
            parent_v[u] = v
            parent_e[u] = i
            stack.append((u, 0))

        return None, None

    def push(parent_v, parent_e):
        cur = t
        while cur != s:
            pv = parent_v[cur]
            ei = parent_e[cur]
            e = g[pv][ei]
            e.cap -= 1
            g[cur][e.reverse].cap += 1
            cur = pv

    flow = 0
    for _ in range(2):
        pv, pe = dfs(s, t)
        if pv is None:
            break
        push(pv, pe)
        flow += 1

    if flow < 2:
        print("NO")
        return

    def restore_path(s, t):
        parent_v = [-1] * n
        parent_e = [-1] * n
        visited = [False] * n

        stack = [(s, 0)]
        visited[s] = True
        parent_v[s] = s

        while stack:
            v, i = stack[-1]
            if v == t:
                break

            if i >= len(g[v]):
                stack.pop()
                continue

            stack[-1] = (v, i + 1)
            e = g[v][i]
            if e.orig == 0:
                continue

            if g[e.to][e.reverse].cap <= 0:
                continue

            u = e.to
            if visited[u]:
                continue

            visited[u] = True
            parent_v[u] = v
            parent_e[u] = i
            stack.append((u, 0))

        path = []
        cur = t
        while cur != s:
            path.append(cur)
            pv = parent_v[cur]
            ei = parent_e[cur]
            e = g[pv][ei]

            g[cur][e.reverse].cap -= 1
            e.cap += 1

            cur = pv

        path.append(s)
        path.reverse()

        return [v + 1 for v in path]

    print('YES')
    print(*restore_path(s, t))
    print(*restore_path(s, t))



def main():
    task_1()


if __name__ == "__main__":
    main()
