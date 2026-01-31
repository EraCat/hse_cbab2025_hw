def task_1():
    n = int(input())

    g = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v, w = map(int, input().split())
        g[u].append((v, w))
        g[v].append((u, w))

    depth = [0] * n
    dist = [0] * n
    logn = n.bit_length()
    up = [[0 for _ in range(logn)] for _ in range(n)]

    stack = [(0, 0)]
    while stack:
        q, p = stack.pop()
        up[q][0] = p
        for j in range(1, logn):
            up[q][j] = up[up[q][j - 1]][j - 1]

        for to, wei in g[q]:
            if to == p:
                continue
            depth[to] = depth[q] + 1
            dist[to] = dist[q] + wei
            stack.append((to, q))

    def lca(a, b):
        if depth[a] < depth[b]:
            a, b = b, a

        diff = depth[a] - depth[b]
        k = 0
        while diff:
            if diff & 1:
                a = up[a][k]
            diff >>= 1
            k += 1

        if a == b:
            return a

        for jj in range(logn - 1, -1, -1):
            if up[a][jj] != up[b][jj]:
                a = up[a][jj]
                b = up[b][jj]

        return up[a][0]

    m = int(input())

    for i in range(m):
        a1, b1 = map(int, input().split())
        x = lca(a1, b1)
        print(dist[a1] + dist[b1] - 2 * dist[x])


def main():
    task_1()


if __name__ == '__main__':
    main()
