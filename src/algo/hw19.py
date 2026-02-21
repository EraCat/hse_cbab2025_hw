import sys


def task_1():
    input_data = sys.stdin.buffer

    n, m = map(int, input_data.readline().split())

    g = [[] for _ in range(n)]
    for i in range(n):
        for j in list(map(int, input_data.readline().split())):
            if j == 0:
                break
            g[i].append(j - 1)

    matching = [-1] * m

    def dfs(v):
        if used[v]:
            return False

        used[v] = True
        for u in g[v]:
            if matching[u] == -1 or dfs(matching[u]):
                matching[u] = v
                return True
        return False

    for a in range(n):
        used = [False] * (n)
        dfs(a)

    res = []
    for v, u in enumerate(matching):
        if u != -1:
            res.append((u + 1, v + 1))

    print(len(res))
    for u, v in res:
        print(u, v)


def task_2():
    input_data = sys.stdin.buffer
    n, m = map(int, input_data.readline().split())

    adj = [[] for _ in range(m)]
    for i in range(m):
        data = list(map(int, input_data.readline().split()))
        adj[i] = [v - 1 for v in data[1:]]

    raw = list(map(int, input_data.readline().split()))
    matchL = [-1] * m
    for i in range(m):
        x = raw[i]
        matchL[i] = -1 if x == 0 else (x - 1)

    matchR = [-1] * n
    for i in range(m):
        j = matchL[i]
        if j != -1:
            matchR[j] = i





def main():
    task_1()


if __name__ == "__main__":
    main()
