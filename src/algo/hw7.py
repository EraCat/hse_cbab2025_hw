def shortest_g_path(martix):
    INF = 10 ** 9
    n = len(martix)
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    for i in range(len(martix)):
        dp[1 << i][i] = 0

    for mask in range(1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == INF:
                continue

            for j in range(n):
                if mask & (1 << j):
                    continue

                new_mask = mask | (1 << j)
                new_dist = dp[mask][i] + martix[i][j]

                if new_dist < dp[new_mask][j]:
                    dp[new_mask][j] = new_dist
                    parent[new_mask][j] = i

    ALL = (1 << n) - 1

    best_end = -1
    best_dist = INF
    for v in range(n):
        if dp[ALL][v] < best_dist:
            best_dist = dp[ALL][v]
            best_end = v

    path = []
    mask = ALL
    v = best_end

    while v != -1:
        path.append(v + 1)
        pv = parent[mask][v]
        mask ^= (1 << v)
        v = pv

    path.reverse()

    print(best_dist)
    print(*path)


def run_shortest_g_path():
    n = int(input())
    m = [[0] * n for _ in range(n)]
    for i in range(n):
        m[i] = list(map(int, input().split()))

    shortest_g_path(m)


def mark_and_jump(n, x, y, marks, unmarks):
    dp = [False] * (1 << (n + 1))

    for s in marks:
        mask = 0
        for e in s:
            mask |= (1 << e)
        dp[mask] = True

        sub = mask
        while sub != 0:
            sub = (sub - 1) & mask
            dp[sub] = True

    for s in unmarks:
        mask = 0
        for e in s:
            mask |= (1 << e)
        dp[mask] = False

        sub = mask
        while sub != 0:
            sub = (sub - 1) & mask
            dp[sub] = False

    count = 0
    for i in range(len(dp)):
        if dp[i]:
            count += 1

    return count


def run_mark_and_jump():
    data = list(map(int, input().split()))
    n = data[0]
    x = data[1]
    y = data[2]
    marks = []
    unmarks = []
    for _ in range(x):
        l = list(map(int, input().split()))
        marks.append(l[1:])
    for _ in range(y):
        l = list(map(int, input().split()))
        unmarks.append(l[1:])

    print(mark_and_jump(n, x, y, marks, unmarks))


def mark_and_jump_2(n, x, y, marks, unmarks):
    marks_masks = [0] * (1 << n)
    unmarks_masks = [0] * (1 << n)

    for s in marks:
        mask = 0
        for e in s:
            mask |= (1 << (e - 1))
        marks_masks[mask] = 1

    for s in unmarks:
        mask = 0
        for e in s:
            mask |= (1 << (e - 1))
        unmarks_masks[mask] = 1

    marks_subs_masks = marks_masks[:]
    unmarks_subs_masks = unmarks_masks[:]

    for bit in range(n):
        for mask in range(1 << n):
            if mask & (1 << bit) == 0:
                marks_subs_masks[mask] |= marks_subs_masks[mask | (1 << bit)]
                unmarks_subs_masks[mask] |= unmarks_subs_masks[mask | (1 << bit)]

    count = 0
    for mask in range(1 << n):
        if marks_subs_masks[mask] == 1 and unmarks_subs_masks[mask] != 1:
            count += 1

    return count


def run_mark_and_jump_2():
    data = list(map(int, input().split()))
    n = data[0]
    x = data[1]
    y = data[2]
    marks = []
    unmarks = []
    for _ in range(x):
        l = list(map(int, input().split()))
        marks.append(l[1:])
    for _ in range(y):
        l = list(map(int, input().split()))
        unmarks.append(l[1:])

    print(mark_and_jump_2(n, x, y, marks, unmarks))


def main():
    # run_shortest_g_path()
    # run_mark_and_jump()
    run_mark_and_jump_2()


if __name__ == '__main__':
    main()
