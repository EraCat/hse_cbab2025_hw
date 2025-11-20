import math
import sys


def rabbit(n, path: str):
    dp = [0] * n
    dp[0] = get_cost(0, path)
    for i in range(1, n):
        if path[i] == 'w':
            dp[i] = -1
            continue

        max_cost = -1
        if i >= 1 and dp[i - 1] >= 0:
            max_cost = max(max_cost, dp[i - 1])
        if i >= 3 and dp[i - 3] >= 0:
            max_cost = max(max_cost, dp[i - 3])
        if i >= 5 and dp[i - 5] >= 0:
            max_cost = max(max_cost, dp[i - 5])

        if max_cost == -1:
            dp[i] = -1
            continue

        dp[i] = max_cost + get_cost(i, path)

    return dp[-1]


def get_cost(i, path):
    if path[i] == 'w':
        return -1
    if path[i] == '.':
        return 0
    if path[i] == '"':
        return 1


def run_rabbit():
    n = int(input())
    path = input()

    print(rabbit(n, path))


cube_dp = [10 ** 9] * 50001
cubes = []


def prepare_dp_cube():
    cube_dp[0] = 0
    cube_dp[1] = 1

    i = 1
    while i ** 3 <= len(cube_dp):
        cubes.append(i ** 3)
        cube_dp[i ** 3] = 1
        i += 1

    for i in range(1, len(cube_dp)):
        for c in cubes:
            if c > i:
                break
            cube_dp[i] = min(cube_dp[i], cube_dp[i - c] + 1)


def cube(n):
    return cube_dp[n]


def run_cube():
    prepare_dp_cube()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        print(cube(int(line)))


def backpack(weights, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w = weights[i - 1]
        for j in range(capacity + 1):
            s = dp[i - 1][j]
            if w <= j:
                s = max(s, dp[i - 1][j - w] + w)
            dp[i][j] = s

    return dp[n][capacity]


def run_backpack():
    s, n = map(int, input().split())
    weights = list(map(int, input().split()))

    print(backpack(weights, s))


def calculator(n: int):
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 0
    for i in range(2, n + 1):
        cur = dp[i - 1] + 1
        if i % 2 == 0:
            cur = min(cur, dp[int(i / 2)] + 1)
        if i % 3 == 0:
            cur = min(cur, dp[int(i / 3)] + 1)
        dp[i] = cur

    print(dp[n])
    print(*restore_answer(dp, n))


def restore_answer(dp, n):
    ans = [0] * (dp[n] + 1)
    i = dp[n]
    while n >= 1:
        ans[i] = n

        if n % 2 == 0 and dp[int(n / 2)] == i - 1:
            n = int(n / 2)
        elif n % 3 == 0 and dp[int(n / 3)] == i - 1:
            n = int(n / 3)
        else:
            n = n - 1
        i -= 1

    return ans


def run_calculator():
    n = int(input())
    calculator(n)


def three_subseq(a: list[int], b: list[int], c: list[int]):
    dp = [[[0] * (len(a) + 1) for _ in range(len(b) + 1)] for _ in range(len(c) + 1)]

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            for k in range(1, len(c) + 1):
                if a[i - 1] == b[j - 1] == c[k - 1]:
                    dp[k][j][i] = dp[k - 1][j - 1][i - 1] + 1
                else:
                    dp[k][j][i] = max(dp[k][j - 1][i], dp[k - 1][j][i], dp[k][j][i - 1])

    return dp[-1][-1][-1]


def run_three_subseq():
    input()
    a = list(map(int, input().split()))
    input()
    b = list(map(int, input().split()))
    input()
    c = list(map(int, input().split()))

    print(three_subseq(a, b, c))


def npp(a: list[int]):
    dp = [1] * len(a)

    for i in range(len(a)):
        for j in range(i):
            if a[i] % a[j] == 0:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def run_npp():
    n = int(input())
    a = list(map(int, input().split()))

    print(npp(a))


def post_office(n, m, x: list[int]):
    pref = [0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + x[i]

    def segment_cost(i2, j2):
        mid2 = (i2 + j2) // 2
        left = x[mid2] * (mid2 - i2 + 1) - (pref[mid2 + 1] - pref[i2])
        right = (pref[j2 + 1] - pref[mid2 + 1]) - x[mid2] * (j2 - mid2)
        return left + right

    cost = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            cost[i][j] = segment_cost(i, j)

    dp = [[math.inf] * (n + 1) for _ in range(m + 1)]
    prev = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, n + 1):
        dp[1][i] = cost[1][i]
        prev[1][i] = 0

    # основная динамика
    for k in range(2, m + 1):
        for i in range(k, n + 1):
            best = math.inf
            best_t = 0
            for t in range(k - 1, i):
                cur = dp[k - 1][t] + cost[t + 1][i]
                if cur < best:
                    best = cur
                    best_t = t
            dp[k][i] = best
            prev[k][i] = best_t

    print(dp[m][n])

    #restore
    offices = []
    k, i = m, n
    while k >= 1:
        t = prev[k][i]
        L, R = t + 1, i
        mid = (L + R) // 2
        offices.append(x[mid])  # отделение в медиане последнего блока
        i = t
        k -= 1

    offices.reverse()
    print(*offices)


def run_post_office():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))

    post_office(n, m, x)

def main():
    # run_rabbit()
    # run_cube()
    # run_backpack()
    # run_calculator()
    # run_three_subseq()
    # run_npp()
    run_post_office()



if __name__ == '__main__':
    main()
