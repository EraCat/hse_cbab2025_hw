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


#     1 - *2
# 2 *3
# 1


def main():
    # run_rabbit()
    # run_cube()
    # run_backpack()
    run_calculator()


if __name__ == '__main__':
    main()
