from bisect import bisect_right, bisect_left


def print_binomial(start, left, current):
    if left == 0:
        print(*current)
        return

    for i in range(start, left - 1, -1):
        current.append(i)
        print_binomial(i - 1, left - 1, current)
        current.pop()


best = 0.0


def actek_gold(max_w, items):
    brute(0, max_w, 0, items)
    return best


def brute(i, cap, cur, items):
    global best
    if i == len(items):
        best = max(best, cur)
        return
    w, c = items[i]
    if w <= cap:
        brute(i + 1, cap - w, cur + c, items)
    brute(i + 1, cap, cur, items)


def task_2():
    line_1 = input().split()
    n = int(line_1[0])
    W = float(line_1[1])
    items = []
    for i in range(n):
        items.append(list(map(float, input().split())))

    print(f"{actek_gold(W, items):.9f}")


def task_1():
    n, k = map(int, input().split())
    print_binomial(n, k, [])


def task_3():
    line_1 = input().split()
    n = int(line_1[0])
    W = int(line_1[1])
    items = []
    for i in range(n):
        items.append((map(float, input().split())))

    print(f"{gold_sand(W, items):.3f}")


def gold_sand(capacity, items):
    ans = 0.0
    rest = []
    for c, w in items:
        if w == 0:
            ans += c
        else:
            rest.append((c, w))

    rest.sort(key=lambda t: t[0] / t[1], reverse=True)

    for c, w in rest:
        if capacity >= w:
            ans += c
            capacity -= w
        else:
            ans += c * capacity / w
            capacity = 0
            break

    return ans


def task_4():
    n = int(input())

    split_to_terms(n)


def split_to_terms(n):
    cur = []

    def dfs(rest: int, prev: int):
        if rest == 0:
            print(*cur)
            return
        for x in range(1, min(prev, rest) + 1):
            cur.append(x)
            dfs(rest - x, x)
            cur.pop()

    dfs(n, n)


def task_5():
    n, m = map(int, input().split())
    left = []
    right = []
    for i in range(n):
        l, r = map(int, input().split())
        if l > r:
            left.append(r)
            right.append(l)
        else:
            left.append(l)
            right.append(r)

    point = list(map(int, input().split()))

    points_and_segments(left, right, point)


def points_and_segments(left_s_p, right_s_p, points):
    left_s_p.sort()
    right_s_p.sort()

    ans = []
    for x in points:
        count_l = bisect_right(left_s_p, x)
        count_r = bisect_left(right_s_p, x)
        ans.append(count_l - count_r)

    print(*ans)


def task_6():
    n = int(input())
    segments = []
    for i in range(n):
        l, r = map(int, input().split())
        segments.append((l, r))

    cover_points(segments)


def cover_points(segments):
    segments.sort(key=lambda t: t[1])

    points = []
    current = -1
    for l,r in segments:
        if current < l:
            current = r
            points.append(current)

    print(len(points))
    print(*points)


def main():
    # task_2()
    # task_3()
    # task_4()
    # task_5()
    task_6()


if __name__ == '__main__':
    main()
