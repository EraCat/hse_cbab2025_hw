from bisect import bisect_left, bisect_right


def task1(arr: list[int]) -> tuple[int, int]:
    if arr[0] == 1:
        return -1, 0
    if arr[-1] == 0:
        return len(arr) - 1, len(arr)

    l = 0
    r = len(arr)

    while l < r:
        mid = (l + r) // 2
        if arr[mid] == 1:
            r = mid
        else:
            l = mid + 1

    return l - 1, r


def run_task1():
    n = int(input())
    arr = list(map(int, input()))
    print(*task1(arr))


def task2(k: int, arr: list[int]) -> int:
    l = 0
    r = len(arr)

    while l < r:
        mid = (l + r) // 2
        if arr[mid] < k:
            l = mid + 1
        elif arr[mid] > k:
            r = mid
        else:
            return mid

    return -1


def run_task2():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for bi in b:
        if task2(bi, a) == -1:
            print('NO')
        else:
            print('YES')


def task3(arr, l, r):
    left = bisect_left(arr, l)
    right = bisect_right(arr, r)

    return right - left


def run_task3():
    n = int(input())
    arr = list(map(int, input().split()))
    arr.sort()
    k = int(input())

    res = []
    for _ in range(k):
        l, r = map(int, input().split())
        res.append(task3(arr, l, r))

    print(*res)


def task4(arr, k):
    l = 0
    r = max(arr)

    while l < r:
        m = (l + r + 1 ) // 2
        count = 0
        for item in arr:
            count += item // m

        if count < k:
            r = m - 1
        else:
            l = m

    return l


def run_task4():
    n, k = map(int, input().split())
    arr = []
    for _ in range(n):
        arr.append(int(input()))

    print(task4(arr, k))


def main():
    # run_task1()
    # run_task2()
    # run_task3()
    run_task4()


if __name__ == '__main__':
    main()
