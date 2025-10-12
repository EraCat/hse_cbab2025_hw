def task1(arr: list[int]) -> tuple[int, int]:
    if arr[0] == 1:
        return -1, 0
    if arr[-1] == 0:
        return len(arr) - 1, len(arr)

    l = 0
    r = len(arr)

    while l < r - 1:
        mid = (l + r) // 2
        if arr[mid] == 0:
            l = mid
        else:
            r = mid

    return l, r


def task2(k: int, arr: list[int]) -> int:
    l = 0
    r = len(arr)

    while l < r:
        mid = (l + r) // 2
        if arr[mid] < k:
            l = mid +1
        elif arr[mid] > k:
            r = mid
        else:
            return mid

    return -1


def main():
    # n = int(input())
    # arr = list(map(int, input()))
    # l, r = task1(arr)
    # print(l, r)

    data = list(map(int, input().split()))
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for bi in b:
        if task2(bi, a) == -1:
            print('NO')
        else:
            print('YES')


if __name__ == '__main__':
    main()
