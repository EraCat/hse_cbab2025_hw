def task_1(n: int):
    prev = 0
    cur = 1
    for i in range(n):
        temp = cur
        cur = prev + cur
        prev = temp

    return cur


def task_2(a: list):
    for i in range(len(a) - 1, -1, -1):
        print(a[i], end=' ')


def task_4(n: int):
    s = None
    a_m, b_m, c_m = 0, 0, 0

    a = 1
    while a * a * a <= n:
        if n % a == 0:
            m = n // a
            b = a

            while b * b <= m:
                if m % b == 0:
                    c = m // b
                    s_c = 2 * (a * b + b * c + a * c)
                    if s is None or s_c < s:
                        s = s_c
                        a_m = a
                        b_m = b
                        c_m = c
                b += 1
        a += 1

    print(s, a_m, b_m, c_m)

    return s


def main():
    n = int(input())

    task_4(n)


if __name__ == '__main__':
    main()
