import cmath
import math


def fft(p):
    n = len(p)
    if n == 1:
        return p

    w = cmath.exp(2 * math.pi * 1j / n)
    pe = p[::2]
    po = p[1::2]
    ye = fft(pe)
    yo = fft(po)
    y = [0j] * n
    for j in range(n // 2):
        wj = w**j
        y[j] = ye[j] + wj * yo[j]
        y[j + n // 2] = ye[j] - wj * yo[j]

    return y


def ifft(p):
    n = len(p)
    if n == 1:
        return p

    w = cmath.exp(-2 * math.pi * 1j / n)
    pe = p[::2]
    po = p[1::2]
    ye = ifft(pe)
    yo = ifft(po)
    y = [0j] * n
    for j in range(n // 2):
        wj = w**j
        y[j] = (ye[j] + wj * yo[j]) / 2
        y[j + n // 2] = (ye[j] - wj * yo[j]) / 2

    return y


def task_3():
    s1 = input()
    s2 = input()

    p = [int(c) for c in s1[::-1]]
    q = [int(c) for c in s2[::-1]]

    n = 1
    while n < len(p) + len(q):
        n <<= 1

    p += [0] * (n - len(p))
    q += [0] * (n - len(q))

    yp = fft(p)
    yq = fft(q)

    y = [0j] * n
    for i in range(n):
        y[i] = yp[i] * yq[i]

    res = ifft(y)
    res = [round(x.real) for x in res]

    carry = 0
    for i in range(n):
        total = res[i] + carry
        res[i] = total % 10
        carry = total // 10

    while carry > 0:
        res.append(carry%10)
        carry //= 10

    while len(res) > 1 and res[-1] == 0:
        res.pop()

    print(''.join(map(str, res[::-1])))


def task_1():
    n, k, m = map(int, input().split())
    p = list(map(int, input().split()))

    b = [0] * (n + k)
    for i in range(n):
        b[i] = (b[i] - p[i]) % m
        b[i + k] = p[i] % m

    print(*b)


def task_2():
    n, k, m = map(int, input().split())
    p = list(map(int, input().split()))

    if n <= k:
        r = p + [0] * (k - n)
        return [0], r

    a = [0] * (n - k)
    for i in range(n - 1, k - 1, -1):
        a[i - k] = p[i] % m
        p[i - k] = (p[i - k] + a[i - k]) % m
        p[i] = 0

    r = p[:k]

    return a, r


def main():
    task_3()


if __name__ == "__main__":
    main()
