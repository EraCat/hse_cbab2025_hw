import random
import sys


def task_1():
    x, y, r = [], [], []
    x_min, x_max, y_min, y_max = 0, 0, 0, 0
    for _ in range(3):
        x1, y1, r1 = map(float, input().split())
        x.append(x1)
        y.append(y1)
        r.append(r1)
        x_min = min(x_min, x1 - r1)
        x_max = max(x_max, x1 + r1)
        y_min = min(y_min, y1 - r1)
        y_max = max(y_max, y1 + r1)

    rect_area = (x_max - x_min) * (y_max - y_min)

    counter = 0
    samples = 100000
    for _ in range(samples):
        a = random.uniform(x_min, x_max)
        b = random.uniform(y_min, y_max)

        in_circles = True
        for i in range(3):
            if (a - x[i]) ** 2 + (b - y[i]) ** 2 > r[i] ** 2:
                in_circles = False
                break
        if in_circles:
            counter += 1

    ans = rect_area * counter / samples
    print(ans)



def parity(x):
    p = 0
    while x:
        p ^= 1
        x &= x - 1
    return p

def task_2():
    data = sys.stdin.buffer.read().split()
    pos = 0

    n = int(data[pos])
    pos += 1

    hex_len = (n + 3) // 4
    shift = 4 * hex_len - n

    def read_matrix():
        nonlocal pos
        matrix = []
        for _ in range(n):
            matrix.append(int(data[pos], 16) >> shift)
            pos += 1
        return matrix

    A = read_matrix()
    B = read_matrix()
    C = read_matrix()

    print("YES" if check_mat(A, B, C, n) else "NO")


def mult_vec_and_mat(A, x):
    res = 0
    for row in A:
        res = (res << 1) | parity(row & x)

    return res

def check_mat(A, B, C, n):
    getrandbits = random.getrandbits
    mul = mult_vec_and_mat

    x = (1 << n) - 1
    for _ in range(1):
        if mul(A, mul(B, x)) != mul(C, x):
            return False
        x = getrandbits(n)

    return True


def main():
    task_2()


if __name__ == "__main__":
    main()
