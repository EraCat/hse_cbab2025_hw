
def count_sort(order, c, clr):
    n = len(order)
    cnt = [0] * clr
    for x in order:
        cnt[c[x]] += 1

    pos = [0] * clr
    for i in range(1, clr):
        pos[i] = pos[i - 1] + cnt[i - 1]

    new_order = [0] * n
    for x in order:
        cl = c[x]
        new_order[pos[cl]] = x
        pos[cl] += 1

    return new_order


def task_1():
    s = input()
    
    s += '\0'
    n = len(s)

    p = [0] * n
    c = [0] * n

    max_ch = 128

    count = [0] * max_ch
    for ch in s:
        count[ord(ch)] += 1

    pos = [0] * max_ch
    for i in range(1, max_ch):
        pos[i] = pos[i-1] + count[i-1]

    for i, ch in enumerate(s):
        code = ord(ch)
        p[pos[code]] = i
        pos[code] += 1


    clr = 1
    c[p[0]] = 0
    for i in range(1, n):
        if s[p[i]] != s[p[i - 1]]:
            clr += 1
        c[p[i]] = clr - 1

    k = 0
    while (1 << k) < n:
        shift = 1 << k

        p = [(x - shift) % n for x in p]
        p = count_sort(p, c, clr)

        new_c = [0] * n
        new_c[p[0]] = 0
        new_clr = 1

        for i in range(1, n):
            cur = p[i]
            prev = p[i - 1]

            cur_pair = (c[cur], c[(cur + shift) % n])
            prev_pair = (c[prev], c[(prev + shift) % n])

            if cur_pair != prev_pair:
                new_clr += 1
            new_c[cur] = new_clr - 1

        c = new_c
        clr = new_clr
        k += 1

    return p[1:]











def main():
    sa = task_1()
    print(*(x + 1 for x in sa))


if __name__== '__main__':
    main()