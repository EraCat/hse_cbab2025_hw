import math

import numpy as np


def kstat_gen(source, k):
    # items: [ [tuple, read_index, first_yield_index_or_None], ... ]
    items = []
    read_i = 0
    yield_i = 0

    def sort_key(item):
        t, r, first_y = item
        x = t[0]

        yielded_flag = 0 if first_y is not None else 1
        first_y_val = first_y if first_y is not None else 0
        return -x, yielded_flag, first_y_val, r

    for t in source:
        items.append([t, read_i, None])
        read_i += 1

        if len(items) < k:
            continue

        items.sort(key=sort_key)
        kth = items[k - 1]  # k-я по величине (в 0-индексации)

        if len(items) == k:
            yield items[-1][0]

        if kth[2] is None:  # первый раз порождаем этот элемент
            kth[2] = yield_i
            yield_i += 1

        yield kth[0]


def guess(number_func):
    upb = yield

    bad = 0
    while not (type(upb) is int and upb > 10):
        bad += 1
        if bad <= 3:
            upb = yield "?"
        else:
            yield "bye"
            return

    secret = number_func(upb)
    max_attempts = int(math.log2(upb)) + 2

    attempts = 0
    x = yield "start"

    while True:
        attempts += 1

        if type(x) is not int:
            # неверный тип — "?" и это считается попыткой
            if attempts == max_attempts:
                x = yield "?"
                yield "bye"
                return
            x = yield "?"
            continue

        if x == secret:
            yield "ok"
            return

        resp = "<" if x < secret else ">"

        if attempts == max_attempts:
            x = yield resp
            yield "bye"
            return

        x = yield resp


import heapq


def schedule(tasks):
    it = iter(tasks)
    backlog = []  # heap of (-priority, start_time, estimate, index, reality)

    cur_time = 0
    next_item = None  # (index, task)
    idx = 0

    def take_next():
        nonlocal next_item, idx
        if next_item is None:
            try:
                t = next(it)
            except StopIteration:
                return False
            next_item = (idx, t)
            idx += 1
        return True

    def push_next():
        nonlocal next_item
        i, t = next_item
        heapq.heappush(backlog, (-t.priority, t.start_time, t.estimate, i, t.reality))
        next_item = None

    if not take_next():
        return

    while backlog or next_item is not None:
        while next_item is not None and next_item[1].start_time <= cur_time:
            push_next()
            if not take_next():
                break

        if not backlog:
            if next_item is None:
                return

            st = next_item[1].start_time
            if st > cur_time:
                cur_time = st

            while next_item is not None and next_item[1].start_time <= cur_time:
                push_next()
                if not take_next():
                    break

        neg_p, st, est, i, reality = heapq.heappop(backlog)

        start_work = cur_time if st <= cur_time else st
        yield (i, start_work)

        cur_time = start_work + reality


def join(data1, data2):
    END = object()

    def nxt(it):
        try:
            return next(it)
        except StopIteration:
            return END

    it1, it2 = iter(data1), iter(data2)
    a = nxt(it1)
    b = nxt(it2)

    while a is not END and b is not END:
        if a < b:
            # двигаем первый до a >= b (или конца)
            while a is not END and a < b:
                a = nxt(it1)
            continue

        if b < a:
            # двигаем второй до b >= a (или конца)
            while b is not END and b < a:
                b = nxt(it2)
            continue

        v = a

        c1 = 1
        c2 = 1
        emitted = 0

        a = nxt(it1)
        b = nxt(it2)

        turn = 0

        while True:
            need = c1 * c2
            while emitted < need:
                yield v
                emitted += 1

            progressed = False

            if turn == 0:
                if a == v:
                    c1 += 1
                    a = nxt(it1)
                    progressed = True
                elif b == v:
                    c2 += 1
                    b = nxt(it2)
                    progressed = True
            else:
                if b == v:
                    c2 += 1
                    b = nxt(it2)
                    progressed = True
                elif a == v:
                    c1 += 1
                    a = nxt(it1)
                    progressed = True

            turn ^= 1

            if not progressed:
                break

    return


def dedup(data):
    prev = set()
    for x in data:

        if x not in prev:
            yield x
            prev.add(x)


class NonPositiveError(Exception):
    pass


class PositiveList(list):

    def append(self, v):
        if v < 0:
            raise NonPositiveError()
        super().append(v)


def main():
    g = guess(lambda v: v // 3)
    next(g)

    print(g.send('123'))
    print(g.send('123'))
    print(g.send('123'))

    print(g.send(20))  # start
    print(g.send('123'))
    print(g.send(20))  # >
    print(g.send(10))  # >
    print(g.send(0))  # <
    print(g.send(-6))  # <
    print(g.send(-100))  # <
    print(g.send(6))  # bye


if __name__ == '__main__':
    main()
