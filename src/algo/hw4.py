# 1
def sort_by_count(queries: list[int]):
    arr = [0] * 100

    for q in queries:
        arr[q - 1] = arr[q - 1] + 1
        index = -1
        for i in range(q - 1, -1, -1):
            index += arr[i]
        print(index, end=' ')


def dictionary(queries: list[tuple[str, int]]):
    d = {}
    for i, q in enumerate(queries):
        if q[0] == '+':
            d[q[1]] = d.get(q[1], 0) + 1
        elif q[0] == '-':
            if d.get(q[1], 0) == 0:
                print(f"query #{i + 1}: can not delete {q[1]}")
            else:
                d[q[1]] = d.get(q[1], 0) - 1
                if d[q[1]] == 0:
                    print(f"after query {i + 1} number {q[1]} disappeared")


def call_dictionary():
    n = input()
    data = []
    for i in range(int(n)):
        split = input().split()
        data.append((split[0], int(split[1])))
    dictionary(data)


def swap_result(queries: list[tuple[int, int]]):
    d = {}
    for q in queries:
        print(abs(d.get(q[0], q[0]) - d.get(q[1], q[1])))
        temp = d.get(q[0], q[0])
        d[q[0]] = d.get(q[1], q[1])
        d[q[1]] = temp


def call_swap():
    n = input()
    data = []
    for i in range(int(n)):
        split = input().split()
        data.append((int(split[0]), int(split[1])))
    swap_result(data)


def restruct_graph():
    ev = input().split()
    vertices_num = int(ev[0])
    e = int(ev[1])
    d = {}
    for i in range(e):
        split = input().split()
        a = int(split[0])
        b = int(split[1])
        if a > b:
            edges = d.get(b, set())
            edges.add(a)
            d[b] = edges
        elif a < b:
            edges = d.get(a, set())
            edges.add(b)
            d[a] = edges

    e = 0
    for v in d.values():
        e += len(v)

    print(vertices_num, e)
    for k in d:
        for vertices_num in d[k]:
            print(k, vertices_num)



def main():
    # call_dictionary()
    # call_swap()
    restruct_graph()


if __name__ == '__main__':
    main()