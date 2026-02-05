


class Node:
    def __init__(self):
        self.to = {}
        self.term_idx = -1

    def update(self):
        pass


class Bor:
    def __init__(self, root: Node):
        self.root = root
        self.max_len = 0

    def add_string(self, s: str, idx: int):
        self.max_len = max(self.max_len, len(s))

        v = self.root
        v += 1
        for c in s:
            if c not in v.to:
                v.to[c] = Node()
            v.number += 1
            v = v.to[c]


        v.term_idx = idx

    def scan_text(self, text: str, found):
        n = len(text)
        L = self.max_len
        for i in range(n):
            v = self.root
            end = i + L
            if end > n:
                end = n
            for j in range(i, end):
                v = v.to.get(text[j])
                if v is None:
                    break
                if v.term_idx != -1:
                    found[v.term_idx] = True



def task_1():
    s = input()

    bor = Bor(Node())

    m = int(input())
    for idx in range(m):
        word = input()
        bor.add_string(word, idx)
    
    found = [False] * m
    bor.scan_text(s, found)


    for f in found:
        print('Yes') if f else print('No')


class NodeWithChildNumber(Node):
    def __init__(self):
        self.to = {}
        self.terms = 0
        self.subs = 0

class BorWithKstat:
    def __init__(self, root: NodeWithChildNumber):
        self.root = root
        self.max_len = 0

    def add_string(self, s: str):
        v = self.root
        v.subs += 1
        for c in s:
            if c not in v.to:
                v.to[c] = NodeWithChildNumber()
            v = v.to[c]
            v.subs += 1

        v.terms +=1

    def kth_stat(self, k):
        v = self.root
        res = []

        while True:
            if v.terms > 0:
                if k <= v.terms:
                    return "".join(res)
                k-=v.terms

            for c_id in range(ord('a'), ord('z') +1):
                c = chr(c_id)
                u = v.to.get(c)
                if u is None:
                    continue

                if k > u.subs:
                    k-= u.subs
                else:
                    res.append(c)
                    v = u
                    break


    
def task_2():
    n = int(input())

    bor = BorWithKstat(NodeWithChildNumber())

    for _ in range(n):
        line = input()
        if line.isdigit():
            k = int(line)
            print(bor.kth_stat(k))
        else:
            bor.add_string(line)


def main():
    # task_1()
    task_2()


if __name__ == '__main__':
    main()