


from collections import deque
import sys


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



def prefix_fun(s: str):
    n = len(s)
    d = [0] * n
    for i in range(1, n):
        curr_len = d[i-1]
        while curr_len > 0 and s[i] != s[curr_len]:
            curr_len = d[curr_len - 1]
        if s[i] == s[curr_len]:
            curr_len +=1
        d[i] = curr_len

    return d


def task_3():
    s = input()
    print(sum(prefix_fun(s)))


def task_4():
    n = int(input())
    z = list(map(int, input().split()))

    d = [0] * n

    for i in range(1, n):
        cur_len = z[i]
        for j in range(cur_len - 1, -1 ,-1):
            pos = i+j
            if d[pos] != 0:
                break
            d[pos] = j+1

    return d



def task_5():
    class Node:
        def __init__(self):
            self.to = {}
            self.link = 0
            self.out = False

    class AhoCorasick:
        def __init__(self):
            self.nodes = [Node()]

        def add(self, word: bytes):
            v = 0
            for c in word:
                next_id = self.nodes[v].to.get(c)
                if next_id is None:
                    next_id = len(self.nodes)
                    self.nodes[v].to[c] = next_id
                    self.nodes.append(Node())
                v = next_id
            self.nodes[v].out = True

        def build(self):
            q = deque()
            root = 0

            for ch, u in self.nodes[root].to.items():
                self.nodes[u].link = root
                q.append(u)

            while q:
                v = q.popleft()
                link_v = self.nodes[v].link

                if self.nodes[link_v].out:
                    self.nodes[v].out = True

                for ch, u in self.nodes[v].to.items():
                    j = link_v
                    while j != 0 and ch not in self.nodes[j].to:
                        j = self.nodes[j].link
                    if ch in self.nodes[j].to:
                        self.nodes[u].link = self.nodes[j].to[ch]
                    else:
                        self.nodes[u].link = 0
                    q.append(u)

        def contains_any(self, text: bytes):
            v = 0
            for ch in text:
                while v != 0 and ch not in self.nodes[v].to:
                    v = self.nodes[v].link
                nxt = self.nodes[v].to.get(ch)
                if nxt is not None:
                    v = nxt
                if self.nodes[v].out:
                    return True
            return False



    # n = int(input())

    # aho = AhoCorasick()
    # for _ in range(n):
    #     aho.add(input())

    # aho.build()



    # data = []
    # while True:
    #     try:
    #         s = input()
    #     except EOFError:
    #         break
    #     if s == "":
    #         break
    #     data.append(s)

    # for s in data:
    #     if aho.contains_any(s):
    #         print(s)
    buf = sys.stdin.buffer
    out = sys.stdout.buffer

    first = buf.readline()
    if not first:
        return
    n = int(first)

    aho = AhoCorasick()
    for _ in range(n):
        pat = buf.readline().rstrip(b"\r\n")
        aho.add(pat)

    aho.build()
    buf = sys.stdin.buffer
    for raw in buf:
        if raw and aho.contains_any(raw):
            out.write(raw)




def main():
    # task_1()
    # task_2()
    # task_3()
    # print(*task_4())
    task_5()


if __name__ == '__main__':
    main()
