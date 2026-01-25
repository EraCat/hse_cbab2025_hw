import random
from typing import Tuple


class TreeNode:
    def __init__(self, value, priority=None):
        self.size = 1
        self.value = value
        self.priority = priority if priority is not None else random.random()
        self.left = None
        self.right = None
        self.rotate = False
        self.sum = self.value

    def update(self):
        self.size = 1 + Treap.size(self.left) + Treap.size(self.right)
        self.sum = self.value
        if self.left:
            self.sum += self.left.sum
        if self.right:
            self.sum += self.right.sum

    def delayed_update(self):
        if self.rotate:
            if self.left:
                self.left.rotate ^= True
            if self.right:
                self.right.rotate ^= True
            self.left, self.right = self.right, self.left
        self.rotate = False


class Treap:
    def __init__(self, root=None):
        self.root = root

    def build_from_array(self, arr):
        for x in arr:
            self.root = self.merge(self.root, TreeNode(x))

    @staticmethod
    def size(t: TreeNode):
        return 0 if t is None else t.size

    def add(self, i, x):
        if self.root is None:
            self.root = TreeNode(x)
            return

        l, r = self.split(self.root, i)
        self.root = self.merge(self.merge(l, TreeNode(x)), r)
        self.root.update()

    def delete(self, i):
        l, r = self.split(self.root, i + 1)
        l, m = self.split(l, i)
        self.root = self.merge(l, r)
        if self.root:
            self.root.update()

    @staticmethod
    def merge(l, r):
        if not l:
            return r
        if not r:
            return l

        l.delayed_update()
        r.delayed_update()
        if l.priority > r.priority:
            l.right = Treap.merge(l.right, r)
            l.update()
            return l
        else:
            r.left = Treap.merge(l, r.left)
            r.update()
            return r

    @staticmethod
    def split(curr: TreeNode, i) -> Tuple[TreeNode, TreeNode]:
        if curr is None:
            return None, None
        curr.delayed_update()
        left_size = Treap.size(curr.left)
        if i <= left_size:
            l, mid = Treap.split(curr.left, i)
            curr.left = mid
            curr.update()
            return l, curr
        else:
            mid, r = Treap.split(curr.right, i - left_size - 1)
            curr.right = mid
            curr.update()
            return curr, r

    def to_arr(self):
        arr = []

        def dfs(t):
            if not t:
                return
            dfs(t.left)
            arr.append(t.value)
            dfs(t.right)

        dfs(self.root)
        return arr


def task_1():
    n, m = map(int, input().split())
    arr = list(map(int, input().split()))

    treap = Treap()
    treap.build_from_array(arr)

    for _ in range(m):
        cmd = input().split()
        args = list(map(int, cmd[1:]))
        if cmd[0] == 'add':
            i = args[0]
            x = args[1]
            treap.add(i, x)
        if cmd[0] == 'del':
            i = args[0]
            treap.delete(i - 1)

    print(0 if treap.root is None else treap.root.size)
    print(*treap.to_arr())


def task_2():
    n, m = map(int, input().split())

    treap = Treap()
    arr = [i for i in range(1, n + 1)]
    treap.build_from_array(arr)

    for _ in range(m):
        l, r = map(int, input().split())
        t1, t2 = Treap.split(treap.root, l - 1)
        t3, t4 = Treap.split(t2, r - l + 1)
        t = treap.merge(t3, t1)
        new_root = treap.merge(t, t4)
        treap = Treap(new_root)

    print(*treap.to_arr())


def task_3():
    n, m = map(int, input().split())
    arr = list(map(int, input().split()))
    treap = Treap()
    treap.build_from_array(arr)

    for _ in range(m):
        q, l, r = map(int, input().split())

        t1, t2 = Treap.split(treap.root, l - 1)
        t3, t4 = Treap.split(t2, r - l + 1)

        if q == 0:
            print(t3.sum)
        elif q == 1:
            t3.rotate ^= True

        root = Treap.merge(Treap.merge(t1, t3), t4)
        treap = Treap(root)


def main():
    # task_1()
    # task_2()
    task_3()


if __name__ == '__main__':
    main()
