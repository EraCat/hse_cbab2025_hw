def is_balanced(seq: str) -> bool:
    stack = []
    for c in seq:
        if c == '[':
            stack.append(c)
        elif c == '(':
            stack.append(c)
        elif c == '<':
            stack.append(c)
        elif c == '{':
            stack.append(c)
        else:
            if len(stack) == 0: return False
            end = stack.pop()
            if c == ']' and end != '[': return False
            if c == ')' and end != '(': return False
            if c == '>' and end != '<': return False
            if c == '}' and end != '{': return False
    return len(stack) == 0


def py_indent_are_correct(lines_data):
    if len(lines_data) == 0: return True
    if lines_data[-1][0] == "compound": return False
    if lines_data[0][1] != 0: return False

    stack = [0]
    prev_compound = False
    for t, indent in lines_data:
        top = stack[-1]

        if prev_compound and indent <= top:
            return False

        if indent > top:
            if not prev_compound:
                return False
            stack.append(indent)
        elif indent < top:
            while stack and stack[-1] > indent:
                stack.pop()
            if not stack or stack[-1] != indent:
                return False

        prev_compound = (t == "compound")

    return True


def rle(s):
    if len(s) == 0: return ''

    result = ''
    count = 1
    curr_char = s[0]
    for i in range(1, len(s)):
        if s[i] == curr_char:
            count += 1
        else:
            if count > 1:
                result += str(count) + curr_char
            else:
                result += curr_char
            count = 1
            curr_char = s[i]

    if count > 1:
        result += str(count) + curr_char
    elif count == 1:
        result += curr_char

    return result


def unrle(s):
    if len(s) == 0:
        return ''
    n = len(s)
    result = ''
    prev_group_char = None
    i = 0
    while i < n:
        c = s[i]

        if c.isdigit():
            if c == '0':
                return None

            count = 0
            while i < n and s[i].isdigit():
                count = count * 10 + int(s[i])
                i += 1
            if i >= n:
                return None

            ch = s[i]
            if not ch.isalpha():
                return None
            if count <= 1:
                return None
            if ch == prev_group_char:
                return None

            result += ch * count
            prev_group_char = ch
            i += 1
        else:
            if not c.isalpha():
                return None

            if c == prev_group_char:
                return None

            result += c
            prev_group_char = c
            i += 1

    return result


def build_str(commands):
    from collections import deque

    result = deque()
    for text, count in commands:
        if count > 0:
            result.append(text * count)
        else:
            result.appendleft(text * (-count))

    return ''.join(result)


def profile_sorted(data):
    count = 0

    def compare(x):
        nonlocal count
        count += 1
        return x

    res = sorted(data, key=compare)

    return res, count


def sort_substrs(s: str) -> int:
    n = len(s)
    k = n // 2 + 1

    idxs = list(range(k))

    def cmp(a: int, b: int) -> int:
        if a == b:
            return 0
        la = n - 2 * a
        lb = n - 2 * b

        slice_a = s[a: a + la]
        slice_b = s[b: b + lb]

        if slice_a < slice_b:
            return -1
        if slice_a > slice_b:
            return 1
        return 0

    from functools import cmp_to_key
    idxs.sort(key=cmp_to_key(cmp))

    total = 0
    cnt = (k + 1) // 2
    for i in idxs[::2]:
        total += n - 2 * i

    return round(total / cnt)


def count_non_tuples(data):
    count = 0

    # (obj, state): state=0 -> вход, state=1 -> выход (нужно для корректного in_path по спискам)
    stack = [(data, 0)]
    in_path_lists = set()  # id(list) для списков, которые сейчас в пути

    while stack:
        obj, state = stack.pop()

        if isinstance(obj, list):
            oid = id(obj)

            if state == 0:
                # список как элемент -> считаем его
                count += 1

                # цикл?
                if oid in in_path_lists:
                    return -1
                in_path_lists.add(oid)

                # событие "выхода" из списка
                stack.append((obj, 1))

                # разворачиваем элементы списка
                for x in reversed(obj):
                    stack.append((x, 0))
            else:
                in_path_lists.remove(oid)

        elif isinstance(obj, tuple):
            # кортеж НЕ считаем как элемент, просто раскрываем
            for x in reversed(obj):
                stack.append((x, 0))

        else:
            # всё остальное считаем как 1 и внутрь не лезем
            count += 1

    return count


def hash_func(p, n, s):
    """
    s - строка
    p - простое число (гарантируется)
    n - натуральное число (гарантируется)
    """

    h = 0
    for ch in s:
        h = (h * p + ord(ch)) % n

    return h

def main():
    input_seq = input()
    # print(is_balanced(input_seq))
    # print(rle(input_seq))
    # print(unrle(input_seq))
    # print(profile_sorted([5,2,3,2,1]))
    # print(sort_substrs(input_seq))
    print(hash_func(101, 1000000007, input_seq))


if __name__ == "__main__":
    main()
