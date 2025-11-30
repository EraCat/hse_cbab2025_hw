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
    pass


def main():
    input_seq = input()
    # print(is_balanced(input_seq))
    # print(rle(input_seq))
    print(unrle(input_seq))


if __name__ == "__main__":
    main()
