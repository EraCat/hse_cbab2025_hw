# 1
def bits_seqs(n):
    if n == 0:
        return 0

    count = 0

    if n > 0:
        x = n
        k = x.bit_length()
        prev = 1
        for _ in range(k):
            b = x & 1
            if b == 0 and prev == 1:
                count += 1
            prev = b
            x >>= 1
    else:
        x = n
        k = x.bit_length() - 1
        prev = 0
        for _ in range(k):
            b = x & 1
            if b == 1 and prev == 0:
                count += 1
            prev = b
            x >>= 1

    return count


# 2
def invert_digits(n):
    y = n
    x = 0
    leading_zero_counter = 0
    first_zero_digits = True

    while y > 0:
        if first_zero_digits:
            if y % 10 == 0:
                leading_zero_counter += 1
            else:
                first_zero_digits = False

        x = x * 10 + y % 10
        y //= 10

    if leading_zero_counter > 0:
        return x, leading_zero_counter

    return x


def varint_encode(n):
    result = bytearray()
    if n == 0:
        result.append(0)
        return result

    while n >= 128:
        part = n & 127
        n = n >> 7
        if n > 0:
            result.append(part | 128)

    result.append(n)

    return result


def zigzag_encode(n):
    n = n * 2
    if n < 0:
        n = abs(n + 1)

    return varint_encode(n)


def varint_decode(byte_arr) -> list[int | None]:
    if len(byte_arr) == 0:
        return []

    result = []
    val = 0
    shift = 0
    in_number = False

    for b in byte_arr:
        in_number = True
        val |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
        else:
            result.append(val)
            val = 0
            shift = 0
            in_number = False

    if in_number:
        result.append(None)

    return result


def zigzag_decode(byte_arr):
    decoded = varint_decode(byte_arr)

    for i in range(len(decoded)):
        if decoded[i] is not None:
            if decoded[i] % 2 == 0:
                decoded[i] = decoded[i] // 2
            else:
                decoded[i] = -(decoded[i] + 1) // 2

    return decoded


def ba_dedup(arr):
    if len(arr) <= 1:
        return None

    new_arr = []

    prev = arr[0]
    new_arr.append(arr[0])
    for i in range(1, len(arr)):
        if arr[i] != prev:
            new_arr.append(arr[i])

        prev = arr[i]

    if len(new_arr) < len(arr):
        new_arr.append(255)

    while len(arr) > len(new_arr):
        max_i = len(new_arr)
        i = 0
        while len(arr) > len(new_arr) and i < max_i:
            new_arr.append(new_arr[i])
            i += 1

    for i in range(len(new_arr)):
        arr[i] = new_arr[i]

    return None


def is_prime_number(n) -> bool | None:
    if not isinstance(n, int):
        return None

    if n <= 0:
        return None

    if n == 1:
        return False

    if n == 2:
        return True

    if n % 2 == 0:
        return False

    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2

    return True


def utf8_size_hist(src):
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0

    index = -1
    start_index = -1
    seq_len = 0
    expected = 0
    for byte in src:
        index += 1
        if expected == 0:
            if (byte & 0xF8) == 0xF0:
                expected = 3
                seq_len = 4
                start_index = index
            elif (byte & 0xF0) == 0xE0:
                expected = 2
                seq_len = 3
                start_index = index
            elif (byte & 0xE0) == 0xC0:
                expected = 1
                seq_len = 2
                start_index = index
            elif (byte & 0x80) == 0x00:
                count_1 += 1
            else:
                return (count_1, count_2, count_3, count_4), index
        else:
            if (byte & 0xC0) == 0x80:
                expected -= 1
                if expected == 0:
                    if seq_len == 4:
                        count_4 += 1
                    elif seq_len == 3:
                        count_3 += 1
                    elif seq_len == 2:
                        count_2 += 1
            else:
                return (count_1, count_2, count_3, count_4), start_index

    if expected > 0:
        return (count_1, count_2, count_3, count_4), start_index

    return (count_1, count_2, count_3, count_4), -1


def division_steps(a: int, b: int) -> list[str]:
    a_bin = format(a, 'b')
    b_bin = format(b, 'b')

    lines = [f"{a_bin}/{b_bin}"]

    result = ""
    rem = a

    if rem == 0:
        lines.append("0")
        return lines

    while True:
        if rem < b:
            rem <<= 1
            if result == "":
                result = "0."
            else:
                result += "0"
            lines.append(f"*: {format(rem, 'b')}/{b_bin} {result}")
        else:
            rem = (rem - b) << 1
            if result == "":
                result = "0.1"
            else:
                result += "1"
            lines.append(f"-: {format(rem, 'b')}/{b_bin} {result}")

        if rem == 0:
            break

    lines.append(result)
    return lines


def main():
    print(is_prime_number(1))

    input = [-1, -2, -3, -4, -5, -11, 18, 0]
    for n in input:
        print(bits_seqs(n))

    l = [1, 1, 1, 1, 1, 1, 1]
    l = [1, 2, 3, 4, 5, 5, 1]
    ba_dedup(l)
    print(l)

    print(is_prime_number(5))
    print(invert_digits(1020000000))
    print(varint_encode(127))
    print(varint_encode(128))
    print(varint_encode(129))
    print(varint_encode(256))
    print(varint_encode(300))
    print(varint_decode(varint_encode(300)))
    print(varint_decode(varint_encode(127)))

    print(utf8_size_hist([12, 23, 34]))
    print(utf8_size_hist([12, 23, 34, 128, 00]))
    print(utf8_size_hist([12, 23, 34, 192, 128, 128]))
    print(utf8_size_hist(
        [192, 128, 224, 128, 129, 224, 129, 130, 240, 130, 131, 132, 240, 131, 132, 133, 240, 132, 133, 134, 0, 1, 2,
         3]))

    steps = division_steps(7, 11)
    for step in steps:
        print(step)


if __name__ == "__main__":
    main()
