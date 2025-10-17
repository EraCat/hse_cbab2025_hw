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
                leading_zero_counter+=1
            else:
                first_zero_digits = False

        x = x * 10 + y % 10
        y //= 10


    if leading_zero_counter > 0:
        return x, leading_zero_counter

    return x


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
    print(invert_digits(123456789))
    print(invert_digits(89976))
    print(invert_digits(0))
    print(invert_digits(1020000000))


if __name__ == "__main__":
    main()
