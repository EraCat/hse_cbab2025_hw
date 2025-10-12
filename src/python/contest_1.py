def is_prime_number(n) -> bool | None:
    if n is not int:
        return None

    if n <= 0:
        return None

    if n < 2:
        return False

    return True


def main():
    print(is_prime_number(1))


if __name__ == "__main__":
    main()
