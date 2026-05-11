def gcd(a: int, b: int) -> int:
    a, b = sorted((a, b), reverse=True)
    while b > 0:
        a, b = b, a % b
    return a
