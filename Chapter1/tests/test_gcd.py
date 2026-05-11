from gcd import gcd
import random
import math


def test_gcd():
    for _ in range(1000):
        a = random.randint(1, 1000)
        b = random.randint(1, 1000)
        assert gcd(a, b) == math.gcd(a, b)
