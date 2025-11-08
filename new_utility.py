import numpy as np
from numba import njit

tau = np.pi * 2


@njit(inline='always')
def is_power_of_two(n):
    return (n & (n - 1)) == 0 and n > 0


@njit(inline='always')
def highest_power_of_2_dividing(n):
    return n & (~(n - 1))


@njit(inline='always')
def identity(x):
    return x


class Ticker:
    def __init__(self, value=0):
        self.count = value

    def increment(self):
        self.count += 1

    def reset(self, value=0):
        self.count = value

    def read(self):
        return self.count

    def __repr__(self):
        return f'Ticker({self.count})'


class DSU:
    def __init__(self):
        self.rank = 0
        self.parent = self

    def find(self):
        if self.parent is self:
            return self
        root = self.parent.find()
        self.parent = root
        return root

    def union(x, y):
        x = x.find()
        y = y.find()
        if x is y:
            return False
        if x.rank < y.rank:
            x, y = y, x
        y.parent = x
        if x.rank == y.rank:
            x.rank += 1
        return True


prime_at_most = 1_000_000  # a max number
prime_flags = [True] * (prime_at_most + 1)
prime_flags[0] = prime_flags[1] = False
for i in range(2, int(prime_at_most**0.5) + 1):
    prime_flags[i * i : prime_at_most + 1 : i] = [False] * len(prime_flags[i * i : prime_at_most + 1 : i])
prime_flags = np.asarray(prime_flags, dtype=bool)
primes = [i for i in range(prime_at_most + 1) if prime_flags[i]]


cardinal_directions = np.asarray(((1, 0), (0, 1), (-1, 0), (0, -1)))
ordinal_directions = np.asarray(((1, 1), (1, -1), (-1, 1), (-1, -1)))
all_directions = np.concatenate((cardinal_directions, ordinal_directions))
