import numpy as np
from numba import njit

tau = np.pi * 2

null_njit = njit(lambda *args: None)


@njit(inline='always')
def is_power_of_two(n):
    return (n & (n - 1)) == 0 and n > 0


@njit(inline='always')
def highest_power_of_2_dividing(n):
    return n & (~(n - 1))


@njit(inline='always')
def identity(x):
    return x


@njit(inline='always')
def prod_njit(a):
    assert len(a) >= 1
    out = a[0]
    for i in range(1, len(a)):
        out *= a[i]
    return out


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


def get_primes_up_to(prime_at_most):
    prime_flags = [True] * (prime_at_most + 1)
    prime_flags[0] = prime_flags[1] = False
    for i in range(2, int(prime_at_most**0.5) + 1):
        prime_flags[i * i : prime_at_most + 1 : i] = [False] * len(prime_flags[i * i : prime_at_most + 1 : i])
    prime_flags = np.asarray(prime_flags, dtype=bool)
    primes = [i for i in range(prime_at_most + 1) if prime_flags[i]]
    return primes


cardinal_directions = np.asarray(((1, 0), (0, 1), (-1, 0), (0, -1)))
ordinal_directions = np.asarray(((1, 1), (1, -1), (-1, 1), (-1, -1)))
all_directions = np.concatenate((cardinal_directions, ordinal_directions))


@njit
def invert_permutation(permutation):
    inverse = np.empty_like(permutation)
    for i in range(len(permutation)):
        inverse[permutation[i]] = i
    return inverse


def get_human_readable_time_delta(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f'{int(minutes)}m{seconds:.1f}s'
