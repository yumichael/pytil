from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass


@cache
def get_disjoint_set_union_special_jitclass():
    spec = (
        ('n', nb.int64),  # total number of elements
        ('parent', nb.int64[:]),  # parent pointer for each element
        ('component_count', nb.int64),  # number of disjoint sets
    )

    @jitclass(spec)
    class DisjointSetUnionSpecial:
        def __init__(self, n):
            self.n = n
            self.parent = np.empty(n, dtype=np.int64)
            for i in range(n):
                self.parent[i] = i  # Each element starts as its own parent.
            self.component_count = n

        def find(self, x):
            """
            Find the representative (root) of the set containing 'x'
            using path halving for path compression.
            """
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union_by_first(self, x, y):
            """
            Merge the sets containing 'x' and 'y'.
            Returns True if a merge happened, or False if already in the same set.
            Always puts y under x.
            """
            root_x = self.find(x)
            root_y = self.find(y)
            if root_x == root_y:
                return False
            self.parent[root_y] = root_x
            self.component_count -= 1
            return True

        def connected(self, x, y):
            """
            Return True if 'x' and 'y' are in the same set.
            """
            return self.find(x) == self.find(y)

        def __str__(self):
            """
            Return a string representation of the DSU's parent arrays.
            """
            s_parent = '['
            for i in range(self.n):
                s_parent += str(self.parent[i])
                if i < self.n - 1:
                    s_parent += ', '
            s_parent += ']'

            return "DisjointSetUnionSpecial(parent: " + s_parent + ", components: " + str(self.component_count) + ")"

    return DisjointSetUnionSpecial
