# Instructions for running standalone
# python -m pytil.tests.data_structures.test_array_heap

from functools import cache

import numba as nb
import numpy as np
from numba.typed import List

# Suppose your jitclass factory is in array_heap.py
from pytil.data_structures.array_heap import (
    get_array_heap_1d_items_jitclass_indirect as get_array_heap_1d_items_jitclass,
)


@nb.njit
def make_item_int64():
    """Return a 1D np.array[int64] with a random int in [-1000,1000)."""
    return np.array([np.random.randint(-1000, 1000)], dtype=np.int64)


@nb.njit
def make_item_float64():
    """Return a 1D np.array[float64] with a random float in [-1000,1000)."""
    return np.array([2000.0 * np.random.random() - 1000.0], dtype=np.float64)


@cache
def create_heap_tests(data_type):
    """
    A factory function (decorated with @cache) that:
      1) Creates a specialized heap jitclass based on `data_type`.
      2) Chooses the right random item generator and typed-list type.
      3) Defines two nopython-mode functions:
         - correctness_test(n_ops: int) -> bool
         - benchmark_test(n_ops: int) -> (int or float)
      4) Returns them as a tuple (correctness_test, benchmark_test).

    Usage:
        correctness_fn, benchmark_fn = create_heap_tests(nb.int64)
        ok = correctness_fn(2000)
        total = benchmark_fn(100_000)
        ...
    """

    # 1) Create the specialized jitclass
    HeapClass = get_array_heap_1d_items_jitclass(data_type)

    # 2) Decide which item-maker to use and what typed-list element type to use for the oracle
    if data_type == nb.int64:
        make_item = make_item_int64
        list_dtype = nb.int64
    elif data_type == nb.float64:
        make_item = make_item_float64
        list_dtype = nb.float64
    else:
        raise TypeError("Unsupported data_type; use nb.int64 or nb.float64")

    @nb.njit
    def correctness_test(n_ops: int) -> bool:
        """
        Performs random correctness testing by comparing each heap operation
        to a naive 'oracle' that stores items in ascending order (typed list).
        Returns True if all checks pass, otherwise False.
        """
        ds = HeapClass(n_ops + 10, 1)  # positional args
        oracle = List.empty_list(list_dtype)

        for _ in range(n_ops):
            op = np.random.randint(0, 3)
            if op == 0:
                # heappush
                if ds.size < ds.capacity:
                    val = make_item()
                    ds.heappush(val)
                    # Insert val[0] in ascending order in oracle
                    inserted = False
                    for i in range(len(oracle)):
                        if oracle[i] > val[0]:
                            oracle.insert(i, val[0])
                            inserted = True
                            break
                    if not inserted:
                        oracle.append(val[0])

            elif op == 1:
                # heappop
                if ds.size > 0:
                    got = ds.heappop()
                    if len(oracle) == 0:
                        return False
                    ref = oracle[0]
                    del oracle[0]
                    if got[0] != ref:
                        return False

            else:
                # heappeek
                if ds.size > 0:
                    got = ds.heappeek()
                    if len(oracle) == 0:
                        return False
                    if got[0] != oracle[0]:
                        return False

        # Final length check
        if ds.size != len(oracle):
            return False
        return True

    @nb.njit
    def benchmark_test(n_ops: int):
        """
        Performs random heappush, heappop, heappeek operations in nopython mode,
        measuring performance. Returns a sum (int or float) so the compiler
        cannot optimize the loop away.
        """
        ds = HeapClass(n_ops + 5, 1)
        total = 0

        for _ in range(n_ops):
            op = np.random.randint(0, 3)
            if op == 0:
                # heappush
                if ds.size < ds.capacity:
                    val = make_item()
                    ds.heappush(val)
            elif op == 1:
                # heappop
                if ds.size > 0:
                    got = ds.heappop()
                    total += got[0]
            else:
                # heappeek
                if ds.size > 0:
                    got = ds.heappeek()
                    total += got[0]

        return total

    # Return both jitted functions from the factory
    return correctness_test, benchmark_test


# ---------------------------------------------------------------------
# USAGE EXAMPLE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import time

    correctness_n_ops = 1000
    benchmark_n_ops = 1_000_000

    # Set seed for reproducibility
    np.random.seed(42)

    # 1) Get int64 test functions
    correctness_int64, benchmark_int64 = create_heap_tests(nb.int64)

    # 2) Run correctness for int64
    ok_int = correctness_int64(correctness_n_ops)
    print("Heap correctness int64:", ok_int)

    # 3) Run benchmark for int64
    t0 = time.time()
    sum_int = benchmark_int64(benchmark_n_ops)
    t1 = time.time()
    print(f"Heap benchmark int64 sum={sum_int}, time={t1 - t0:.4f}s")

    # 4) Similarly for float64
    correctness_float64, benchmark_float64 = create_heap_tests(nb.float64)

    ok_float = correctness_float64(correctness_n_ops)
    print("Heap correctness float64:", ok_float)

    t0 = time.time()
    sum_float = benchmark_float64(benchmark_n_ops)
    t1 = time.time()
    print(f"Heap benchmark float64 sum={sum_float}, time={t1 - t0:.4f}s")
