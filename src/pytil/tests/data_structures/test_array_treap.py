# Instructions for running standalone
# python -m pytil.tests.data_structures.test_array_treap

from functools import cache

import numba as nb
import numpy as np
from numba.typed import List

# Import your treap jitclass factory.
from pytil.data_structures.array_treap import (
    create_array_treap_1d_items_jitclass,
    create_array_treap_1d_items_jitclass_fast,
)


# ---------------------------------------------------------------------
# Helpers: Create a single-element random item for each data type.
# ---------------------------------------------------------------------
@nb.njit
def make_item_int64():
    return np.array([np.random.randint(-1000, 1000)], dtype=np.int64)


@nb.njit
def make_item_float64():
    return np.array([2000.0 * np.random.random() - 1000.0], dtype=np.float64)


# ---------------------------------------------------------------------
# Helpers: Random value generators (compiled so they can be inlined).
# ---------------------------------------------------------------------
@nb.njit
def random_val_int():
    return np.random.randint(-1000, 1000)


@nb.njit
def random_val_float():
    return 2000.0 * np.random.random() - 1000.0


# ---------------------------------------------------------------------
# Factory Function for Treap Tests (only public methods)
# ---------------------------------------------------------------------
@cache
def create_treap_tests(
    data_type,
    create_array_treap_1d_items_jitclass_function=create_array_treap_1d_items_jitclass_fast,
):
    """
    Factory that creates two specialized Numba-jitted functions for testing
    your treap implementation (from array_treap.py). Only public methods
    are tested (insert, __getitem__, slice, remove, and delitem). The inner
    functions do not compare data_type values.

    Returns a tuple:
       (correctness_test, benchmark_test)
    """
    # Create the specialized jitclass for the treap.
    TreapClass = create_array_treap_1d_items_jitclass_function(data_type)

    # Choose the proper item generator, oracle element type, random value
    # generator, and initial total value—all done outside the njit functions.
    if data_type == nb.int64:
        item_generator = make_item_int64
        list_dtype = nb.int64
        rand_val = random_val_int
        initial_total = 0
    elif data_type == nb.float64:
        item_generator = make_item_float64
        list_dtype = nb.float64
        rand_val = random_val_float
        initial_total = 0.0
    else:
        raise TypeError("Unsupported data_type; use nb.int64 or nb.float64")

    @nb.njit
    def correctness_test(n_ops: int) -> bool:
        """
        Executes a randomized sequence of operations on the treap,
        comparing its behavior against an oracle (a sorted typed list).
        Operations include:
          0: insert(item)
          1: remove(item)
          2: __getitem__(index)
          3: slice(start, stop)
          4: delitem(index)
        Returns True if all checks pass.
        """
        treap = TreapClass(n_ops + 10, 1)  # Use positional arguments.
        oracle = List.empty_list(list_dtype)

        for _ in range(n_ops):
            op = np.random.randint(0, 5)
            if op == 0:
                # INSERT operation.
                item = item_generator()
                treap.insert(item)
                # Insert item[0] into oracle in ascending order.
                inserted = False
                for i in range(len(oracle)):
                    if oracle[i] > item[0]:
                        oracle.insert(i, item[0])
                        inserted = True
                        break
                if not inserted:
                    oracle.append(item[0])

            elif op == 1:
                # REMOVE operation: generate a candidate value.
                val = rand_val()
                arr = np.array([val], dtype=data_type)
                treap.remove(arr)
                # Remove first occurrence from oracle if present.
                for i in range(len(oracle)):
                    if oracle[i] == val:
                        del oracle[i]
                        break

            elif op == 2:
                # __getitem__ test.
                if treap.size > 0:
                    idx = np.random.randint(0, treap.size)
                    item_from_treap = treap[idx]
                    if item_from_treap[0] != oracle[idx]:
                        return False

            elif op == 3:
                # slice test.
                if treap.size > 0:
                    start = np.random.randint(-treap.size, treap.size)
                    stop = np.random.randint(-treap.size, treap.size + 1)
                    sliced = treap.slice(start, stop)
                    # Emulate Python slicing on oracle.
                    n = len(oracle)
                    s = start if start >= 0 else start + n
                    e = stop if stop >= 0 else stop + n
                    if s < 0:
                        s = 0
                    if s > n:
                        s = n
                    if e < s:
                        e = s
                    if e > n:
                        e = n
                    expected_len = e - s
                    if sliced.shape[0] != expected_len:
                        return False
                    for j in range(sliced.shape[0]):
                        if sliced[j, 0] != oracle[s + j]:
                            return False

            elif op == 4:
                # delitem test.
                if treap.size > 0:
                    idx = np.random.randint(0, treap.size)
                    treap.delitem(idx)
                    del oracle[idx]

            if treap.size != len(oracle):
                return False
        return True

    @nb.njit
    def benchmark_test(n_ops: int):
        """
        Executes a randomized sequence of operations on the treap for benchmarking.
        The operations are the same as in correctness_test, but without an oracle.
        A running total is accumulated (from __getitem__ and slice operations) so that
        the work cannot be optimized away.
        Returns the accumulated total.
        """
        treap = TreapClass(n_ops + 10, 1)
        total = initial_total
        for _ in range(n_ops):
            op = np.random.randint(0, 5)
            if op == 0:
                if treap.size < (n_ops + 10):
                    treap.insert(item_generator())
            elif op == 1:
                # Remove: generate a random value.
                val = rand_val()
                arr = np.array([val], dtype=data_type)
                treap.remove(arr)
            elif op == 2:
                if treap.size > 0:
                    idx = np.random.randint(0, treap.size)
                    total += treap[idx][0]
            elif op == 3:
                if treap.size > 0:
                    start = np.random.randint(-treap.size, treap.size)
                    stop = np.random.randint(-treap.size, treap.size + 1)
                    sliced = treap.slice(start, stop)
                    if sliced.shape[0] > 0:
                        total += sliced[0, 0]
            elif op == 4:
                if treap.size > 0:
                    idx = np.random.randint(0, treap.size)
                    treap.delitem(idx)
            total += treap.size  # Prevent dead-code elimination.
        return total

    return correctness_test, benchmark_test


# ---------------------------------------------------------------------
# USAGE EXAMPLE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import time

    correctness_n_ops = 1000
    benchmark_n_ops = 200_000

    np.random.seed(42)

    # Generate tests for int64.
    treap_correctness_int64, treap_benchmark_int64 = create_treap_tests(nb.int64)
    ok_int = treap_correctness_int64(correctness_n_ops)
    print("Treap correctness (int64):", ok_int)

    t0 = time.time()
    total_int = treap_benchmark_int64(benchmark_n_ops)
    t1 = time.time()
    print(f"Treap benchmark (int64): total={total_int}, time={t1 - t0:.4f}s")

    # Generate tests for float64.
    treap_correctness_float64, treap_benchmark_float64 = create_treap_tests(nb.float64)
    ok_float = treap_correctness_float64(correctness_n_ops)
    print("Treap correctness (float64):", ok_float)

    t0 = time.time()
    total_float = treap_benchmark_float64(benchmark_n_ops)
    t1 = time.time()
    print(f"Treap benchmark (float64): total={total_float}, time={t1 - t0:.4f}s")
