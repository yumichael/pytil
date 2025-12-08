# Instructions for running standalone
# python -m pytil.tests.data_structures.test_nearest_neighbor_map

import time
from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numpy.typing import NDArray

from pytil.data_structures.nearest_neighbor_map import get_nearest_neighbor_map_jitclass
from pytil.numba_utility import set_seed

# ==============================================================================
# 2. Testing & Benchmarking Module
# ==============================================================================

# NOTE Numba does not allow catching specific exceptions in njit functions.
# So we use generic except: blocks and add comments about the expected exceptions.


@cache
def create_nn_map_tests(coord_type, key_type, count_type):
    """
    Factory that creates two specialized Numba-jitted functions:
    1. correctness_test(n_ops)
    2. benchmark_test(n_ops)
    """
    # The factory no longer takes dimension_count
    NNMapClass = get_nearest_neighbor_map_jitclass(coord_type, key_type, count_type)

    # Constants for the test
    DIM = 3
    MAX_CAPACITY = 2**24
    MAX_KEY_SIZE = 2**24

    # Helper: Generate random point
    @njit
    def make_random_point():
        res = np.empty(DIM, dtype=coord_type)
        for i in range(DIM):
            res[i] = np.random.rand() * 100.0  # 0 to 100
        return res

    # Helper: Generate random key
    @njit
    def make_random_key():
        return key_type(np.random.randint(0, MAX_KEY_SIZE))

    # Helper: Check if points are identical (within float tolerance)
    @njit
    def points_equal(p1, p2):
        if p1.shape != p2.shape:
            return False
        return np.allclose(p1, p2, atol=1e-9)

    # Helper: Linear scan closest neighbor search (the Oracle)
    # Optimized to iterate only over active keys
    @njit
    def oracle_closest_points(ref_point, oracle_points, oracle_active_keys, oracle_num_active, keys_buffer):
        min_dist_sq = np.inf
        buffer_count = 0
        max_buffer_len = len(keys_buffer)

        # Optimization: Iterate only over active keys, not the entire key space
        for i in range(oracle_num_active):
            k = oracle_active_keys[i]
            p = oracle_points[k]

            dist_sq = 0.0
            for d in range(DIM):
                diff = p[d] - ref_point[d]
                dist_sq += diff * diff

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                buffer_count = 0
                keys_buffer[buffer_count] = k
                buffer_count += 1
            elif np.abs(dist_sq - min_dist_sq) <= 1e-12:
                if buffer_count < max_buffer_len:
                    keys_buffer[buffer_count] = k
                    buffer_count += 1
                # Note: We don't raise IndexError in the oracle, just skip adding.
        return buffer_count, min_dist_sq

    @njit
    def correctness_test(n_ops: int) -> bool:
        """
        Executes a randomized sequence of operations, comparing results against an Oracle (linear scan).
        Returns True if all checks pass.
        """
        # Init Data Structure
        # Note: DIM is passed as the dimension_count parameter.
        nn_map = NNMapClass(MAX_CAPACITY, MAX_KEY_SIZE, DIM)

        # Init Oracle Optimized Structures
        # 1. oracle_points: Stores coordinates for key 'k' at index 'k'
        oracle_points = np.zeros((MAX_KEY_SIZE, DIM), dtype=coord_type)

        # 2. Dense tracking of active keys
        oracle_active_keys = np.zeros(MAX_CAPACITY, dtype=key_type)
        oracle_num_active = 0

        # 3. Map from key -> index in oracle_active_keys (for O(1) deletion)
        #    Value -1 implies key is not present.
        oracle_map_key_to_idx = np.full(MAX_KEY_SIZE, -1, dtype=np.int64)

        # Buffer for closest keys
        query_buffer_size = 5
        nn_map_keys_buffer = np.zeros(query_buffer_size, dtype=key_type)
        oracle_keys_buffer = np.zeros(query_buffer_size, dtype=key_type)

        for i in range(n_ops):
            op = np.random.randint(0, 3)

            # --- OP 0: INSERT/UPDATE ---
            if op == 0:
                key = make_random_key()
                point = make_random_point()

                key_idx = oracle_map_key_to_idx[key]
                key_exists = key_idx != -1

                # Update Map
                nn_map[key] = point  # Handles tuple assignment due to earlier fix

                # Update Oracle
                if not key_exists:
                    # Add new key to the end of active list
                    new_idx = oracle_num_active
                    oracle_active_keys[new_idx] = key
                    oracle_map_key_to_idx[key] = new_idx
                    oracle_num_active += 1

                # Assignment optimization: Slice assignment
                oracle_points[key] = point

            # --- OP 1: REMOVE (Deletion) ---
            elif op == 1:
                # Optimization: Only delete existing keys if possible
                if oracle_num_active == 0:
                    continue

                # Pick a random active key
                rand_idx = np.random.randint(0, oracle_num_active)
                key = oracle_active_keys[rand_idx]

                # Check existence before operation
                key_idx = oracle_map_key_to_idx[key]
                key_exists = key_idx != -1

                # Update Map
                try:
                    nn_map.remove(key)
                except:  # KeyError:
                    # If map raises KeyError, key must not have existed.
                    if key_exists:
                        print(f"Delete failed unexpectedly (KeyError raised for existing key {key}).")
                        return False
                    continue  # Correct behavior, key was already gone.

                # If removal succeeded, the key MUST have existed.
                if not key_exists:
                    print(f"Delete succeeded unexpectedly for non-existent key {key}.")
                    return False

                # Update Oracle (If successful)
                # Swap-remove logic to keep active array dense
                last_idx = oracle_num_active - 1
                last_key = oracle_active_keys[last_idx]

                # Move last key to the spot of the deleted key
                oracle_active_keys[key_idx] = last_key
                oracle_map_key_to_idx[last_key] = key_idx

                # Clear the deleted spot and mapping
                oracle_map_key_to_idx[key] = -1
                oracle_num_active -= 1

            # --- OP 2: QUERY ---
            elif op == 2:
                if oracle_num_active == 0:
                    continue

                ref_point = make_random_point()

                # Get results from Oracle
                o_count, o_min_dist_sq = oracle_closest_points(
                    ref_point, oracle_points, oracle_active_keys, oracle_num_active, oracle_keys_buffer
                )

                # Get results from KD-Tree
                try:
                    m_count = nn_map.get_closest_points_assign(ref_point, nn_map_keys_buffer)
                except:  # IndexError:
                    # The KD-Tree is designed to raise IndexError if buffer is too small.
                    # We must ensure the Oracle also found too many ties for the buffer size.
                    if o_count > query_buffer_size:
                        continue  # Expected behavior, buffer limit reached in both
                    print("Map raised IndexError unexpectedly.")
                    return False

                # Compare counts
                if m_count != o_count:
                    print(f"Query count mismatch: Map found {m_count}, Oracle found {o_count}.")
                    return False

                # Compare contents
                if m_count > 0:
                    # Sort both buffers to compare them
                    m_keys_sorted = np.sort(nn_map_keys_buffer[:m_count])
                    o_keys_sorted = np.sort(oracle_keys_buffer[:o_count])

                    if not np.all(m_keys_sorted == o_keys_sorted):
                        print("Query key mismatch: Closest keys found do not match Oracle keys.")
                        return False

            # Final consistency check
            if len(nn_map) != oracle_num_active:
                print(f"Size mismatch: Map size {len(nn_map)}, Oracle size {oracle_num_active}.")
                return False

        return True

    @njit
    def benchmark_test(n_ops: int):
        """
        Executes a randomized sequence of operations on the treap for benchmarking.
        A running total is accumulated so that the work cannot be optimized away.
        Returns the accumulated total.
        """
        # Init Data Structure
        nn_map = NNMapClass(MAX_CAPACITY, MAX_KEY_SIZE, DIM)
        total = 0.0

        # Buffer for closest keys (must be initialized for njit)
        keys_buffer = np.zeros(5, dtype=key_type)

        for _ in range(n_ops):
            op = np.random.randint(0, 3)

            # --- OP 0: REMOVE (Deletion) ---
            if op == 0:
                key = make_random_key()
                try:
                    nn_map.remove(key)
                except:  # KeyError:
                    # Key not found, expected behavior in a benchmark; continue.
                    pass

            # --- OP 1: INSERT/UPDATE ---
            elif op == 1:
                key = make_random_key()
                point = make_random_point()
                # Use tuple assignment to test the flexibility
                nn_map[key] = (point[0], point[1], point[2])

            # --- OP 2: QUERY ---
            elif op == 2:
                if len(nn_map) > 0:
                    ref_point = make_random_point()
                    try:
                        count = nn_map.get_closest_points_assign(ref_point, keys_buffer)
                        total += count
                    except:  # IndexError:
                        # Buffer too small, still counts as work done.
                        pass

            total += len(nn_map)  # Prevent dead-code elimination.
        return total

    return correctness_test, benchmark_test


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================
if __name__ == "__main__":
    # Test Configuration
    coord_type = np.float64  # High precision for strict distance checks
    key_type = np.int64
    count_type = np.int64

    # Set random seed for reproducibility
    set_seed(42)

    print("Generating tests and compiling JIT functions...")
    # The factory call is correct without DIM argument
    correctness_func, benchmark_func = create_nn_map_tests(coord_type, key_type, count_type)

    # 1. Correctness Test
    # Using fewer ops for correctness because the Oracle (Linear Scan) is slow O(N)
    n_correctness_ops = 10_000
    print(f"Running Correctness Test ({n_correctness_ops} ops)...")

    # Run once to compile
    t0 = time.time()
    # Note: Using a small number of ops for compilation speed
    result = correctness_func(100)
    print(f"  (Compilation took {time.time()-t0:.4f}s)")

    # Run actual test
    t0 = time.time()
    result = correctness_func(n_correctness_ops)
    dt = time.time() - t0

    if result:
        print(f"✅ Correctness Test PASSED in {dt:.4f}s")
    else:
        print(f"❌ Correctness Test FAILED in {dt:.4f}s")
        exit(1)

    # 2. Benchmark Test
    # The map structure is fast, so we can run many more ops
    n_benchmark_ops = 2**24
    print(f"\nRunning Benchmark Test ({n_benchmark_ops} ops)...")

    # Run once to compile
    t0 = time.time()
    total_result = benchmark_func(100)
    print(f"  (Compilation took {time.time()-t0:.4f}s)")

    # Run actual benchmark
    t0 = time.time()
    total_result = benchmark_func(n_benchmark_ops)
    t1 = time.time()
    dt = t1 - t0

    print(f"Benchmark Test (Total: {total_result}): {dt:.4f}s")
