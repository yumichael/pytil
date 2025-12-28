# Instructions for running standalone
# python -m pytil.tests.data_structures.test_nearest_neighbor_index

import time
from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numpy.typing import NDArray

# Updated imports to reflect "index" naming
from pytil.data_structures.nearest_neighbor_index import (
    get_nearest_neighbor_index_jitclass,
)

# ==============================================================================
# 2. Testing & Benchmarking Module
# ==============================================================================


@cache
def get_nn_index_tests(count_type, coord_type, label_type):
    """
    Factory that creates two specialized Numba-jitted functions:
    1. correctness_test(rng, n_ops, weights, target_size)
    2. benchmark_test(rng, n_ops, weights, target_size)
    """
    NNIndexClass = get_nearest_neighbor_index_jitclass(count_type, coord_type, label_type)

    # Constants
    DIM = 3
    MAX_CAPACITY = 2**24
    MAX_LABEL_SIZE = 2**24
    MAX_QUERY_BUFFER_SIZE = 256

    # Operation Enums
    OP_INSERT_NEW = 0
    OP_UPDATE_EXIST = 1
    OP_QUERY_NEAREST = 2
    OP_QUERY_ALL = 3
    OP_DEL_EXIST = 4
    OP_DEL_MISSING = 5

    @njit
    def make_random_point(rng):
        res = np.empty(DIM, dtype=coord_type)
        for i in range(DIM):
            res[i] = rng.random()
        return res

    @njit
    def make_random_label(rng):
        return label_type(rng.integers(0, MAX_LABEL_SIZE))

    @njit
    def pick_operation(rng, weights, current_size, target_size):
        weights_length = len(weights)
        w = np.empty(weights_length, dtype=np.float64)
        for i in range(weights_length):
            w[i] = weights[i]

        if current_size < target_size:
            w[OP_DEL_EXIST] = 0.0
            w[OP_DEL_MISSING] = 0.0

        if current_size == 0:
            w[OP_UPDATE_EXIST] = 0.0
            w[OP_DEL_EXIST] = 0.0
            w[OP_QUERY_NEAREST] = 0.0
            w[OP_QUERY_ALL] = 0.0

        total = np.sum(w)
        if total <= 1e-12:
            assert False

        r = rng.random() * total
        cumulative = 0.0
        for i in range(weights_length):
            cumulative += w[i]
            if r < cumulative:
                return i
        return weights_length - 1

    @njit
    def oracle_closest_points(ref_point, oracle_points, oracle_active_labels, oracle_num_active, labels_buffer):
        min_dist_sq = np.inf
        buffer_count = 0
        max_buffer_len = len(labels_buffer)

        for i in range(oracle_num_active):
            l = oracle_active_labels[i]
            p = oracle_points[l]

            dist_sq = 0.0
            for d in range(DIM):
                diff = p[d] - ref_point[d]
                dist_sq += diff * diff

            if dist_sq < min_dist_sq - 1e-12:
                min_dist_sq = dist_sq
                buffer_count = 0
                labels_buffer[buffer_count] = l
                buffer_count += 1
            elif np.abs(dist_sq - min_dist_sq) <= 1e-12:
                assert buffer_count < max_buffer_len
                labels_buffer[buffer_count] = l
                buffer_count += 1
        return buffer_count, min_dist_sq

    @njit
    def correctness_test(rng, n_ops: int, weights: NDArray, target_size: int, record: NDArray) -> tuple[bool, str]:
        nn_index = NNIndexClass(MAX_CAPACITY, DIM, MAX_LABEL_SIZE)

        # Oracle Structures
        oracle_points = np.zeros((MAX_LABEL_SIZE, DIM), dtype=coord_type)
        oracle_active_labels = np.zeros(MAX_CAPACITY, dtype=label_type)
        oracle_num_active = 0
        oracle_label_to_idx = np.full(MAX_LABEL_SIZE, -1, dtype=np.int64)

        query_buffer_size = MAX_QUERY_BUFFER_SIZE
        nn_index_labels_buffer = np.zeros(query_buffer_size, dtype=label_type)
        oracle_labels_buffer = np.zeros(query_buffer_size, dtype=label_type)

        for op_num in range(n_ops):
            op = pick_operation(rng, weights, oracle_num_active, target_size)
            record[op] += 1

            if op == OP_INSERT_NEW:
                label = make_random_label(rng)
                retries = 0
                while oracle_label_to_idx[label] != -1 and retries < 10:
                    label = make_random_label(rng)
                    retries += 1
                if oracle_label_to_idx[label] != -1:
                    continue

                point = make_random_point(rng)
                nn_index[label] = point
                idx = oracle_num_active
                oracle_active_labels[idx] = label
                oracle_label_to_idx[label] = idx
                oracle_points[label] = point
                oracle_num_active += 1

            elif op == OP_UPDATE_EXIST:
                rand_idx = rng.integers(0, oracle_num_active)
                label = oracle_active_labels[rand_idx]
                point = make_random_point(rng)
                nn_index[label] = point
                oracle_points[label] = point

            elif op == OP_QUERY_NEAREST:
                ref_point = make_random_point(rng)
                _, o_min_dist = oracle_closest_points(
                    ref_point, oracle_points, oracle_active_labels, oracle_num_active, oracle_labels_buffer
                )

                res_point, res_label = nn_index.nearest(ref_point)

                dist_sq = 0.0
                for d in range(DIM):
                    diff = res_point[d] - ref_point[d]
                    dist_sq += diff * diff

                if np.abs(dist_sq - o_min_dist) > 1e-12:
                    return (
                        False,
                        f"QUERY_NEAREST distance mismatch at op {op_num}: nn_index={dist_sq}, oracle={o_min_dist}",
                    )

            elif op == OP_QUERY_ALL:
                ref_point = make_random_point(rng)
                o_count, _ = oracle_closest_points(
                    ref_point, oracle_points, oracle_active_labels, oracle_num_active, oracle_labels_buffer
                )
                # try:
                m_count = nn_index.nearest_ties_labels_assign(ref_point, nn_index_labels_buffer)
                # except Exception as e:
                #     return False, f"QUERY_ALL exception at op {op_num}: {e}"

                if m_count != o_count:
                    return False, f"QUERY_ALL count mismatch at op {op_num}: nn_index={m_count}, oracle={o_count}"

                if m_count > 0:
                    m_labels_sorted = np.sort(nn_index_labels_buffer[:m_count])
                    o_labels_sorted = np.sort(oracle_labels_buffer[:o_count])
                    if not np.all(m_labels_sorted == o_labels_sorted):
                        return False, f"QUERY_ALL labels mismatch at op {op_num}: nn_index labels != oracle labels"

            elif op == OP_DEL_EXIST:
                rand_idx = rng.integers(0, oracle_num_active)
                label = oracle_active_labels[rand_idx]
                # try:
                nn_index.remove(label)
                # except Exception as e:
                #     return False, f"DEL_EXIST exception at op {op_num} for label {label}: {e}"

                last_idx = oracle_num_active - 1
                last_label = oracle_active_labels[last_idx]
                oracle_active_labels[rand_idx] = last_label
                oracle_label_to_idx[last_label] = rand_idx
                oracle_label_to_idx[label] = -1
                oracle_num_active -= 1

            elif op == OP_DEL_MISSING:
                label = make_random_label(rng)
                while oracle_label_to_idx[label] != -1:
                    label = make_random_label(rng)
                try:
                    nn_index.remove(label)
                    return False, f"DEL_MISSING should have raised exception at op {op_num} for label {label}"
                except:
                    pass

            else:
                return False, f"Unknown operation {op} at op {op_num}"

            if len(nn_index) != oracle_num_active:
                return (
                    False,
                    f"Size mismatch at op {op_num}: nn_index size={len(nn_index)}, oracle size={oracle_num_active}",
                )

        return True, "All tests passed"

    @njit
    def benchmark_test(rng, n_ops: int, weights: NDArray, target_size: int, record: NDArray):
        nn_index = NNIndexClass(MAX_CAPACITY, DIM, MAX_LABEL_SIZE)
        labels_buffer = np.zeros(MAX_QUERY_BUFFER_SIZE, dtype=label_type)
        bench_active_labels = np.zeros(MAX_CAPACITY, dtype=label_type)
        bench_num_active = 0
        bench_label_to_idx = np.full(MAX_LABEL_SIZE, -1, dtype=np.int64)

        for _ in range(n_ops):
            op = pick_operation(rng, weights, bench_num_active, target_size)
            record[op] += 1

            if op == OP_INSERT_NEW:
                label = make_random_label(rng)
                if bench_label_to_idx[label] == -1:
                    point = make_random_point(rng)
                    nn_index[label] = (point[0], point[1], point[2])
                    idx = bench_num_active
                    bench_active_labels[idx] = label
                    bench_label_to_idx[label] = idx
                    bench_num_active += 1

            elif op == OP_UPDATE_EXIST:
                if bench_num_active > 0:
                    rand_idx = rng.integers(0, bench_num_active)
                    label = bench_active_labels[rand_idx]
                    point = make_random_point(rng)
                    nn_index[label] = (point[0], point[1], point[2])

            elif op == OP_QUERY_NEAREST:
                if bench_num_active > 0:
                    ref_point = make_random_point(rng)
                    res_point, res_label = nn_index.nearest(ref_point)

            elif op == OP_QUERY_ALL:
                if bench_num_active > 0:
                    ref_point = make_random_point(rng)
                    count = nn_index.nearest_ties_labels_assign(ref_point, labels_buffer)

            elif op == OP_DEL_EXIST:
                if bench_num_active > 0:
                    rand_idx = rng.integers(0, bench_num_active)
                    label = bench_active_labels[rand_idx]
                    nn_index.remove(label)
                    last_idx = bench_num_active - 1
                    last_label = bench_active_labels[last_idx]
                    bench_active_labels[rand_idx] = last_label
                    bench_label_to_idx[last_label] = rand_idx
                    bench_label_to_idx[label] = -1
                    bench_num_active -= 1

            elif op == OP_DEL_MISSING:
                label = make_random_label(rng)
                if bench_label_to_idx[label] == -1:
                    try:
                        nn_index.remove(label)
                    except:
                        pass

            else:
                assert False, 'Unknown operation'

    return correctness_test, benchmark_test


if __name__ == "__main__":
    count_type, coord_type, label_type = np.int32, np.float64, np.int32

    rng = np.random.default_rng(42)

    correctness_func, benchmark_func = get_nn_index_tests(count_type, coord_type, label_type)

    # Weights: [InsertNew, UpdateExist, QueryNearest, QueryAll, DelExist, DelMissing]

    print(f"Running Correctness Test...")
    w_correctness = np.array([0.2, 0.2, 0.1, 0.1, 0.2, 0.2], dtype=np.float64)
    record = np.zeros(len(w_correctness), dtype=np.int64)
    correctness_func(rng, 100, w_correctness, 50, record)  # Warmup

    record = np.zeros(len(w_correctness), dtype=np.int64)
    ok, message = correctness_func(rng, 10_000, w_correctness, 500, record)
    print("Operation counts:", record)
    if ok:
        print(f"✅ Correctness Test PASSED: {message}")
    else:
        print(f"❌ Correctness Test FAILED: {message}")
        exit(1)

    print(f"\nRunning Benchmark Test...")
    w_benchmark = np.array([0.25, 0.25, 0.25, 0.0, 0.25, 0.0], dtype=np.float64)
    record = np.zeros(len(w_correctness), dtype=np.int64)
    benchmark_func(rng, 100, w_benchmark, 50, record)  # Warmup

    t0 = time.time()
    record = np.zeros(len(w_correctness), dtype=np.int64)
    benchmark_func(rng, 2**22, w_benchmark, 4096 * 4, record)
    print("Operation counts:", record)
    print(f"Benchmark finished in {time.time()-t0:.4f}s")
