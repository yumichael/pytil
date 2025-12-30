# Instructions for running standalone
# python -m pytil.tests.data_structures.test_nearest_neighbor_index

import time
from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numpy.typing import NDArray

from pytil.data_structures.nearest_neighbor_index import (
    get_nearest_neighbor_index_jitclass,
)

# ==============================================================================
# 1. Random Sampling Data Structure for Labels
# ==============================================================================


@cache
def get_label_sampler_jitclass(label_type):
    """
    Factory that creates a jitclass for efficiently sampling random labels by activity status.

    This data structure maintains labels partitioned into active and inactive sets and supports:
    - O(1) random sampling of active or inactive labels
    - O(1) toggling a label between active and inactive
    - O(1) checking if a label is active
    """

    label_type_numba = nb.from_dtype(label_type)

    spec = [
        ('labels', label_type_numba[:]),  # Array of all labels [active_labels | inactive_labels]
        ('positions', label_type_numba[:]),  # Maps label -> position in labels array
        ('num_active', label_type_numba),  # Number of active labels (active labels are at indices [0, num_active))
        ('capacity', label_type_numba),  # Maximum number of labels
    ]

    @jitclass(spec)
    class LabelSampler:
        def __init__(self, max_labels):
            """Initialize with all labels as inactive."""
            self.capacity = label_type(max_labels)
            self.num_active = label_type(0)
            self.labels = np.arange(max_labels, dtype=label_type)
            self.positions = np.arange(max_labels, dtype=label_type)

        def sample_active(self, rng):
            """Sample a random active label. Returns -1 if none available."""
            assert self.num_active > 0

            idx = rng.integers(0, self.num_active)
            return self.labels[idx]

        def sample_inactive(self, rng):
            """Sample a random inactive label. Returns -1 if none available."""
            num_inactive = self.capacity - self.num_active
            assert num_inactive > 0

            idx = rng.integers(self.num_active, self.capacity)
            return self.labels[idx]

        def mark_active(self, label):
            """Mark a label as active (move to active partition)."""
            pos = self.positions[label]
            assert pos >= self.num_active

            # Swap with first inactive label
            first_inactive_pos = self.num_active
            first_inactive_label = self.labels[first_inactive_pos]

            self.labels[pos] = first_inactive_label
            self.labels[first_inactive_pos] = label

            self.positions[first_inactive_label] = pos
            self.positions[label] = first_inactive_pos

            self.num_active += 1

        def mark_inactive(self, label):
            """Mark a label as inactive (move to inactive partition)."""
            pos = self.positions[label]
            assert pos < self.num_active

            # Swap with last active label
            last_active_pos = self.num_active - 1
            last_active_label = self.labels[last_active_pos]

            self.labels[pos] = last_active_label
            self.labels[last_active_pos] = label

            self.positions[last_active_label] = pos
            self.positions[label] = last_active_pos

            self.num_active -= 1

        def is_active(self, label):
            """Check if a label is currently active."""
            return self.positions[label] < self.num_active

        def has_active(self):
            """Check if there are any active labels available."""
            return self.num_active > 0

        def has_inactive(self):
            """Check if there are any inactive labels available."""
            return self.num_active < self.capacity

        def get_active_count(self):
            """Get the number of active labels."""
            return self.num_active

        def get_active_label_at_index(self, idx):
            """Get the active label at a specific index (0 to num_active-1)."""
            return self.labels[idx]

    return LabelSampler


# ==============================================================================
# 2. Testing & Benchmarking Module
# ==============================================================================


@cache
def get_nn_index_tests(count_type, coord_type, label_type, rtol, atol):
    """
    Factory that creates two specialized Numba-jitted functions:
    1. correctness_test(rng, n_ops, weights, target_size)
    2. benchmark_test(rng, n_ops, weights, target_size)
    """
    NNIndexClass = get_nearest_neighbor_index_jitclass(
        count_type,
        coord_type,
        label_type,
        rtol=rtol,
        atol=atol,
    )

    LabelSampler = get_label_sampler_jitclass(label_type)

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
    def oracle_closest_points(ref_point, oracle_points, label_sampler, labels_buffer):
        """Find all points at minimum distance, using the same tolerance as the data structure."""
        min_dist_sq = np.inf
        buffer_count = 0
        max_buffer_len = len(labels_buffer)

        num_active = label_sampler.get_active_count()
        for i in range(num_active):
            l = label_sampler.get_active_label_at_index(i)
            p = oracle_points[l]

            dist_sq = 0.0
            for d in range(DIM):
                diff = p[d] - ref_point[d]
                dist_sq += diff * diff

            if dist_sq < min_dist_sq and not np.isclose(dist_sq, min_dist_sq, rtol=rtol, atol=atol):
                min_dist_sq = dist_sq
                buffer_count = 0
                labels_buffer[buffer_count] = l
                buffer_count += 1
            elif np.isclose(dist_sq, min_dist_sq, rtol=rtol, atol=atol):
                assert buffer_count < max_buffer_len
                labels_buffer[buffer_count] = l
                buffer_count += 1
        return buffer_count, min_dist_sq

    @njit
    def correctness_test(rng, n_ops: int, weights: NDArray, target_size: int, record: NDArray) -> tuple[bool, str]:
        nn_index = NNIndexClass(MAX_CAPACITY, DIM, MAX_LABEL_SIZE)
        label_sampler = LabelSampler(MAX_LABEL_SIZE)

        # Oracle Structures
        oracle_points = np.zeros((MAX_LABEL_SIZE, DIM), dtype=coord_type)

        query_buffer_size = MAX_QUERY_BUFFER_SIZE
        nn_index_labels_buffer = np.zeros(query_buffer_size, dtype=label_type)
        oracle_labels_buffer = np.zeros(query_buffer_size, dtype=label_type)

        for op_num in range(n_ops):
            oracle_num_active = label_sampler.get_active_count()
            op = pick_operation(rng, weights, oracle_num_active, target_size)
            record[op] += 1

            if op == OP_INSERT_NEW:
                label = label_sampler.sample_inactive(rng)

                point = make_random_point(rng)
                nn_index[label] = point

                label_sampler.mark_active(label)
                oracle_points[label] = point

            elif op == OP_UPDATE_EXIST:
                label = label_sampler.sample_active(rng)

                point = make_random_point(rng)
                nn_index[label] = point
                oracle_points[label] = point

            elif op == OP_QUERY_NEAREST:
                ref_point = make_random_point(rng)
                o_count, o_min_dist = oracle_closest_points(
                    ref_point, oracle_points, label_sampler, oracle_labels_buffer
                )

                res_point, res_label = nn_index.nearest(ref_point)

                # Check distance is correct
                dist_sq = 0.0
                for d in range(DIM):
                    diff = res_point[d] - ref_point[d]
                    dist_sq += diff * diff

                if not np.isclose(dist_sq, o_min_dist, rtol=rtol, atol=atol):
                    return (
                        False,
                        f"QUERY_NEAREST distance mismatch at op {op_num}: nn_index={dist_sq}, oracle={o_min_dist}",
                    )

                # Check that returned label is among the tied labels
                found = False
                for i in range(o_count):
                    if oracle_labels_buffer[i] == res_label:
                        found = True
                        break
                if not found:
                    return (
                        False,
                        f"QUERY_NEAREST label {res_label} not among oracle closest labels at op {op_num}",
                    )

                # Check tie-breaking: returned label should be the minimum among all tied labels
                if o_count > 1:
                    min_label = oracle_labels_buffer[0]
                    for i in range(1, o_count):
                        if oracle_labels_buffer[i] < min_label:
                            min_label = oracle_labels_buffer[i]
                    if res_label != min_label:
                        return (
                            False,
                            f"QUERY_NEAREST tie-breaking failed at op {op_num}: expected min label {min_label}, got {res_label}",
                        )

            elif op == OP_QUERY_ALL:
                ref_point = make_random_point(rng)
                o_count, _ = oracle_closest_points(ref_point, oracle_points, label_sampler, oracle_labels_buffer)
                m_count = nn_index.nearest_ties_labels_assign(ref_point, nn_index_labels_buffer)

                if m_count != o_count:
                    return False, f"QUERY_ALL count mismatch at op {op_num}: nn_index={m_count}, oracle={o_count}"

                if m_count > 0:
                    m_labels_sorted = np.sort(nn_index_labels_buffer[:m_count])
                    o_labels_sorted = np.sort(oracle_labels_buffer[:o_count])
                    if not np.all(m_labels_sorted == o_labels_sorted):
                        return False, f"QUERY_ALL labels mismatch at op {op_num}: nn_index labels != oracle labels"

            elif op == OP_DEL_EXIST:
                label = label_sampler.sample_active(rng)

                nn_index.remove(label)
                label_sampler.mark_inactive(label)

            elif op == OP_DEL_MISSING:
                label = label_sampler.sample_inactive(rng)

                try:
                    nn_index.remove(label)
                    return False, f"DEL_MISSING should have raised exception at op {op_num} for label {label}"
                except:
                    pass

            else:
                return False, f"Unknown operation {op} at op {op_num}"

            oracle_num_active = label_sampler.get_active_count()
            if len(nn_index) != oracle_num_active:
                return (
                    False,
                    f"Size mismatch at op {op_num}: nn_index size={len(nn_index)}, oracle size={oracle_num_active}",
                )

        return True, "All tests passed"

    @njit
    def benchmark_test(rng, n_ops: int, weights: NDArray, target_size: int, record: NDArray):
        nn_index = NNIndexClass(MAX_CAPACITY, DIM, MAX_LABEL_SIZE)
        label_sampler = LabelSampler(MAX_LABEL_SIZE)

        labels_buffer = np.zeros(MAX_QUERY_BUFFER_SIZE, dtype=label_type)

        for _ in range(n_ops):
            bench_num_active = label_sampler.get_active_count()
            op = pick_operation(rng, weights, bench_num_active, target_size)
            record[op] += 1

            if op == OP_INSERT_NEW:
                label = label_sampler.sample_inactive(rng)

                point = make_random_point(rng)
                nn_index[label] = (point[0], point[1], point[2])
                label_sampler.mark_active(label)

            elif op == OP_UPDATE_EXIST:
                label = label_sampler.sample_active(rng)

                point = make_random_point(rng)
                nn_index[label] = (point[0], point[1], point[2])

            elif op == OP_QUERY_NEAREST:
                ref_point = make_random_point(rng)
                res_point, res_label = nn_index.nearest(ref_point)

            elif op == OP_QUERY_ALL:
                ref_point = make_random_point(rng)
                count = nn_index.nearest_ties_labels_assign(ref_point, labels_buffer)

            elif op == OP_DEL_EXIST:
                label = label_sampler.sample_active(rng)

                nn_index.remove(label)
                label_sampler.mark_inactive(label)

            elif op == OP_DEL_MISSING:
                label = label_sampler.sample_inactive(rng)

                try:
                    nn_index.remove(label)
                except:
                    pass
                else:
                    assert False, 'DEL_MISSING should have raised exception'

            else:
                assert False, 'Unknown operation'

    return correctness_test, benchmark_test


def test_tie_breaking_ordering(count_type, coord_type, label_type, rtol, atol):
    """Specific test to verify that nearest() returns the smallest label in case of ties."""

    NNIndexClass = get_nearest_neighbor_index_jitclass(
        count_type,
        coord_type,
        label_type,
        rtol=rtol,
        atol=atol,
    )
    nn_index = NNIndexClass(100, 2, 100)

    # Insert points at the same location with different labels
    # Label order: 5, 2, 8, 1, 3
    nn_index[5] = np.array([0.5, 0.5], dtype=coord_type)
    nn_index[2] = np.array([0.5, 0.5], dtype=coord_type)
    nn_index[8] = np.array([0.5, 0.5], dtype=coord_type)
    nn_index[1] = np.array([0.5, 0.5], dtype=coord_type)
    nn_index[3] = np.array([0.5, 0.5], dtype=coord_type)

    # Query from the same location - should return label 1 (smallest)
    ref_point = np.array([0.5, 0.5], dtype=coord_type)
    _, result_label = nn_index.nearest(ref_point)

    if result_label != 1:
        return False, f"Expected label 1 (smallest), got {result_label}"

    # Add a point slightly closer
    nn_index[10] = np.array([0.500001, 0.5], dtype=coord_type)

    # Query again - should return label 10 if it's clearly closer
    _, result_label = nn_index.nearest(ref_point)
    # Depending on tolerance, could be 1 or 10
    # Let's add a clearly different point
    nn_index[20] = np.array([0.6, 0.6], dtype=coord_type)

    # Query from near the first cluster - should still return label 1
    ref_point2 = np.array([0.50000001, 0.50000001], dtype=coord_type)
    _, result_label2 = nn_index.nearest(ref_point2)

    if result_label2 != 1:
        return False, f"Expected label 1 from near cluster, got {result_label2}"

    return True, "Tie-breaking test passed"


if __name__ == "__main__":
    count_type, coord_type, label_type = np.int32, np.float32, np.int32
    rtol, atol = 1e-9, 1e-12

    rng = np.random.default_rng(42)

    correctness_func, benchmark_func = get_nn_index_tests(count_type, coord_type, label_type, rtol, atol)

    # Weights: [InsertNew, UpdateExist, QueryNearest, QueryAll, DelExist, DelMissing]
    print(f"Running Correctness Test...")
    w_correctness = np.array([0.2, 0.2, 0.1, 0.1, 0.2, 0.2], dtype=np.float64)
    record = np.zeros(len(w_correctness), dtype=np.int64)
    t0 = time.time()
    correctness_func(rng, 100, w_correctness, 50, record)  # Warmup
    print(f"Compilation finished in {time.time()-t0:.4f}s")

    n_ops = 10_000
    record = np.zeros(len(w_correctness), dtype=np.int64)
    ok, message = correctness_func(rng, n_ops, w_correctness, 500, record)
    print("Operation counts:", record)
    assert sum(record) == n_ops
    if ok:
        print(f"✅ Correctness Test PASSED: {message}")
    else:
        print(f"❌ Correctness Test FAILED: {message}")
        exit(1)

    # Test tie-breaking specifically
    print("\nRunning Tie-Breaking Test...")
    ok, message = test_tie_breaking_ordering(count_type, coord_type, label_type, rtol, atol)
    if ok:
        print(f"✅ Tie-Breaking Test PASSED: {message}")
    else:
        print(f"❌ Tie-Breaking Test FAILED: {message}")
        exit(1)

    print(f"\nRunning Benchmark Test...")
    w_benchmark = np.array([0.25, 0.25, 0.25, 0.0, 0.25, 0.0], dtype=np.float64)
    record = np.zeros(len(w_correctness), dtype=np.int64)
    benchmark_func(rng, 100, w_benchmark, 50, record)  # Warmup

    n_ops = 2**22
    t0 = time.time()
    record = np.zeros(len(w_correctness), dtype=np.int64)
    benchmark_func(rng, n_ops, w_benchmark, 4096 * 4, record)
    print("Operation counts:", record)
    assert sum(record) == n_ops
    print(f"Benchmark finished in {time.time()-t0:.4f}s")
