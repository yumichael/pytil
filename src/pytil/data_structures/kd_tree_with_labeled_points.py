from collections.abc import Sequence
from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numpy.typing import NDArray


@njit
def _build_tree_recursive_njit(points, tree_data, indices, depth, dim, count_type, coordinate_type):
    """
    External njit helper function to recursively build a balanced KD-Tree.
    (Numba jitclass does not support recursive methods.)

    tree_data layout (per node):
    [0] = left child index
    [1] = right child index
    [2] = axis
    [3] = valid flag (as count_type: 0 or 1)
    """
    N = len(indices)
    if N == 0:
        return -1

    axis = depth % dim
    mid = N // 2

    # Sort indices based on points[idx, axis] to find median
    vals = np.empty(N, dtype=coordinate_type)
    for i in range(N):
        vals[i] = points[indices[i], axis]

    sorted_arg_indices = np.argsort(vals)
    sorted_indices = indices[sorted_arg_indices]

    node_idx = sorted_indices[mid]
    tree_data[node_idx, 2] = axis  # axis

    left_indices = sorted_indices[:mid]
    right_indices = sorted_indices[mid + 1 :]

    tree_data[node_idx, 0] = _build_tree_recursive_njit(
        points, tree_data, left_indices, depth + 1, dim, count_type, coordinate_type
    )
    tree_data[node_idx, 1] = _build_tree_recursive_njit(
        points, tree_data, right_indices, depth + 1, dim, count_type, coordinate_type
    )

    return node_idx


@cache
def get_kd_tree_with_labeled_points_jitclass(count_type, coordinate_type, label_type):
    """
    Factory to create a specialized JIT-compiled KD-Tree class.

    Args:
        count_type: Dtype for internal indexing (e.g., np.int32).
        coordinate_type: Dtype for point coordinates (e.g., np.float32 or np.float64).
        label_type: Dtype for external labels (e.g., np.int32).
    """

    # Rebuild constant: Trigger rebuild if next_free_idx / num_active > REBUILD_RATIO
    REBUILD_RATIO = 2.0

    coordinate_type_numba = nb.from_dtype(coordinate_type)
    label_type_numba = nb.from_dtype(label_type)
    count_type_numba = nb.from_dtype(count_type)

    # Tree data indices (for AoS structure)
    IDX_LEFT = 0
    IDX_RIGHT = 1
    IDX_AXIS = 2
    IDX_VALID = 3

    kd_tree_with_labeled_points_spec = [
        # --- Tree Storage (Array of Structures approach) ---
        ('points', coordinate_type_numba[:, :]),
        # (max_size,) - External labels
        ('tree_labels', label_type_numba[:]),
        # (max_size, 4) - Combined tree data: [left, right, axis, valid]
        ('tree_data', count_type_numba[:, :]),
        # --- Tree State ---
        ('root', count_type_numba),
        ('next_free_idx', count_type_numba),
        ('num_active', count_type_numba),
        ('dim', count_type_numba),
        ('capacity', count_type_numba),
        # --- Direct Array Mapping ---
        ('label_to_index_map', count_type_numba[:]),
        ('max_label_size', count_type_numba),
        # --- Pre-allocated Query Stack (adaptive size) ---
        ('_query_stack', count_type_numba[:]),
        ('_max_query_depth', count_type_numba),
    ]

    @jitclass(kd_tree_with_labeled_points_spec)
    class KdTreeWithLabeledPoints:
        def __init__(self, max_size: int, dimension_count: int, max_label_size: int):
            '''Initialize the data structure allowing for at most max_size number of points
            and labels bounded by max_label_size. Dimension is set upfront.'''
            self.capacity = max_size
            self.num_active = 0
            self.next_free_idx = 0
            self.root = -1
            self.dim = dimension_count
            self.max_label_size = max_label_size

            self._max_query_depth = max_size

            # Initialize Tree Arrays
            self.points = np.zeros((max_size, dimension_count), dtype=coordinate_type)
            self.tree_labels = np.zeros(max_size, dtype=label_type)

            # Combined tree data: [left, right, axis, valid]
            self.tree_data = np.zeros((max_size, 4), dtype=count_type)
            self.tree_data[:, IDX_LEFT] = -1
            self.tree_data[:, IDX_RIGHT] = -1

            # Initialize Direct Array Map
            self.label_to_index_map = np.full(max_label_size, -1, dtype=count_type)

            # Pre-allocated stack for nearest neighbor queries
            self._query_stack = np.empty(self._max_query_depth, dtype=count_type)

        def _map_get(self, label: label_type) -> int:
            """Returns index in tree arrays, or -1 if not found."""
            if label < 0 or label >= self.max_label_size:
                return -1
            return self.label_to_index_map[label]

        def _map_put(self, label: label_type, tree_idx: int):
            """Insert or Update mapping label -> tree_idx."""
            self.label_to_index_map[label] = tree_idx

        def _map_delete(self, label: label_type):
            """Delete label mapping."""
            if label < 0 or label >= self.max_label_size:
                return
            self.label_to_index_map[label] = -1

        def __len__(self) -> int:
            return self.num_active

        def _rebuild(self):
            """Compacts arrays and rebuilds a balanced KD-Tree."""
            if self.num_active == 0:
                self.root = -1
                self.next_free_idx = 0
                self.label_to_index_map[:] = -1
                return

            write_ptr = 0
            for read_ptr in range(self.next_free_idx):
                if self.tree_data[read_ptr, IDX_VALID] == 1:
                    if read_ptr != write_ptr:
                        self.points[write_ptr] = self.points[read_ptr]
                        self.tree_labels[write_ptr] = self.tree_labels[read_ptr]
                    write_ptr += 1

            self.next_free_idx = write_ptr
            self.label_to_index_map[:] = -1
            for i in range(self.next_free_idx):
                lbl = self.tree_labels[i]
                self._map_put(lbl, i)
                self.tree_data[i, IDX_VALID] = 1
                self.tree_data[i, IDX_LEFT] = -1
                self.tree_data[i, IDX_RIGHT] = -1

            indices = np.arange(self.next_free_idx, dtype=count_type)
            self.root = _build_tree_recursive_njit(
                self.points, self.tree_data, indices, 0, self.dim, count_type, coordinate_type
            )

        def __setitem__(self, label: label_type, point: Sequence[coordinate_type]):
            if label < 0 or label >= self.max_label_size:
                raise IndexError("Label out of bounds for direct mapping.")

            existing_idx = self._map_get(label)
            if existing_idx != -1:
                self.tree_data[existing_idx, IDX_VALID] = 0
                self.num_active -= 1

            rebuild_needed = self.next_free_idx >= self.capacity or self.next_free_idx > self.num_active * REBUILD_RATIO
            if rebuild_needed:
                self._rebuild()
                if self.next_free_idx >= self.capacity:
                    raise IndexError("KdTreeWithLabeledPoints capacity full.")

            idx = self.next_free_idx
            self.next_free_idx += 1
            self.points[idx] = point
            self.tree_labels[idx] = label
            self.tree_data[idx, IDX_VALID] = 1
            self.tree_data[idx, IDX_LEFT] = -1
            self.tree_data[idx, IDX_RIGHT] = -1

            self._map_put(label, idx)
            self.num_active += 1

            if self.root == -1:
                self.root = idx
                self.tree_data[idx, IDX_AXIS] = 0
                return

            curr = self.root
            while True:
                axis = self.tree_data[curr, IDX_AXIS]
                if point[axis] < self.points[curr, axis]:
                    next_node = self.tree_data[curr, IDX_LEFT]
                    if next_node == -1:
                        self.tree_data[curr, IDX_LEFT] = idx
                        self.tree_data[idx, IDX_AXIS] = (axis + 1) % self.dim
                        break
                    curr = next_node
                else:
                    next_node = self.tree_data[curr, IDX_RIGHT]
                    if next_node == -1:
                        self.tree_data[curr, IDX_RIGHT] = idx
                        self.tree_data[idx, IDX_AXIS] = (axis + 1) % self.dim
                        break
                    curr = next_node

        def remove(self, label: label_type):
            idx = self._map_get(label)
            if idx == -1:
                raise KeyError("Label not found for deletion.")
            self.tree_data[idx, IDX_VALID] = 0
            self._map_delete(label)
            self.num_active -= 1

        def nearest(self, reference_point: Sequence[coordinate_type]) -> tuple[NDArray[coordinate_type], label_type]:
            '''Return any one nearest point and its label as measured from the reference point.'''
            if self.root == -1 or self.num_active == 0:
                raise ValueError("Tree is empty.")

            stack = self._query_stack
            stack[0] = self.root
            stack_top = 1

            min_dist_sq = np.inf
            best_idx = -1

            while stack_top > 0:
                stack_top -= 1
                curr = stack[stack_top]

                dist_sq = 0.0
                for d in range(self.dim):
                    diff = self.points[curr, d] - reference_point[d]
                    dist_sq += diff * diff

                if self.tree_data[curr, IDX_VALID] == 1:
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_idx = curr

                axis = self.tree_data[curr, IDX_AXIS]
                diff = reference_point[axis] - self.points[curr, axis]
                near_child = self.tree_data[curr, IDX_LEFT] if diff < 0 else self.tree_data[curr, IDX_RIGHT]
                far_child = self.tree_data[curr, IDX_RIGHT] if diff < 0 else self.tree_data[curr, IDX_LEFT]

                if far_child != -1 and (diff * diff) <= min_dist_sq + 1e-12:
                    stack[stack_top] = far_child
                    stack_top += 1
                if near_child != -1:
                    stack[stack_top] = near_child
                    stack_top += 1

            return self.points[best_idx], self.tree_labels[best_idx]

        def nearest_ties_labels_assign(
            self, reference_point: Sequence[coordinate_type], labels_buffer: NDArray[label_type]
        ) -> int:
            if self.root == -1 or self.num_active == 0:
                return 0

            stack = self._query_stack
            stack[0] = self.root
            stack_top = 1

            min_dist_sq = np.inf
            buffer_count = 0
            max_buffer_len = len(labels_buffer)

            while stack_top > 0:
                stack_top -= 1
                curr = stack[stack_top]

                dist_sq = 0.0
                for d in range(self.dim):
                    diff = self.points[curr, d] - reference_point[d]
                    dist_sq += diff * diff

                if self.tree_data[curr, IDX_VALID] == 1:
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        buffer_count = 0
                        labels_buffer[buffer_count] = self.tree_labels[curr]
                        buffer_count += 1
                    elif dist_sq == min_dist_sq:
                        if buffer_count < max_buffer_len:
                            labels_buffer[buffer_count] = self.tree_labels[curr]
                            buffer_count += 1
                        else:
                            raise IndexError("labels_buffer too small to hold all closest points.")

                axis = self.tree_data[curr, IDX_AXIS]
                diff = reference_point[axis] - self.points[curr, axis]
                near_child = self.tree_data[curr, IDX_LEFT] if diff < 0 else self.tree_data[curr, IDX_RIGHT]
                far_child = self.tree_data[curr, IDX_RIGHT] if diff < 0 else self.tree_data[curr, IDX_LEFT]

                if far_child != -1 and (diff * diff) <= min_dist_sq + 1e-12:
                    stack[stack_top] = far_child
                    stack_top += 1
                if near_child != -1:
                    stack[stack_top] = near_child
                    stack_top += 1

            return buffer_count

    return KdTreeWithLabeledPoints
