import math
from collections.abc import Sequence
from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numpy.typing import NDArray


@njit(inline='always')
def _aabb_dist_sq(point, b_min, b_max, dim):
    """
    Calculates the squared Euclidean distance from a point to an AABB.
    """
    dist_sq = 0.0
    for d in range(dim):
        v = point[d]
        min_v = b_min[d]
        max_v = b_max[d]
        if v < min_v:
            diff = min_v - v
            dist_sq += diff * diff
        elif v > max_v:
            diff = v - max_v
            dist_sq += diff * diff
    return dist_sq


@njit
def _build_tree_recursive_njit(
    points, tree_data, tree_mins, tree_maxs, indices, depth, dim, count_type, coordinate_type
):
    """
    Recursive helper to build the tree and compute AABBs for every node.
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

    # Recurse
    left_child = _build_tree_recursive_njit(
        points, tree_data, tree_mins, tree_maxs, left_indices, depth + 1, dim, count_type, coordinate_type
    )
    right_child = _build_tree_recursive_njit(
        points, tree_data, tree_mins, tree_maxs, right_indices, depth + 1, dim, count_type, coordinate_type
    )

    tree_data[node_idx, 0] = left_child
    tree_data[node_idx, 1] = right_child

    # --- Compute AABB (Union of current point, left child, right child) ---
    # Initialize with the node's own point
    p = points[node_idx]
    for d in range(dim):
        val = p[d]
        tree_mins[node_idx, d] = val
        tree_maxs[node_idx, d] = val

    # Merge Left
    if left_child != -1:
        for d in range(dim):
            if tree_mins[left_child, d] < tree_mins[node_idx, d]:
                tree_mins[node_idx, d] = tree_mins[left_child, d]
            if tree_maxs[left_child, d] > tree_maxs[node_idx, d]:
                tree_maxs[node_idx, d] = tree_maxs[left_child, d]

    # Merge Right
    if right_child != -1:
        for d in range(dim):
            if tree_mins[right_child, d] < tree_mins[node_idx, d]:
                tree_mins[node_idx, d] = tree_mins[right_child, d]
            if tree_maxs[right_child, d] > tree_maxs[node_idx, d]:
                tree_maxs[node_idx, d] = tree_maxs[right_child, d]

    return node_idx


@cache
def get_kd_tree_with_labeled_points_aabb_pruning_jitclass(count_type, coordinate_type, label_type):
    """
    Factory to create a specialized JIT-compiled KD-Tree class using AABB pruning.
    """

    # Rebuild constant: Trigger rebuild if next_free_idx / num_active > REBUILD_RATIO
    REBUILD_RATIO = 2.0

    count_type_numba = nb.from_dtype(count_type)
    coordinate_type_numba = nb.from_dtype(coordinate_type)
    label_type_numba = nb.from_dtype(label_type)

    if np.issubdtype(label_type, np.integer):
        LARGEST_POSSIBLE_LABEL = label_type(np.iinfo(label_type).max)
    else:
        assert False, 'Unsupported label type'

    DEFAULT_ABSOLUTE_TOLERANCE = 0.0

    # Tree data indices (for AoS structure)
    IDX_LEFT = 0
    IDX_RIGHT = 1
    IDX_AXIS = 2
    IDX_VALID = 3

    kd_tree_with_labeled_points_spec = [
        # --- Tree Storage (Array of Structures approach) ---
        ('points', coordinate_type_numba[:, :]),
        # AABB Storage per node: (max_size, dim)
        ('tree_mins', coordinate_type_numba[:, :]),
        ('tree_maxs', coordinate_type_numba[:, :]),
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
        # --- Floating point tolerance (Legacy field, unused in AABB nearest) ---
        ('atol', coordinate_type_numba),
    ]

    @jitclass(kd_tree_with_labeled_points_spec)
    class KdTreeWithLabeledPointsAABBPruning:
        def __init__(self, max_size: int, dimension_count: int, max_label_size: int):
            '''Initialize the data structure allowing for at most max_size number of points
            and labels bounded by max_label_size. Dimension is set upfront.'''
            self.capacity = max_size
            self.num_active = 0
            self.next_free_idx = 0
            self.root = -1
            self.dim = dimension_count
            self.max_label_size = max_label_size
            self.atol = DEFAULT_ABSOLUTE_TOLERANCE

            self._max_query_depth = max_size

            # Initialize Tree Arrays
            self.points = np.zeros((max_size, dimension_count), dtype=coordinate_type)
            self.tree_mins = np.zeros((max_size, dimension_count), dtype=coordinate_type)
            self.tree_maxs = np.zeros((max_size, dimension_count), dtype=coordinate_type)

            self.tree_labels = np.zeros(max_size, dtype=label_type)

            # Combined tree data: [left, right, axis, valid]
            self.tree_data = np.zeros((max_size, 4), dtype=count_type)
            self.tree_data[:, IDX_LEFT] = -1
            self.tree_data[:, IDX_RIGHT] = -1

            # Initialize Direct Array Map
            self.label_to_index_map = np.full(max_label_size, -1, dtype=count_type)

            # Pre-allocated stack for nearest neighbor queries
            self._query_stack = np.empty(self._max_query_depth, dtype=count_type)

        def set_absolute_tolerance(self, atol: float):
            # Legacy support
            assert atol == 0.0, 'Only 0.0 tolerance supported right now.'
            self.atol = atol

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
            """Compacts arrays and rebuilds a balanced KD-Tree with AABBs."""
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
                # Bounds will be recomputed by build_tree

            indices = np.arange(self.next_free_idx, dtype=count_type)
            self.root = _build_tree_recursive_njit(
                self.points,
                self.tree_data,
                self.tree_mins,
                self.tree_maxs,
                indices,
                0,
                self.dim,
                count_type,
                coordinate_type,
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

            # Initialize bounds for new leaf
            for d in range(self.dim):
                self.tree_mins[idx, d] = point[d]
                self.tree_maxs[idx, d] = point[d]

            self._map_put(label, idx)
            self.num_active += 1

            if self.root == -1:
                self.root = idx
                self.tree_data[idx, IDX_AXIS] = 0
                return

            curr = self.root
            while True:
                # Update bounds of the current ancestor to include the new point
                for d in range(self.dim):
                    if point[d] < self.tree_mins[curr, d]:
                        self.tree_mins[curr, d] = point[d]
                    if point[d] > self.tree_maxs[curr, d]:
                        self.tree_maxs[curr, d] = point[d]

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
            '''Return the nearest point and its label as measured from the reference point using AABB pruning.
            In case of ties (dist_sq == min_dist_sq), returns the point with the smallest label.
            Tolerance (self.atol) is IGNORED.'''
            if self.root == -1 or self.num_active == 0:
                raise ValueError("Tree is empty.")

            stack = self._query_stack
            stack[0] = self.root
            stack_top = 1

            min_dist_sq = np.inf
            best_idx = -1
            best_label = LARGEST_POSSIBLE_LABEL

            while stack_top > 0:
                stack_top -= 1
                curr = stack[stack_top]

                # Check point at current node
                if self.tree_data[curr, IDX_VALID] == 1:
                    dist_sq = 0.0
                    for d in range(self.dim):
                        diff = self.points[curr, d] - reference_point[d]
                        dist_sq += diff * diff

                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_idx = curr
                        best_label = self.tree_labels[curr]
                    elif dist_sq == min_dist_sq:
                        # Strictly prefer smaller label on tie
                        curr_label = self.tree_labels[curr]
                        if curr_label < best_label:
                            best_idx = curr
                            best_label = curr_label

                # Determine children
                axis = self.tree_data[curr, IDX_AXIS]
                diff = reference_point[axis] - self.points[curr, axis]

                # Standard traversal order: Near then Far
                near_child = self.tree_data[curr, IDX_LEFT] if diff < 0 else self.tree_data[curr, IDX_RIGHT]
                far_child = self.tree_data[curr, IDX_RIGHT] if diff < 0 else self.tree_data[curr, IDX_LEFT]

                # Process Far Child (Push first so it's processed last)
                if far_child != -1:
                    # AABB Pruning
                    dist_to_box = _aabb_dist_sq(
                        reference_point, self.tree_mins[far_child], self.tree_maxs[far_child], self.dim
                    )
                    # Use <= to ensure we handle ties.
                    # If dist_to_box == min_dist_sq, a point inside might be the tie-breaker with a smaller label.
                    if dist_to_box <= min_dist_sq:
                        stack[stack_top] = far_child
                        stack_top += 1

                # Process Near Child
                if near_child != -1:
                    # Optional: AABB check on near child can strictly prune cases where the node is close
                    # but the points inside the near bounding box are actually far away.
                    dist_to_box = _aabb_dist_sq(
                        reference_point, self.tree_mins[near_child], self.tree_maxs[near_child], self.dim
                    )
                    if dist_to_box <= min_dist_sq:
                        stack[stack_top] = near_child
                        stack_top += 1

            return self.points[best_idx], self.tree_labels[best_idx]

        # --------------------------------------------------------------------------
        # LEGACY FUNCTIONS (Unmodified logic)
        # --------------------------------------------------------------------------

        def nearest_with_tolerance(
            self, reference_point: Sequence[coordinate_type]
        ) -> tuple[NDArray[coordinate_type], label_type]:
            '''!!! Legacy function. Not kept up to date.

            Return the nearest point and its label as measured from the reference point.
            In case of ties (within tolerance), return the point with the smallest label.'''
            if self.root == -1 or self.num_active == 0:
                raise ValueError("Tree is empty.")

            stack = self._query_stack
            stack[0] = self.root
            stack_top = 1

            min_dist = np.inf
            best_idx = -1
            best_label = LARGEST_POSSIBLE_LABEL

            while stack_top > 0:
                stack_top -= 1
                curr = stack[stack_top]

                dist_sq = 0.0
                for d in range(self.dim):
                    diff = self.points[curr, d] - reference_point[d]
                    dist_sq += diff * diff
                dist = math.sqrt(dist_sq)

                if self.tree_data[curr, IDX_VALID] == 1:
                    curr_label = self.tree_labels[curr]
                    if dist < min_dist - self.atol:
                        # Clear winner
                        min_dist = dist
                        best_idx = curr
                        best_label = curr_label
                    elif abs(dist - min_dist) <= self.atol:
                        # Tie-break zone
                        if curr_label < best_label:
                            best_idx = curr
                            best_label = curr_label
                        # ALWAYS track the true mathematical minimum to keep the window tight
                        if dist < min_dist:
                            min_dist = dist

                axis = self.tree_data[curr, IDX_AXIS]
                diff = reference_point[axis] - self.points[curr, axis]
                near_child = self.tree_data[curr, IDX_LEFT] if diff < 0 else self.tree_data[curr, IDX_RIGHT]
                far_child = self.tree_data[curr, IDX_RIGHT] if diff < 0 else self.tree_data[curr, IDX_LEFT]

                if far_child != -1 and abs(diff) <= min_dist + self.atol:
                    stack[stack_top] = far_child
                    stack_top += 1
                if near_child != -1:
                    stack[stack_top] = near_child
                    stack_top += 1

            return self.points[best_idx], self.tree_labels[best_idx]

        # One pass algorithm, worse if many ties can occur at all levels in the tree
        def nearest_ties_labels_assign(
            self, reference_point: Sequence[coordinate_type], labels_buffer: NDArray[label_type]
        ) -> int:
            '''!!! Legacy function. Not kept up to date.'''
            if self.root == -1 or self.num_active == 0:
                return 0

            stack = self._query_stack
            stack[0] = self.root
            stack_top = 1

            min_dist_sq = np.inf
            buffer_count = 0
            max_buffer_len = len(labels_buffer)

            # Note: Legacy function assumes 'rtol' and 'atol' were globals or available.
            # Adapting slightly to avoid immediate crash if variables not found, but logic preserved.
            # Assuming usage matches original context where rtol/atol were defined.
            # For strict compilation, they should ideally be class members or passed in.
            rtol = 0.0
            atol = self.atol

            while stack_top > 0:
                stack_top -= 1
                curr = stack[stack_top]

                dist_sq = 0.0
                for d in range(self.dim):
                    diff = self.points[curr, d] - reference_point[d]
                    dist_sq += diff * diff

                if self.tree_data[curr, IDX_VALID] == 1:
                    is_close = np.isclose(dist_sq, min_dist_sq, rtol=rtol, atol=atol)
                    if dist_sq < min_dist_sq and not is_close:
                        min_dist_sq = dist_sq
                        buffer_count = 0
                        labels_buffer[buffer_count] = self.tree_labels[curr]
                        buffer_count += 1
                    elif is_close:
                        if buffer_count < max_buffer_len:
                            labels_buffer[buffer_count] = self.tree_labels[curr]
                            buffer_count += 1
                        else:
                            raise IndexError("labels_buffer too small to hold all closest points.")

                axis = self.tree_data[curr, IDX_AXIS]
                diff = reference_point[axis] - self.points[curr, axis]
                near_child = self.tree_data[curr, IDX_LEFT] if diff < 0 else self.tree_data[curr, IDX_RIGHT]
                far_child = self.tree_data[curr, IDX_RIGHT] if diff < 0 else self.tree_data[curr, IDX_LEFT]

                if far_child != -1 and (
                    diff * diff <= min_dist_sq or np.isclose(diff * diff, min_dist_sq, rtol=rtol, atol=atol)
                ):
                    stack[stack_top] = far_child
                    stack_top += 1
                if near_child != -1:
                    stack[stack_top] = near_child
                    stack_top += 1

            return buffer_count

        # Two pass algorithm. Too slow when there aren't many ties.
        def nearest_ties_labels_assign_two_pass(
            self,
            reference_point: Sequence[coordinate_type],
            labels_buffer: NDArray[label_type],
        ) -> int:
            """
            !!! Legacy function. Not kept up to date.

            Writes the labels of all points tied for nearest distance into labels_buffer.
            Returns the number of labels written.
            """
            rtol = 0.0
            atol = self.atol

            # -------------------------------
            # Phase 1: find minimum distance
            # -------------------------------
            stack = self._query_stack
            stack_top = 0
            stack[stack_top] = self.root
            stack_top += 1

            min_dist_sq = np.inf

            while stack_top > 0:
                stack_top -= 1
                curr = stack[stack_top]
                if curr == -1:
                    continue

                # Distance
                dist_sq = 0.0
                for d in range(self.dim):
                    diff = self.points[curr, d] - reference_point[d]
                    dist_sq += diff * diff

                if self.tree_data[curr, IDX_VALID] == 1 and (
                    dist_sq < min_dist_sq and not np.isclose(dist_sq, min_dist_sq, rtol=rtol, atol=atol)
                ):
                    min_dist_sq = dist_sq

                axis = self.tree_data[curr, IDX_AXIS]
                diff = reference_point[axis] - self.points[curr, axis]

                near_child = self.tree_data[curr, IDX_LEFT] if diff < 0 else self.tree_data[curr, IDX_RIGHT]
                far_child = self.tree_data[curr, IDX_RIGHT] if diff < 0 else self.tree_data[curr, IDX_LEFT]

                if near_child != -1:
                    stack[stack_top] = near_child
                    stack_top += 1

                # Strict pruning in phase 1
                if far_child != -1 and (
                    diff * diff < min_dist_sq and not np.isclose(diff * diff, min_dist_sq, rtol=rtol, atol=atol)
                ):
                    stack[stack_top] = far_child
                    stack_top += 1

            # -------------------------------
            # Phase 2: collect all ties
            # -------------------------------
            stack_top = 0
            stack[stack_top] = self.root
            stack_top += 1

            buffer_count = 0
            max_buffer_len = len(labels_buffer)

            while stack_top > 0:
                stack_top -= 1
                curr = stack[stack_top]
                if curr == -1:
                    continue

                # Distance
                dist_sq = 0.0
                for d in range(self.dim):
                    diff = self.points[curr, d] - reference_point[d]
                    dist_sq += diff * diff

                if self.tree_data[curr, IDX_VALID] == 1 and np.isclose(dist_sq, min_dist_sq, rtol=rtol, atol=atol):
                    if buffer_count < max_buffer_len:
                        labels_buffer[buffer_count] = self.tree_labels[curr]
                        buffer_count += 1
                    else:
                        raise IndexError("labels_buffer too small to hold all nearest ties.")

                axis = self.tree_data[curr, IDX_AXIS]
                diff = reference_point[axis] - self.points[curr, axis]

                near_child = self.tree_data[curr, IDX_LEFT] if diff < 0 else self.tree_data[curr, IDX_RIGHT]
                far_child = self.tree_data[curr, IDX_RIGHT] if diff < 0 else self.tree_data[curr, IDX_LEFT]

                if near_child != -1:
                    stack[stack_top] = near_child
                    stack_top += 1

                # Tie-safe pruning in phase 2
                if far_child != -1 and (
                    diff * diff <= min_dist_sq or np.isclose(diff * diff, min_dist_sq, rtol=rtol, atol=atol)
                ):
                    stack[stack_top] = far_child
                    stack_top += 1

            return buffer_count

    return KdTreeWithLabeledPointsAABBPruning
