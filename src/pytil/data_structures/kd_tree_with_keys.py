from collections.abc import Sequence
from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numpy.typing import NDArray

# TODO Let's consider AoS vs SoA. By AoS, I mean make the relevant array variables a single 2-D array instead, and index into the second dimension with CONSTANTS to find which of the different values we want. I'm not familiar with the inner workings of the code, but isn't this at least possible to do with tree_left and tree_right? Maybe you can add tree_axis and tree_valid into the 2-D array as well. I don't know, maybe even you can add tree_keys if the logic makes sense, although you would have to unify the count_type with the key_type (which is fine for me by the way, in fact please unify the count_type with key_type if it meant you could make a more efficient AoS).

# TODO Would you be sure that with any possible reasonable number of operations the stack size of 64 is enough? Could unbalanced trees cause this to blow up?


@njit
def _build_tree_recursive_njit(
    points, tree_axis, tree_left, tree_right, indices, depth, dim, coordinate_type, count_type
):
    """
    External njit helper function to recursively build a balanced KD-Tree.
    (Numba jitclass does not support recursive methods.)
    """
    N = len(indices)
    if N == 0:
        return -1

    axis = depth % dim
    mid = N // 2

    # Sort indices based on points[idx, axis] to find median
    # This block performs a stable median selection/split
    vals = np.empty(N, dtype=coordinate_type)
    for i in range(N):
        vals[i] = points[indices[i], axis]

    sorted_arg_indices = np.argsort(vals)
    sorted_indices = indices[sorted_arg_indices]

    node_idx = sorted_indices[mid]
    tree_axis[node_idx] = axis

    left_indices = sorted_indices[:mid]
    right_indices = sorted_indices[mid + 1 :]

    tree_left[node_idx] = _build_tree_recursive_njit(
        points, tree_axis, tree_left, tree_right, left_indices, depth + 1, dim, coordinate_type, count_type
    )
    tree_right[node_idx] = _build_tree_recursive_njit(
        points, tree_axis, tree_left, tree_right, right_indices, depth + 1, dim, coordinate_type, count_type
    )

    return node_idx


@cache
def get_kd_tree_with_keys_jitclass(coordinate_type, key_type, count_type):
    """
    Factory to create a specialized JIT-compiled KD-Tree class.

    Args:
        coordinate_type: Dtype for point coordinates (e.g., np.float32 or np.float64).
        key_type: Dtype for external keys (e.g., np.int32).
        count_type: Dtype for internal indexing (e.g., np.int32).
    """

    # Rebuild constant: Trigger rebuild if next_free_idx / num_active > REBUILD_RATIO
    # This prevents tree degeneracy and high memory fragmentation.
    REBUILD_RATIO = 2.0

    # Fixed stack size for query recursion depth (log2(N) max)
    MAX_QUERY_DEPTH = 64
    coordinate_type_numba = nb.from_dtype(coordinate_type)
    key_type_numba = nb.from_dtype(key_type)
    count_type_numba = nb.from_dtype(count_type)

    # We need a boolean type for validity masks
    bool_numba = nb.boolean

    nearest_neighbor_map_spec = [
        # --- Tree Storage ---
        # (max_size, dim) - Flattened 2D array for point storage
        ('points', coordinate_type_numba[:, :]),
        # (max_size,) - Stores the external key for each internal node
        ('tree_keys', key_type_numba[:]),
        # (max_size,) - Index of left child
        ('tree_left', count_type_numba[:]),
        # (max_size,) - Index of right child
        ('tree_right', count_type_numba[:]),
        # (max_size,) - Splitting axis (0, 1, 2...)
        ('tree_axis', count_type_numba[:]),
        # (max_size,) - Boolean flag: is this node active?
        ('tree_valid', bool_numba[:]),
        # --- Tree State ---
        ('root', count_type_numba),
        ('next_free_idx', count_type_numba),  # Pointer to next empty slot in arrays
        ('num_active', count_type_numba),  # Actual count of valid items
        ('dim', count_type_numba),
        ('capacity', count_type_numba),
        # --- Direct Array Mapping ---
        # Map user key (0 to max_key_size-1) to tree index.
        ('key_to_index_map', count_type_numba[:]),
        ('max_key_size', count_type_numba),
        # --- Pre-allocated Query Stack ---
        ('_query_stack', count_type_numba[:]),
    ]

    @jitclass(nearest_neighbor_map_spec)
    class KdTreeWithKeys:
        def __init__(self, max_size: int, max_key_size: int, dimension_count: int):
            '''Initialize the data structure allowing for at most max_size number of points
            and keys bounded by max_key_size. Dimension is set upfront.'''
            self.capacity = max_size
            self.num_active = 0
            self.next_free_idx = 0
            self.root = -1
            self.dim = dimension_count
            self.max_key_size = max_key_size

            # Initialize Tree Arrays
            self.points = np.zeros((max_size, dimension_count), dtype=coordinate_type)
            self.tree_keys = np.zeros(max_size, dtype=key_type)

            # Initialize pointers to -1 (indicating null/none)
            self.tree_left = np.full(max_size, -1, dtype=count_type)
            self.tree_right = np.full(max_size, -1, dtype=count_type)
            self.tree_axis = np.zeros(max_size, dtype=count_type)
            self.tree_valid = np.zeros(max_size, dtype=bool_numba)

            # Initialize Direct Array Map
            # Value -1 means key is not present/deleted.
            self.key_to_index_map = np.full(max_key_size, -1, dtype=count_type)

            # Pre-allocated stack for nearest neighbor queries
            self._query_stack = np.empty(MAX_QUERY_DEPTH, dtype=count_type)

        @property
        def _map_bound(self) -> int:
            return self.max_key_size

        def _map_get(self, key: key_type) -> int:
            """Returns index in tree arrays, or -1 if not found."""
            if key < 0 or key >= self.max_key_size:
                return -1
            return self.key_to_index_map[key]

        def _map_put(self, key: key_type, tree_idx: int):
            """Insert or Update mapping key -> tree_idx."""
            # Bound check should be done by caller or here
            self.key_to_index_map[key] = tree_idx

        def _map_delete(self, key: key_type):
            """Delete key mapping."""
            if key < 0 or key >= self.max_key_size:
                return
            self.key_to_index_map[key] = -1

        def __len__(self) -> int:
            return self.num_active

        def _rebuild(self):
            """
            Compacts the arrays (removing holes) and rebuilds a balanced KD-Tree.
            """
            old_count = self.num_active
            if old_count == 0:
                self.root = -1
                self.next_free_idx = 0
                self.key_to_index_map[:] = -1
                return

            # 1. Compact data: Move valid nodes to the front
            write_ptr = 0
            for read_ptr in range(self.next_free_idx):
                if self.tree_valid[read_ptr]:
                    if read_ptr != write_ptr:
                        # Copy data
                        self.points[write_ptr] = self.points[read_ptr]
                        self.tree_keys[write_ptr] = self.tree_keys[read_ptr]

                    write_ptr += 1

            self.next_free_idx = write_ptr

            # Reset Direct Map to point to new indices
            self.key_to_index_map[:] = -1
            for i in range(self.next_free_idx):
                k = self.tree_keys[i]
                self._map_put(k, i)
                self.tree_valid[i] = True
                self.tree_left[i] = -1
                self.tree_right[i] = -1

            # Rebuild Tree (Balanced)
            indices = np.arange(self.next_free_idx, dtype=count_type)

            # Call the external njit helper function
            self.root = _build_tree_recursive_njit(
                self.points,
                self.tree_axis,
                self.tree_left,
                self.tree_right,
                indices,
                0,
                self.dim,
                coordinate_type,
                count_type,
            )

        def __setitem__(self, key: key_type, point: Sequence[coordinate_type]):
            if key < 0 or key >= self.max_key_size:
                raise IndexError("Key out of bounds for direct mapping.")

            # Check if key exists (Lazy Deletion)
            existing_idx = self._map_get(key)
            if existing_idx != -1:
                # Lazy deletion
                self.tree_valid[existing_idx] = False
                self.num_active -= 1

            # Check capacity and fragmentation to decide on rebuild (O(log n) amortized guarantee)
            rebuild_needed = self.next_free_idx >= self.capacity or self.next_free_idx > self.num_active * REBUILD_RATIO

            if rebuild_needed:
                self._rebuild()
                # Check capacity after potential compaction
                if self.next_free_idx >= self.capacity:
                    # If we are still full after rebuilding (i.e., num_active == capacity),
                    # we cannot insert the new item.
                    raise IndexError("KdTreeWithKeys capacity full.")

            # Insert new node
            idx = self.next_free_idx
            self.next_free_idx += 1

            # Copy Point
            self.points[idx] = point

            self.tree_keys[idx] = key
            self.tree_valid[idx] = True
            self.tree_left[idx] = -1
            self.tree_right[idx] = -1

            # Map Update
            self._map_put(key, idx)
            self.num_active += 1

            # Insert into Tree (Insertion is O(log n) but causes imbalance)
            if self.root == -1:
                self.root = idx
                self.tree_axis[idx] = 0
                return

            curr = self.root
            while True:
                axis = self.tree_axis[curr]
                val = self.points[curr, axis]
                p_val = point[axis]

                if p_val < val:
                    next_node = self.tree_left[curr]
                    if next_node == -1:
                        self.tree_left[curr] = idx
                        self.tree_axis[idx] = (axis + 1) % self.dim
                        break
                    curr = next_node
                else:
                    next_node = self.tree_right[curr]
                    if next_node == -1:
                        self.tree_right[curr] = idx
                        self.tree_axis[idx] = (axis + 1) % self.dim
                        break
                    curr = next_node

        def remove(self, key: key_type):
            idx = self._map_get(key)
            if idx == -1:
                raise KeyError("Key not found for deletion.")

            self.tree_valid[idx] = False
            self._map_delete(key)
            self.num_active -= 1

        def get_closest_points_assign(
            self, reference_point: Sequence[coordinate_type], keys_buffer: NDArray[key_type]
        ) -> int:
            if self.root == -1 or self.num_active == 0:
                return 0

            # Use pre-allocated stack for iterative traversal
            stack = self._query_stack
            stack_top = 0
            stack[0] = self.root
            stack_top = 1

            min_dist_sq = np.inf
            buffer_count = 0
            max_buffer_len = len(keys_buffer)

            while stack_top > 0:
                stack_top -= 1
                curr = stack[stack_top]

                # Distance calculation
                dist_sq = 0.0
                for d in range(self.dim):
                    # Accessing reference_point (Sequence/Tuple) directly
                    diff = self.points[curr, d] - reference_point[d]
                    dist_sq += diff * diff

                # Check valid candidate
                if self.tree_valid[curr]:
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        buffer_count = 0
                        keys_buffer[buffer_count] = self.tree_keys[curr]
                        buffer_count += 1
                    elif dist_sq == min_dist_sq:
                        if buffer_count < max_buffer_len:
                            keys_buffer[buffer_count] = self.tree_keys[curr]
                            buffer_count += 1
                        else:
                            raise IndexError("keys_buffer too small to hold all closest points.")

                # Tree traversal logic
                axis = self.tree_axis[curr]
                # Accessing reference_point (Sequence/Tuple) directly
                diff = reference_point[axis] - self.points[curr, axis]

                near_child = self.tree_left[curr] if diff < 0 else self.tree_right[curr]
                far_child = self.tree_right[curr] if diff < 0 else self.tree_left[curr]

                # Check Far Child (Pruning)
                if far_child != -1:
                    # Plane distance check
                    if (diff * diff) <= min_dist_sq + 1e-12:
                        if stack_top >= MAX_QUERY_DEPTH:
                            raise OverflowError("Query stack overflow: tree depth exceeds MAX_QUERY_DEPTH.")
                        stack[stack_top] = far_child
                        stack_top += 1

                # Check Near Child
                if near_child != -1:
                    if stack_top >= MAX_QUERY_DEPTH:
                        raise OverflowError("Query stack overflow: tree depth exceeds MAX_QUERY_DEPTH.")
                    stack[stack_top] = near_child
                    stack_top += 1

            return buffer_count

    return KdTreeWithKeys
