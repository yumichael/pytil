from functools import cache

import numba as nb
import numpy as np
from numba.experimental import jitclass


@cache
def get_index_allocator_jitclass(count_type):
    """
    THIS CLASS IS UNUSED CURRENTLY. The abstraction is too simple to be necessary.

    Factory function to create an IndexAllocator jitclass specialized for a
    given integer type (count_type).

    :param count_type: The NumPy integer type (e.g., np.int32, np.int64)
                       to use for all capacity, index, and counter variables.
    """

    # 1. Convert NumPy type to Numba type
    assert issubclass(count_type.type, np.integer), "count_type must be an integer type."
    count_type_numba = nb.from_dtype(count_type)

    # 2. Define the spec using the Numba type
    alloc_spec = (
        ('capacity', count_type_numba),
        ('top', count_type_numba),
        ('free_list', count_type_numba[:]),
    )

    @jitclass(alloc_spec)
    class IndexAllocator:
        """
        A generic, fixed-capacity index manager using a stack-based free list.
        It is responsible only for allocating and freeing integer indices.
        """

        def __init__(self, capacity):
            self.capacity = capacity
            self.top = capacity
            # Initialize stack with all indices 0 to capacity-1
            self.free_list = np.empty(capacity, dtype=count_type)
            # Use a simple loop for initialization
            for i in range(capacity):
                self.free_list[i] = capacity - 1 - i

        def allocate(self):
            """Returns a free index. Raises error if full."""
            if self.top == 0:
                raise RuntimeError("Capacity exceeded: Index pool is full.")

            self.top -= 1
            return self.free_list[self.top]

        def free(self, index):
            """Returns an index to the free pool."""
            if self.top >= self.capacity:
                raise RuntimeError("Free list corruption: Attempted to free more than capacity.")

            self.free_list[self.top] = index
            self.top += 1

        def is_full(self):
            """Returns True if all indices are allocated."""
            return self.top == 0

        def is_empty(self):
            """Returns True if no indices are allocated (i.e., fully free)."""
            return self.top == self.capacity

        def __len__(self):
            """Returns the number of currently allocated (used) indices."""
            return self.capacity - self.top

    return IndexAllocator
