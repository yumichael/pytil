from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass


@njit(inline='always')
def array_is_less(a, b):
    """
    Compare two arrays element-by-element.
    Return True if 'a' is lexicographically less than 'b';
    False otherwise.
    """
    m = len(a)
    for i in range(m):
        if a[i] < b[i]:
            return True
        elif a[i] > b[i]:
            return False
    # They are equal
    return False


@njit(inline='always')
def stringify_1d_array(array):
    return '[' + ', '.join([str(entry) for entry in array]) + ']'


@cache
def create_array_heap_1d_items_jitclass_slow(data_type):
    '''
    Creates a Heap class that stores indices in the heap structure.
    Puts the user data directly in the array heap, requiring sift up/down to shuffle user data directly all the time.
    '''

    spec = (
        ('capacity', nb.int64),
        ('item_size', nb.int64),
        ('size', nb.int64),
        ('heap', data_type[:, :]),
    )

    @jitclass(spec)
    class ArrayHeap1:
        def __init__(self, capacity, item_size):
            # print(f'Creating ArrayHeap1 with capacity', capacity, 'and item_size', item_size)
            self.capacity = capacity
            self.item_size = item_size
            self.size = 0
            self.heap = np.empty((capacity, item_size), dtype=data_type)

        def heappush(self, item):
            if self.size >= self.capacity:
                raise IndexError("Heap is full")
            self.heap[self.size] = item
            self.size += 1
            self._siftdown(0, self.size - 1)

        def heappop(self):
            if self.size == 0:
                raise IndexError("Pop from empty heap")
            lastelt = self.heap[self.size - 1].copy()
            self.size -= 1
            if self.size > 0:
                returnitem = self.heap[0].copy()
                self.heap[0] = lastelt
                self._siftup(0, self.size)
                return returnitem
            return lastelt

        def heappeek(self):
            if self.size == 0:
                raise IndexError("Peek from empty heap")
            return self.heap[0]  # Return reference

        def heapreplace(self, item):
            if self.size == 0:
                raise IndexError("Replace from empty heap")
            returnitem = self.heap[0].copy()
            self.heap[0] = item
            self._siftup(0, self.size)
            return returnitem

        def heappushpop(self, item):
            if self.size > 0 and array_is_less(self.heap[0], item):
                item, self.heap[0] = self.heap[0], item
                self._siftup(0, self.size)
            return item

        def heapify(self):
            for i in range(self.size // 2 - 1, -1, -1):
                self._siftup(i, self.size)

        def _siftdown(self, startpos, pos):
            newitem = self.heap[pos].copy()
            while pos > startpos:
                parentpos = (pos - 1) >> 1
                parent = self.heap[parentpos]
                if array_is_less(newitem, parent):
                    self.heap[pos] = parent
                    pos = parentpos
                    continue
                break
            self.heap[pos] = newitem

        def _siftup(self, pos, endpos):
            startpos = pos
            newitem = self.heap[pos].copy()
            childpos = 2 * pos + 1
            while childpos < endpos:
                rightpos = childpos + 1
                if rightpos < endpos and not array_is_less(self.heap[childpos], self.heap[rightpos]):
                    childpos = rightpos
                self.heap[pos] = self.heap[childpos]
                pos = childpos
                childpos = 2 * pos + 1
            self.heap[pos] = newitem
            self._siftdown(startpos, pos)

        def __getitem__(self, index):
            if index == 0:
                if self.size == 0:
                    raise IndexError("Index out of range for empty heap")
                # Return reference, not a copy
                return self.heap[0]
            raise IndexError("Only index 0 is supported")

        def __len__(self):
            return self.size

        def __str__(self):
            return '[' + ', '.join([stringify_1d_array(item) for item in self.heap[: self.size]]) + ']'

        insert = heappush

        def pop(self, index=0):
            if index == 0:
                return self.heappop()
            raise IndexError("Only index 0 is supported for pop")

    return ArrayHeap1


@cache
def create_array_heap_1d_items_jitclass(data_type):
    """
    Creates a Heap class that stores indices in the heap structure,
    referencing a separate data array.
    """
    spec = [
        ('capacity', nb.int64),
        ('item_size', nb.int64),
        ('size', nb.int64),
        ('heap', nb.int64[:]),  # Stores indices (the tree)
        ('data', data_type[:, :]),  # Stores actual data
        ('free_list', nb.int64[:]),  # Stack of available data slots
        ('free_list_top', nb.int64),  # Top pointer for free list
    ]

    @jitclass(spec)
    class ArrayHeap1Fast:
        def __init__(self, capacity, item_size):
            # print(f'Creating ArrayHeap1Fast with capacity', capacity, 'and item_size', item_size)
            self.capacity = capacity
            self.item_size = item_size
            self.size = 0

            # The heap structure (stores indices into self.data)
            self.heap = np.empty(capacity, dtype=np.int64)

            # The data storage
            self.data = np.empty((capacity, item_size), dtype=data_type)

            # Initialize free list
            self.free_list = np.empty(capacity, dtype=np.int64)
            for i in range(capacity):
                self.free_list[i] = capacity - 1 - i
            self.free_list_top = capacity

        def _alloc_node(self, item):
            """Pop a free slot index and fill it with data."""
            if self.free_list_top == 0:
                raise IndexError("Heap capacity exceeded")
            self.free_list_top -= 1
            idx = self.free_list[self.free_list_top]
            self.data[idx] = item
            return idx

        def _free_node(self, idx):
            """Push an index back onto the free list."""
            self.free_list[self.free_list_top] = idx
            self.free_list_top += 1

        def heappush(self, item):
            if self.size >= self.capacity:
                raise IndexError("Heap is full")

            # 1. Store data in a fixed location
            data_idx = self._alloc_node(item)

            # 2. Add that index to the end of the heap
            self.heap[self.size] = data_idx
            self.size += 1

            # 3. Sift the index up
            self._siftdown(0, self.size - 1)

        def heappop(self):
            if self.size == 0:
                raise IndexError("Pop from empty heap")

            # 1. Get the index currently at the root
            root_data_idx = self.heap[0]

            # 2. Copy the data to return it (since we are about to free the slot)
            return_item = self.data[root_data_idx].copy()

            # 3. Free the data slot
            self._free_node(root_data_idx)

            # 4. Move the last element to the root
            self.size -= 1
            if self.size > 0:
                last_data_idx = self.heap[self.size]
                self.heap[0] = last_data_idx
                self._siftup(0, self.size)

            return return_item

        def heappeek(self):
            if self.size == 0:
                raise IndexError("Peek from empty heap")
            # Retrieve data using the root index
            return self.data[self.heap[0]]

        def _siftdown(self, startpos, pos):
            """
            'siftdown' in Python heapq terminology actually means bubbling the
            newly added item *up* towards the root (index 0).
            """
            new_item_idx = self.heap[pos]
            # We look up the actual values to compare
            new_item_val = self.data[new_item_idx]

            while pos > startpos:
                parentpos = (pos - 1) >> 1
                parent_idx = self.heap[parentpos]
                parent_val = self.data[parent_idx]

                if array_is_less(new_item_val, parent_val):
                    self.heap[pos] = parent_idx
                    pos = parentpos
                    continue
                break
            self.heap[pos] = new_item_idx

        def _siftup(self, pos, endpos):
            """
            'siftup' in Python heapq terminology means taking the item at 'pos'
            and sinking it down the tree to its correct position.
            """
            startpos = pos
            new_item_idx = self.heap[pos]
            new_item_val = self.data[new_item_idx]

            childpos = 2 * pos + 1
            while childpos < endpos:
                rightpos = childpos + 1
                # Compare the two children via lookups
                child_idx = self.heap[childpos]
                if rightpos < endpos:
                    right_idx = self.heap[rightpos]
                    if not array_is_less(self.data[child_idx], self.data[right_idx]):
                        childpos = rightpos
                        child_idx = right_idx

                # Move child index up
                self.heap[pos] = child_idx
                pos = childpos
                childpos = 2 * pos + 1

            self.heap[pos] = new_item_idx
            # Standard optimization: bubble up just in case we went too far down
            self._siftdown(startpos, pos)

        def __len__(self):
            return self.size

        def __str__(self):
            # Helper to visualize; reconstructs the heap in list form
            items = []
            for i in range(self.size):
                items.append(stringify_1d_array(self.data[self.heap[i]]))
            return '[' + ', '.join(items) + ']'

        # Aliases
        insert = heappush

        def pop(self, index=0):
            if index == 0:
                return self.heappop()
            raise IndexError("Only index 0 is supported for pop")

    return ArrayHeap1Fast
