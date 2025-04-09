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
def create_array_heap_1d_items_jitclass(data_type):
    spec = (
        ('capacity', nb.int64),
        ('item_size', nb.int64),
        ('size', nb.int64),
        ('heap', data_type[:, :]),
    )

    @jitclass(spec)
    class ArrayHeap1:
        def __init__(self, capacity, item_size):
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

    return ArrayHeap1
