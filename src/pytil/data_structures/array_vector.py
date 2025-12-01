from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass


@njit(inline='always')
def _stringify_1d_array(array):
    s = "["
    for i in range(len(array)):
        s += str(array[i])
        if i < len(array) - 1:
            s += ", "
    s += "]"
    return s


@cache
def get_array_vector_1d_items_jitclass(data_type):
    """
    Factory function that creates a jitclass implementing an ArrayVector,
    supporting many of the operations of Python's list (now named vector).
    """
    spec = (
        ('capacity', nb.int64),  # maximum capacity
        ('item_size', nb.int64),  # size of each item (number of entries per row)
        ('size', nb.int64),  # current number of elements
        ('array', data_type[:, :]),  # underlying 2D NumPy array storing the items
    )

    @jitclass(spec)
    class ArrayVector:
        def __init__(self, capacity, item_size):
            self.capacity = capacity
            self.item_size = item_size
            self.size = 0
            self.array = np.empty((capacity, item_size), dtype=data_type)

        def append(self, item):
            """
            Append an item to the end of the vector.
            """
            if self.size >= self.capacity:
                raise IndexError("ArrayVector is full")
            self.array[self.size] = item
            self.size += 1

        def pop(self, index=-1):
            """
            Remove and return the item at the given index (default last).
            Shifts subsequent elements to the left.
            """
            if self.size == 0:
                raise IndexError("Pop from empty ArrayVector")
            if index < 0:
                index += self.size
            if index < 0 or index >= self.size:
                raise IndexError("Index out of range")
            result = self.array[index].copy()
            for i in range(index, self.size - 1):
                self.array[i] = self.array[i + 1]
            self.size -= 1
            return result

        def insert(self, index, item):
            """
            Insert an item at the given index.
            Shifts subsequent elements to the right.
            """
            if self.size >= self.capacity:
                raise IndexError("ArrayVector is full")
            if index < 0:
                index += self.size
            if index < 0:
                index = 0
            if index > self.size:
                index = self.size
            for i in range(self.size, index, -1):
                self.array[i] = self.array[i - 1]
            self.array[index] = item
            self.size += 1

        def remove(self, item):
            """
            Remove the first occurrence of an item.
            """
            for i in range(self.size):
                if self._equal(self.array[i], item):
                    self.__delitem__(i)
                    return
            raise ValueError("ArrayVector.remove(x): x not in vector")

        def _equal(self, a, b):
            """
            Check element-wise equality of two 1D arrays.
            """
            for i in range(self.item_size):
                if a[i] != b[i]:
                    return False
            return True

        def __getitem__(self, index):
            """
            Return the item at the given index (supports negative indices).
            """
            if index < 0:
                index += self.size
            if index < 0 or index >= self.size:
                raise IndexError("Index out of range")
            return self.array[index]

        def __setitem__(self, index, value):
            """
            Set the item at the given index (supports negative indices).
            """
            if index < 0:
                index += self.size
            if index < 0 or index >= self.size:
                raise IndexError("Index out of range")
            self.array[index] = value

        def __delitem__(self, index):
            """
            Delete the item at the given index, shifting elements to the left.
            """
            if index < 0:
                index += self.size
            if index < 0 or index >= self.size:
                raise IndexError("Index out of range")
            for i in range(index, self.size - 1):
                self.array[i] = self.array[i + 1]
            self.size -= 1

        def clear(self):
            """
            Remove all items from the vector.
            """
            self.size = 0

        def __len__(self):
            """
            Return the current number of elements.
            """
            return self.size

        def slice(self, start, stop):
            """
            Return a copy (as a 2D NumPy array) of a slice of the vector from start (inclusive)
            to stop (exclusive). Negative indices are supported.
            """
            if start < 0:
                start += self.size
            if stop < 0:
                stop += self.size
            if start < 0:
                start = 0
            if stop > self.size:
                stop = self.size
            if stop < start:
                return np.empty((0, self.item_size), dtype=self.array.dtype)
            length = stop - start
            out = np.empty((length, self.item_size), dtype=self.array.dtype)
            for i in range(length):
                out[i] = self.array[start + i]
            return out

        def __str__(self):
            """
            Return a string representation of the vector.
            """
            if self.size == 0:
                return "[]"
            s = "["
            for i in range(self.size):
                s += _stringify_1d_array(self.array[i])
                if i < self.size - 1:
                    s += ", "
            s += "]"
            return s

    return ArrayVector
