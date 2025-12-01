from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass


@njit(inline='always')
def stringify_1d_array(array):
    return '[' + ', '.join([str(entry) for entry in array]) + ']'


@cache
def get_array_deque_1d_items_jitclass(data_type):
    spec = (
        ('capacity', nb.int64),  # total capacity
        ('item_size', nb.int64),  # size of each item
        ('size', nb.int64),  # current number of elements
        ('start', nb.int64),  # index of the left (front) element
        ('array', data_type[:, :]),
    )

    @jitclass(spec)
    class ArrayDeque1:
        def __init__(self, capacity, item_size):
            self.capacity = capacity
            self.item_size = item_size
            self.size = 0
            self.start = 0
            self.array = np.empty((capacity, item_size), dtype=data_type)

        def append(self, item):
            if self.size >= self.capacity:
                raise IndexError("Deque is full")
            end_index = (self.start + self.size) % self.capacity
            self.array[end_index] = item
            self.size += 1

        def pop(self):
            if self.size == 0:
                raise IndexError("Pop from empty deque")
            end_index = (self.start + self.size - 1) % self.capacity
            value = self.array[end_index].copy()
            self.size -= 1
            return value

        def pop_no_copy(self):
            if self.size == 0:
                raise IndexError("Pop from empty deque")
            end_index = (self.start + self.size - 1) % self.capacity
            value_ref = self.array[end_index]
            self.size -= 1
            return value_ref

        def appendleft(self, item):
            if self.size >= self.capacity:
                raise IndexError("Deque is full")
            left_index = (self.start - 1) % self.capacity
            self.array[left_index] = item
            self.start = left_index
            self.size += 1

        def popleft(self):
            if self.size == 0:
                raise IndexError("Popleft from empty deque")
            value = self.array[self.start].copy()
            self.start = (self.start + 1) % self.capacity
            self.size -= 1
            return value

        def popleft_no_copy(self):
            if self.size == 0:
                raise IndexError("Popleft from empty deque")
            value_ref = self.array[self.start]
            self.start = (self.start + 1) % self.capacity
            self.size -= 1
            return value_ref

        def peek(self):
            if self.size == 0:
                raise IndexError("Peek from empty deque")
            return self.array[(self.start + self.size - 1) % self.capacity]

        def peekleft(self):
            if self.size == 0:
                raise IndexError("Peek from empty deque")
            return self.array[self.start]

        def __getitem__(self, index):
            if index < 0:
                index += self.size
            if index < 0 or index >= self.size:
                raise IndexError("Index out of range")
            actual_index = (self.start + index) % self.capacity
            return self.array[actual_index]

        def __setitem__(self, index, value):
            if index < 0:
                index += self.size
            if index < 0 or index >= self.size:
                raise IndexError("Index out of range")
            actual_index = (self.start + index) % self.capacity
            self.array[actual_index] = value

        def clear(self):
            self.size = 0
            self.start = 0

        def __len__(self):
            return self.size

        def __str__(self):
            if self.size == 0:
                return "[]"
            elems = []
            for i in range(self.size):
                idx = (self.start + i) % self.capacity
                elems.append(stringify_1d_array(self.array[idx]))
            return '[' + ', '.join(elems) + ']'

    return ArrayDeque1
