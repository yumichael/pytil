from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass


@njit(inline='always')
def stringify_1d_array(array):
    return '[' + ', '.join([str(entry) for entry in array]) + ']'


@cache
def create_array_stack_1d_items_jitclass(data_type):
    spec = (
        ('capacity', nb.int64),  # Maximum capacity of the stack
        ('item_size', nb.int64),  # Size of the each item
        ('size', nb.int64),  # Current size of the stack
        ('array', data_type[:, :]),  # Underlying NumPy array to store elements
    )

    @jitclass(spec)
    class ArrayStack1:
        def __init__(self, capacity, item_size):
            self.capacity = capacity
            self.item_size = item_size
            self.size = 0
            self.array = np.empty((capacity, item_size), dtype=data_type)

        def append(self, item):
            """
            Push an item onto the top of the stack.
            """
            if self.size >= self.capacity:
                raise IndexError("Stack is full")
            self.array[self.size] = item
            self.size += 1

        def pop(self):
            """
            Pop and return a copy of the top item from the stack.
            """
            if self.size == 0:
                raise IndexError("Pop from empty stack")
            value = self.array[self.size - 1].copy()
            self.size -= 1
            return value

        def pop_no_copy(self):
            """
            Pop and return a reference to the top item without copying.
            """
            if self.size == 0:
                raise IndexError("Pop from empty stack")
            value_ref = self.array[self.size - 1]
            self.size -= 1
            return value_ref

        def peek(self):
            """
            Return a reference to the top item without removing it.
            """
            if self.size == 0:
                raise IndexError("Peek from empty stack")
            return self.array[self.size - 1]

        def clear(self):
            """
            Remove all items from the stack.
            """
            self.size = 0

        def __getitem__(self, index):
            if index < -self.size or index >= self.size:
                raise IndexError("Index out of bounds")
            if index < 0:
                index += self.size
            return self.array[index]

        def __setitem__(self, index, value):
            if index < -self.size or index >= self.size:
                raise IndexError("Index out of bounds")
            if index < 0:
                index += self.size
            self.array[index] = value

        def __len__(self):
            return self.size

        def __str__(self):
            return '[' + ', '.join([stringify_1d_array(item) for item in self.array[: self.size]]) + ']'

    return ArrayStack1
