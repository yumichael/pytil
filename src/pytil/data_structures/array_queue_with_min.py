from functools import cache

import numba as nb
import numpy as np
from numba import int64
from numba.experimental import jitclass


@cache
def create_array_queue_with_min_0d_items_jitclass(data_type):
    """
    Factory function to create a queue with min class specialized for storing just single values (as opposed to a numpy array).
    """

    spec = [
        ('data', data_type[:]),  # circular buffer for values
        ('pos_idx', int64[:]),  # circular buffer for per-element unique position index (monotonic)
        ('maxsize', int64),
        ('head', int64),
        ('tail', int64),
        ('size', int64),
        ('idx_counter', int64),  # monotonic counter incremented on each enqueue
        # monotonic deque arrays (store positions and values)
        ('deq_idx', int64[:]),  # positions stored in deque
        ('deq_val', data_type[:]),
        ('deq_head', int64),
        ('deq_tail', int64),
        ('deq_size', int64),  # <-- ADDED: size counter for the deque
    ]

    @jitclass(spec)
    class QueueWithMin0:
        def __init__(self, maxsize):
            self.maxsize = maxsize
            self.data = np.empty(maxsize, dtype=data_type)
            self.pos_idx = np.empty(maxsize, dtype=np.int64)
            self.head = 0
            self.tail = 0
            self.size = 0
            self.idx_counter = 0

            # deque capacity can be at most maxsize (worst case monotonic increasing)
            self.deq_idx = np.empty(maxsize, dtype=np.int64)
            self.deq_val = np.empty(maxsize, dtype=data_type)
            self.deq_head = 0
            self.deq_tail = 0
            self.deq_size = 0  # <-- ADDED: initialize size

        def is_empty(self):
            return self.size == 0

        def is_full(self):
            return self.size == self.maxsize

        def clear(self):
            self.head = 0
            self.tail = 0
            self.size = 0
            self.idx_counter = 0
            self.deq_head = 0
            self.deq_tail = 0
            self.deq_size = 0  # <-- ADDED: clear size

        def enqueue(self, v):
            if self.size == self.maxsize:
                raise IndexError("enqueue on full queue")
            # place value at tail
            self.data[self.tail] = v
            self.pos_idx[self.tail] = self.idx_counter

            # --- MODIFIED: Deque logic (circular) ---
            # pop from back while back value > v
            while self.deq_size > 0:
                # Get the index of the last element
                back_idx = (self.deq_tail - 1 + self.maxsize) % self.maxsize
                if self.deq_val[back_idx] > v:
                    # "Pop" it by moving the tail pointer back
                    self.deq_tail = back_idx
                    self.deq_size -= 1
                else:
                    # Found a smaller or equal element, stop
                    break

            # append new (pos, value) to back (at current deq_tail)
            self.deq_idx[self.deq_tail] = self.idx_counter
            self.deq_val[self.deq_tail] = v

            # advance circular buffer deq_tail
            self.deq_tail = (self.deq_tail + 1) % self.maxsize
            self.deq_size += 1
            # --- END MODIFICATION ---

            # advance main circular buffer tail
            self.tail += 1
            if self.tail == self.maxsize:
                self.tail = 0
            self.size += 1
            self.idx_counter += 1

        def dequeue(self):
            if self.size == 0:
                raise IndexError("dequeue from empty queue")
            v = self.data[self.head]
            pos = self.pos_idx[self.head]

            # --- MODIFIED: Deque logic (circular) ---
            # if the deque front corresponds to this position, pop it
            if self.deq_size > 0 and self.deq_idx[self.deq_head] == pos:
                # advance deq_head
                self.deq_head = (self.deq_head + 1) % self.maxsize
                self.deq_size -= 1
            # --- END MODIFICATION ---

            # advance head
            self.head += 1
            if self.head == self.maxsize:
                self.head = 0
            self.size -= 1
            return v

        def get_min(self):
            if self.size == 0:
                raise IndexError("min from empty queue")
            # The check for self.size == 0 implicitly covers self.deq_size == 0
            return self.deq_val[self.deq_head]

        def get_size(self):
            return self.size

        def __len__(self):
            return self.size

    return QueueWithMin0
