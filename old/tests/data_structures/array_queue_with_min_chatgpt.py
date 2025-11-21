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

        def enqueue(self, v):
            if self.size == self.maxsize:
                raise IndexError("enqueue on full queue")
            # place value at tail
            self.data[self.tail] = v
            self.pos_idx[self.tail] = self.idx_counter

            # maintain monotonic deque (increasing -> front is min)
            # pop from back while back value > v
            while self.deq_head < self.deq_tail and self.deq_val[self.deq_tail - 1] > v:
                self.deq_tail -= 1

            # append new (pos, value) to back
            self.deq_idx[self.deq_tail] = self.idx_counter
            self.deq_val[self.deq_tail] = v
            self.deq_tail += 1

            # advance circular buffer tail
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

            # if the deque front corresponds to this position, pop it
            if self.deq_head < self.deq_tail and self.deq_idx[self.deq_head] == pos:
                self.deq_head += 1

            # advance head
            self.head += 1
            if self.head == self.maxsize:
                self.head = 0
            self.size -= 1
            return v

        def get_min(self):
            if self.size == 0:
                raise IndexError("min from empty queue")
            return self.deq_val[self.deq_head]

        def get_size(self):
            return self.size

        def __len__(self):
            return self.size

    return QueueWithMin0


# Quick demonstration / tests
def _test():
    QueueWithMin = create_array_queue_with_min_0d_items_jitclass(int64)
    q = QueueWithMin(8)
    seq = np.array([5, 3, 7, 2, 2, 9, 1], dtype=np.int64)
    mins = []
    for x in seq:
        q.enqueue(x)
        mins.append(q.get_min())
    out = []
    while not q.is_empty():
        out.append(
            (q.dequeue(), mins[len(out)] if len(out) < len(mins) else None, q.get_min() if not q.is_empty() else None)
        )
    return seq.tolist(), mins, out


# More edge-case tests: duplicates and wrap-around
def _wrap_test():
    QueueWithMin = create_array_queue_with_min_0d_items_jitclass(int64)
    q = QueueWithMin(4)
    q.enqueue(4)
    q.enqueue(3)
    q.enqueue(3)
    q.enqueue(5)
    # queue is full now
    results = []
    results.append(q.get_min())  # should be 3
    results.append(q.dequeue())  # remove 4
    results.append(q.get_min())  # should still be 3
    results.append(q.dequeue())  # remove 3 (first)
    results.append(q.get_min())  # should still be 3 (second)
    q.enqueue(2)  # wrap-around enqueue
    results.append(q.get_min())  # should be 2
    return results


def _main():
    # Run test
    seq, mins, out = _test()
    print("Revision 1 - Basic correctness test")
    print("Enqueued sequence:", seq)
    print("Min after each enqueue:", mins)
    print("Dequeued items and intermediate mins (value, min_when_enqueued, min_after_dequeue):")
    for it in out:
        print(it)

    print("Wrap-around & duplicates test:", _wrap_test())


if __name__ == '__main__':
    _main()
