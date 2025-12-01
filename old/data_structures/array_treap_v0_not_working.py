from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass


@njit(inline='always')
def array_is_less(a, b):
    """
    Compare two arrays element-by-element for a strict lexicographical order.
    Return True if 'a' is lexicographically less than 'b', False otherwise.
    """
    for i in range(len(a)):
        if a[i] < b[i]:
            return True
        elif a[i] > b[i]:
            return False
    return False  # They are equal in all positions.


@njit(inline='always')
def stringify_1d_array(array):
    return '[' + ', '.join([str(entry) for entry in array]) + ']'


@cache
def get_array_treap_1d_items_jitclass(data_type):
    """
    Factory function to create a treap class specialized for 1D items
    using only 'array_is_less' to define BST ordering (and equality detection).
    """
    assert (
        data_type is nb.int64 or data_type is nb.float64
    ), 'Only 64-bit data types supported as the random key generation is defined up to 10**9'

    spec = [
        ('capacity', nb.int64),
        ('item_size', nb.int64),
        ('size', nb.int64),
        ('root', nb.int64),
        ('data', data_type[:, :]),
        ('priorities', nb.int64[:]),
        ('left', nb.int64[:]),
        ('right', nb.int64[:]),
        ('parent', nb.int64[:]),
        ('free_list', nb.int64[:]),
        ('free_list_top', nb.int64),
        ('subtree_size', nb.int64[:]),
    ]

    @jitclass(spec)
    class ArrayTreap1:
        """
        A treap (tree-based heap) implemented with implicit indexing,
        using 'array_is_less' to compare multi-dimensional items.
        """

        def __init__(self, capacity, item_size):
            """
            Initialize the treap with a maximum capacity and item size.
            Each row of 'data' is considered one item.
            """
            self.capacity = capacity
            self.item_size = item_size
            self.size = 0
            self.root = -1

            # Preallocate arrays
            self.data = np.empty((capacity, item_size), dtype=data_type)
            self.priorities = np.empty(capacity, dtype=np.int64)
            self.left = -np.ones(capacity, dtype=np.int64)
            self.right = -np.ones(capacity, dtype=np.int64)
            self.parent = -np.ones(capacity, dtype=np.int64)

            # Free list: stack of available node indices
            self.free_list = np.empty(capacity, dtype=np.int64)
            for i in range(capacity):
                self.free_list[i] = capacity - 1 - i
            self.free_list_top = capacity

            # For each node, store subtree size
            self.subtree_size = np.zeros(capacity, dtype=np.int64)

        def _recalc(self, x):
            """
            Recalculate the subtree size for node x.
            """
            s = 1
            if self.left[x] != -1:
                s += self.subtree_size[self.left[x]]
            if self.right[x] != -1:
                s += self.subtree_size[self.right[x]]
            self.subtree_size[x] = s

        def _new_node(self, item):
            """
            Allocate a new node from the free list and initialize it.
            """
            if self.free_list_top == 0:
                raise Exception("Treap capacity exceeded")

            self.free_list_top -= 1
            idx = self.free_list[self.free_list_top]
            self.data[idx] = item
            self.priorities[idx] = np.random.randint(1, 10**9)  # random priority
            self.left[idx] = -1
            self.right[idx] = -1
            self.parent[idx] = -1
            self.subtree_size[idx] = 1
            self.size += 1
            return idx

        def rotate_right(self, x):
            """
            Perform a right rotation around node x.
            """
            y = self.left[x]
            self.left[x] = self.right[y]
            if self.right[y] != -1:
                self.parent[self.right[y]] = x
            self.right[y] = x
            parent_x = self.parent[x]
            self.parent[y] = parent_x
            self.parent[x] = y

            if parent_x == -1:
                self.root = y
            else:
                if self.left[parent_x] == x:
                    self.left[parent_x] = y
                else:
                    self.right[parent_x] = y

            self._recalc(x)
            self._recalc(y)

        def rotate_left(self, x):
            """
            Perform a left rotation around node x.
            """
            y = self.right[x]
            self.right[x] = self.left[y]
            if self.left[y] != -1:
                self.parent[self.left[y]] = x
            self.left[y] = x
            parent_x = self.parent[x]
            self.parent[y] = parent_x
            self.parent[x] = y

            if parent_x == -1:
                self.root = y
            else:
                if self.left[parent_x] == x:
                    self.left[parent_x] = y
                else:
                    self.right[parent_x] = y

            self._recalc(x)
            self._recalc(y)

        def insert(self, item):
            """
            Insert a new item into the treap, maintaining BST order via array_is_less.
            """
            new_idx = self._new_node(item)

            # If tree is empty, new node is the root
            if self.root == -1:
                self.root = new_idx
                return

            # Search for insertion point
            cur = self.root
            parent = -1
            path = []
            while cur != -1:
                path.append(cur)
                parent = cur
                if array_is_less(item, self.data[cur]):
                    cur = self.left[cur]
                else:
                    # If not less, go right
                    cur = self.right[cur]

            # Attach new_idx to parent
            self.parent[new_idx] = parent
            if array_is_less(item, self.data[parent]):
                self.left[parent] = new_idx
            else:
                self.right[parent] = new_idx

            # Update subtree sizes along the path
            for node in path:
                self.subtree_size[node] += 1

            # Bubble up based on priority
            cur = new_idx
            while self.parent[cur] != -1 and self.priorities[self.parent[cur]] < self.priorities[cur]:
                p = self.parent[cur]
                if self.left[p] == cur:
                    self.rotate_right(p)
                else:
                    self.rotate_left(p)

        def __getitem__(self, index):
            """
            Retrieve the in-order item at position 'index', integer-only.
            Negative indices behave similarly to Python's sequence rules.
            """
            if index < 0:
                index += self.size
            if index < 0 or index >= self.size:
                raise IndexError("Index out of range")
            # We do a standard subtree-based 'select'
            node_idx = self.select(index)
            return self.data[node_idx]

        def slice(self, start, stop):
            """
            Return a slice of the treap's in-order traversal from 'start' (inclusive)
            to 'stop' (exclusive), as a 2D NumPy array (a copy).
            Negative indices behave like Python slices. Indices beyond the
            valid range are clamped. If stop < start after adjustments, the slice is empty.
            """
            # Handle negative indices for start
            if start < 0:
                start += self.size
            if start < 0:
                start = 0
            if start > self.size:
                start = self.size

            # Handle negative indices for stop
            if stop < 0:
                stop += self.size
            if stop < 0:
                stop = 0
            if stop > self.size:
                stop = self.size

            if stop < start:
                # Empty slice
                return np.empty((0, self.item_size), dtype=self.data.dtype)

            length = stop - start
            out = np.empty((length, self.item_size), dtype=self.data.dtype)

            # Partial in-order traversal
            stack = np.empty(self.capacity, dtype=np.int64)
            stack_top = 0
            cur = self.root
            count = 0  # Current in-order index
            out_pos = 0  # Position in 'out'

            while stack_top > 0 or cur != -1:
                if cur != -1:
                    stack[stack_top] = cur
                    stack_top += 1
                    cur = self.left[cur]
                else:
                    stack_top -= 1
                    cur = stack[stack_top]
                    # We are visiting 'cur' in in-order
                    if count >= start and count < stop:
                        out[out_pos] = self.data[cur]
                        out_pos += 1
                        if count + 1 == stop:
                            # Done collecting
                            break
                    count += 1
                    cur = self.right[cur]

            return out

        def search(self, item):
            """
            Search for a node whose stored item is 'item', returning the node index if found, else -1.
            We assume 'item' <-> 'self.data[cur]' if array_is_less says so,
            and equality is the case when neither is less than the other.
            """
            cur = self.root
            while cur != -1:
                if array_is_less(item, self.data[cur]):
                    cur = self.left[cur]
                elif array_is_less(self.data[cur], item):
                    cur = self.right[cur]
                else:
                    # Not less in either direction => they are equal
                    return cur
            return -1

        def remove(self, item):
            """
            Remove the node containing 'item' (found via BST search).
            Equivalent to the old 'delete(item)'.
            If 'item' is not found, no change is made.
            """
            # Find node
            cur = self.root
            while cur != -1:
                if array_is_less(item, self.data[cur]):
                    cur = self.left[cur]
                elif array_is_less(self.data[cur], item):
                    cur = self.right[cur]
                else:
                    # Found
                    break

            if cur == -1:
                return  # Not found

            self.delete(cur)

        def delete(self, node_idx):
            """
            Delete the node at index 'node_idx' directly, without searching by item.
            Bubbles the node down until it becomes a leaf, then removes it.
            """
            # Bubble down until it becomes a leaf
            while self.left[node_idx] != -1 or self.right[node_idx] != -1:
                if self.right[node_idx] == -1 or (
                    self.left[node_idx] != -1
                    and self.priorities[self.left[node_idx]] > self.priorities[self.right[node_idx]]
                ):
                    self.rotate_right(node_idx)
                else:
                    self.rotate_left(node_idx)

                # If node_idx's parent is -1, we've changed the root
                if self.parent[node_idx] == -1:
                    self.root = node_idx

            # Remove the leaf node from its parent
            parent = self.parent[node_idx]
            if parent != -1:
                if self.left[parent] == node_idx:
                    self.left[parent] = -1
                else:
                    self.right[parent] = -1

                # Update subtree sizes for all ancestors
                while parent != -1:
                    self._recalc(parent)
                    parent = self.parent[parent]

            # Return node_idx to the free list
            self.free_list[self.free_list_top] = node_idx
            self.free_list_top += 1
            self.size -= 1

        def delitem(self, index):
            """
            Remove the item at in-order position 'index'. Negative indexing is supported.
            Uses select(index) to locate the node, then calls delete(node_idx).
            """
            if index < 0:
                index += self.size
            if index < 0 or index >= self.size:
                raise IndexError("Index out of range")

            node_idx = self.select(index)
            if node_idx == -1:
                # Shouldn't happen if index is valid, but just in case
                raise IndexError("Index out of range")

            self.delete(node_idx)

        def select(self, order):
            """
            Return the index of the node that is the `order`-th element
            in the in-order (sorted) traversal, 0-indexed.
            """
            if order < 0 or order >= self.size:
                return -1
            cur = self.root
            while cur != -1:
                left_count = self.subtree_size[self.left[cur]] if self.left[cur] != -1 else 0
                if order < left_count:
                    cur = self.left[cur]
                elif order == left_count:
                    return cur
                else:
                    order -= left_count + 1
                    cur = self.right[cur]
            return -1

        def inorder(self):
            """
            Perform an in-order traversal and return a 1D NumPy array
            of node indices in 'sorted' order (as defined by array_is_less).
            """
            result = np.empty(self.size, dtype=np.int64)
            stack = np.empty(self.capacity, dtype=np.int64)
            stack_top = 0
            cur = self.root
            count = 0

            while stack_top > 0 or cur != -1:
                if cur != -1:
                    stack[stack_top] = cur
                    stack_top += 1
                    cur = self.left[cur]
                else:
                    stack_top -= 1
                    cur = stack[stack_top]
                    result[count] = cur
                    count += 1
                    cur = self.right[cur]
            return result

        def get(self, idx):
            """
            Retrieve a direct reference to the item stored at node index 'idx'.
            """
            return self.data[idx]

        def __str__(self):
            """
            Return a string representation of the treap's items in in-order order.
            Uses `stringify_1d_array` for formatting.
            """
            if self.size == 0:
                return "[]"
            inord = self.inorder()
            elems = [stringify_1d_array(self.data[node]) for node in inord]
            return "[" + ", ".join(elems) + "]"

    return ArrayTreap1
