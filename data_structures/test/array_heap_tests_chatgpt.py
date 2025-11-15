# %% [markdown]
# Benchmark results: njit all array heap operations are up to 4 times faster than njit list of array item operations for item size of 3.
# They still don't break even even at item size 200.
# Means array copy and swap is incredibly fast. Either that or dynamic allocation of array is incredible slow.
# I didn't test njit list of tuples because you cannot freely create size n tuples, so the item size has to be hard coded before compilation.
# 

# %%
%load_ext autoreload
%autoreload 3

import numpy as np

import numba as nb
from numba import njit
from numba import int64, njit
from numba.typed import List
from numba.types import Tuple

# %% [markdown]
# # Notice
# Find and replace `item_size = {n}` to change the item_size for all tests.

# %% [markdown]
# ### Testing array_heapq library
# 

# %%
from array_heapq import *

# %% [markdown]
# ##### Heapify test
# 

# %%
capacity = 1_000
item_size = 20
array = np.empty((capacity, item_size), dtype=np.int64)
size = capacity
for i in range(size):
    array[i] = np.random.randint(0, size, size=(item_size,))
count_holder = np.array([size])
array_heapify(array, count_holder)
output = []
for count in range(size, 0, -1):
    output.append(array_heappop(array, count_holder))
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %%
@njit
def test_array_heapify(capacity, array, size, item_size):
    for i in range(size):
        array[i] = np.random.randint(0, capacity, size=(item_size,))
    count_holder = np.array([size])
    array_heapify(array, count_holder)
    output = []
    for count in range(size, 0, -1):
        output.append(array_heappop(array, count_holder))
    return output

# %%
capacity = 1_000_000
item_size = 20
array = np.empty((capacity, item_size), dtype=np.int64)
size = capacity
output = test_array_heapify(capacity, array, size, item_size)
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %% [markdown]
# ##### Heappush test
# 

# %%
capacity = 1_000
item_size = 20
array = np.empty((capacity, item_size), dtype=np.int64)
size = capacity
count_holder = np.array([0])
for count in range(size):
    array_heappush(array, count_holder, np.random.randint(0, capacity, size=(item_size,)))
output = []
for count in range(size, 0, -1):
    output.append(array_heappop(array, count_holder))
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %%
@njit
def test_array_heappush(capacity, array, size, item_size):
    count_holder = np.array([0])
    for count in range(size):
        array_heappush(array, count_holder, np.random.randint(0, capacity, size=(item_size,)))
    output = []
    for count in range(size, 0, -1):
        output.append(array_heappop(array, count_holder))
    return output

# %%
capacity = 1_000_000
item_size = 20
array = np.empty((capacity, item_size), dtype=np.int64)
size = capacity
output = test_array_heappush(capacity, array, size, item_size)
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %% [markdown]
# ##### Interspersed push and pop tests
# 

# %%
capacity = 200
item_size = 20
iterations = capacity * 2_000
array = np.empty((capacity, item_size), dtype=np.int64)
size = capacity // 2
count_holder = np.array([0])
for count in range(size):
    array_heappush(array, count_holder, np.random.randint(0, capacity, size=(item_size,)))
for i in range(iterations):
    if np.random.randint(0, 2) == 0 and size != 0 or size == capacity:
        elem = array_heappop(array, count_holder)
        size -= 1
    else:
        array_heappush(array, count_holder, np.random.randint(0, capacity, size=(item_size,)))
        size += 1
    assert count_holder[0] == size
output = []
for count in range(size, 0, -1):
    output.append(array_heappop(array, count_holder))
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]
print(len(output))

# %%
@njit
def test_array_interspersed(capacity, array, size, iterations, item_size):
    count_holder = np.array([0])
    for count in range(size):
        array_heappush(array, count_holder, np.random.randint(0, capacity, size=(item_size,)))
    for i in range(iterations):
        if np.random.randint(0, 2) == 0 and size != 0 or size == capacity:
            elem = array_heappop(array, count_holder)
            size -= 1
        else:
            array_heappush(array, count_holder, np.random.randint(0, capacity, size=(item_size,)))
            size += 1
        assert count_holder[0] == size
    output = []
    for count in range(size, 0, -1):
        output.append(array_heappop(array, count_holder))
    return output

# %%
capacity = 2_000
item_size = 20
iterations = capacity * 2_000
array = np.empty((capacity, item_size), dtype=np.int64)
size = capacity // 2
output = test_array_interspersed(capacity, array, size, iterations, item_size)
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]
print(len(output))

# %% [markdown]
# ### Testing Numba List of np.arrays heapq
# 

# %%

@njit(inline='always')
def is_less(a, b):
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

item_size = 20
@njit
def njit_np_heappush(heap, item):
    heap.append(item)
    _siftdown_np(heap, 0, len(heap) - 1)

@njit
def njit_np_heappop(heap):
    lastelt = heap.pop()
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_np(heap, 0)
        return returnitem
    return lastelt

@njit
def njit_np_heapify(heap):
    n = len(heap)
    for i in range(n // 2 - 1, -1, -1):
        _siftup_np(heap, i)

@njit
def _siftdown_np(heap, startpos, pos):
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if is_less(newitem, parent):
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

@njit
def _siftup_np(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not is_less(heap[childpos], heap[rightpos]):
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    _siftdown_np(heap, startpos, pos)

# %% [markdown]
# ##### Heapify test
# 

# %%
size = 1_000
item_size = 20
heap_np = List.empty_list(nb.int64[:])
for count in range(size):
    heap_np.append(np.random.randint(0, 1000000, size=(item_size,), dtype=np.int64))
njit_np_heapify(heap_np)
output = [njit_np_heappop(heap_np) for _ in range(size)]
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %%
output

# %% [markdown]
# ##### Heappush test
# 

# %%
size = 1_000_000
item_size = 20
heap_np = List.empty_list(np.ndarray)
for count in range(size):
    njit_np_heappush(heap_np, np.random.randint(0, 1000000, size=(item_size,), dtype=np.int64))
output = [njit_np_heappop(heap_np) for _ in range(size)]
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %% [markdown]
# ##### Interspersed push and pop tests
# 

# %%
capacity = 2_000
item_size = 20
iterations = capacity * 2_000
heap_np = List.empty_list(np.ndarray)
size = capacity // 2
for count in range(size):
    njit_np_heappush(heap_np, np.random.randint(0, capacity, size=(item_size,), dtype=np.int64))
for _ in range(iterations):
    if np.random.randint(0, 2) == 0 and size != 0 or size == capacity:
        njit_np_heappop(heap_np)
        size -= 1
    else:
        njit_np_heappush(heap_np, np.random.randint(0, capacity, size=(item_size,), dtype=np.int64))
        size += 1
output = [njit_np_heappop(heap_np) for _ in range(size)]
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]
print(len(output))

# %%
@njit
def test_njit_np_interspersed(heap_np, size, iterations, item_size):
    for count in range(size):
        njit_np_heappush(heap_np, np.random.randint(0, 1000000, size=(item_size,), dtype=np.int64))
    for i in range(iterations):
        if np.random.randint(0, 2) == 0 and size != 0 or size == 20_000:
            njit_np_heappop(heap_np)
            size -= 1
        else:
            njit_np_heappush(heap_np, np.random.randint(0, 1000000, size=(item_size,), dtype=np.int64))
            size += 1
    output = []
    for count in range(size, 0, -1):
        output.append(njit_np_heappop(heap_np))
    return output

# %%
capacity = 2_000
item_size = 20
iterations = capacity * 2_000
heap_np = List.empty_list(np.ndarray)
size = capacity // 2
output = test_njit_np_interspersed(heap_np, size, iterations, item_size)
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]
print(len(output))

# %% [markdown]
# ##### njit_np Heapify test
# 

# %%
capacity = 1_000
item_size = 20
# For the njit_np_* functions, we use a Numba typed list of np.arrays.
from numba.typed import List
heap = List.empty_list(nb.int64[:])
for count in range(capacity):
    heap.append(np.random.randint(0, capacity, size=(item_size,)))
njit_np_heapify(heap)
output = []
for count in range(capacity, 0, -1):
    output.append(njit_np_heappop(heap))
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %%
@njit
def test_njit_np_heapify(heap, size, item_size, capacity):
    for i in range(size):
        heap.append(np.random.randint(0, capacity, size=(item_size,)))
    njit_np_heapify(heap)
    output = []
    for i in range(size, 0, -1):
        output.append(njit_np_heappop(heap))
    return output

# %%
capacity = 1_000_000
item_size = 20
size = capacity
heap = List.empty_list(nb.int64[:])
output = test_njit_np_heapify(heap, size, item_size, capacity)
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %% [markdown]
# ##### njit_np Heappush test
# 

# %%
capacity = 1_000
item_size = 20
size = capacity
heap = List.empty_list(nb.int64[:])
for count in range(size):
    njit_np_heappush(heap, np.random.randint(0, capacity, size=(item_size,)))
output = []
for count in range(size, 0, -1):
    output.append(njit_np_heappop(heap))
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %%
@njit
def test_njit_np_heappush(heap, size, item_size, capacity):
    for count in range(size):
        njit_np_heappush(heap, np.random.randint(0, capacity, size=(item_size,)))
    output = []
    for count in range(size, 0, -1):
        output.append(njit_np_heappop(heap))
    return output

# %%
capacity = 1_000_000
item_size = 20
size = capacity
heap = List.empty_list(nb.int64[:])
output = test_njit_np_heappush(heap, size, item_size, capacity)
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %% [markdown]
# ##### njit_np Interspersed push and pop tests
# 

# %%
capacity = 200
item_size = 20
iterations = capacity * 2_000
size = capacity // 2
heap = List.empty_list(nb.int64[:])
for count in range(size):
    njit_np_heappush(heap, np.random.randint(0, capacity, size=(item_size,)))
for i in range(iterations):
    if np.random.randint(0, 2) == 0 and size != 0 or size == capacity:
        njit_np_heappop(heap)
        size -= 1
    else:
        njit_np_heappush(heap, np.random.randint(0, capacity, size=(item_size,)))
        size += 1
output = []
for count in range(size, 0, -1):
    output.append(njit_np_heappop(heap))
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]
print(len(output))

# %%
@njit
def test_njit_np_interspersed(capacity, heap, size, iterations, item_size):
    for count in range(size):
        njit_np_heappush(heap, np.random.randint(0, capacity, size=(item_size,)))
    for i in range(iterations):
        if np.random.randint(0, 2) == 0 and size != 0 or size == capacity:
            njit_np_heappop(heap)
            size -= 1
        else:
            njit_np_heappush(heap, np.random.randint(0, capacity, size=(item_size,)))
            size += 1
    output = []
    for count in range(size, 0, -1):
        output.append(njit_np_heappop(heap))
    return output

# %%
capacity = 2_000
item_size = 20
iterations = capacity * 2_000
size = capacity // 2
heap = List.empty_list(nb.int64[:])
output = test_njit_np_interspersed(capacity, heap, size, iterations, item_size)
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]
print(len(output))

# %% [markdown]
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################
# Below is code that don't work!


# %% [markdown]
# ### Testing array_heapq without holder for the size
# 

# %%
capacity = 1_000
item_size = 20
array = np.empty((capacity, item_size), dtype=np.int64)
size = capacity
for i in range(size):
    array[i] = np.random.randint(0, size, size=(item_size,))
array_heapify(array, size)
output = []
for count in range(size, 0, -1):
    output.append(array_heappop(array, size))
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %%
@njit
def test_array_heapify(capacity, array, size, item_size):
    for i in range(size):
        array[i] = np.random.randint(0, capacity, size=(item_size,))
    array_heapify(array, size)
    output = []
    for count in range(size, 0, -1):
        output.append(array_heappop(array, size))
    return output

# %%
capacity = 1_000_000
item_size = 20
array = np.empty((capacity, item_size), dtype=np.int64)
size = capacity
output = test_array_heapify(capacity, array, size, item_size)
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %% [markdown]
# ##### Heappush test
# 

# %%
capacity = 1_000_000
item_size = 20
array = np.empty((capacity, item_size), dtype=np.int64)
size = capacity
for count in range(size):
    array_heappush(array, size, np.random.randint(0, capacity, size=(item_size,)))
output = []
for count in range(size, 0, -1):
    output.append(array_heappop(array, size))
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %%
@njit
def test_array_heappush(capacity, array, size, item_size):
    for count in range(size):
        array_heappush(array, size, np.random.randint(0, capacity, size=(item_size,)))
    output = []
    for count in range(size, 0, -1):
        output.append(array_heappop(array, size))
    return output

# %%
capacity = 1_000_000
item_size = 20
array = np.empty((capacity, item_size), dtype=np.int64)
size = capacity
output = test_array_heappush(capacity, array, size, item_size)
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]

# %% [markdown]
# ##### Interspersed push and pop tests
# 

# %%
capacity = 2_000
item_size = 20
iterations = capacity * 2_000
array = np.empty((capacity, item_size), dtype=np.int64)
size = capacity // 2
for count in range(size):
    array_heappush(array, size, np.random.randint(0, capacity, size=(item_size,)))
for i in range(iterations):
    if np.random.randint(0, 2) == 0 and size != 0 or size == capacity:
        elem = array_heappop(array, size)
        size -= 1
    else:
        array_heappush(array, size, np.random.randint(0, capacity, size=(item_size,)))
        size += 1
output = []
for count in range(size, 0, -1):
    output.append(array_heappop(array, size))
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]
print(len(output))

# %%
@njit
def test_array_interspersed(capacity, array, size, iterations, item_size):
    for count in range(size):
        array_heappush(array, size, np.random.randint(0, capacity, size=(item_size,)))
    for i in range(iterations):
        if np.random.randint(0, 2) == 0 and size != 0 or size == capacity:
            elem = array_heappop(array, size)
            size -= 1
        else:
            array_heappush(array, size, np.random.randint(0, capacity, size=(item_size,)))
            size += 1
    output = []
    for count in range(size, 0, -1):
        output.append(array_heappop(array, size))
    return output

# %%
capacity = 2_000
item_size = 20
iterations = capacity * 2_000
array = np.empty((capacity, item_size), dtype=np.int64)
size = capacity // 2
output = test_array_interspersed(capacity, array, size, iterations, item_size)
assert [tuple(x) for x in sorted(output, key=lambda x: tuple(x))] == [tuple(x) for x in output]
print(len(output))

# %% [markdown]
# ### Testing njit list heapq
# 

# %%

@njit(inline='always')
def is_less(a, b):
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

@njit
def heappush(heap, item):
    heap.append(item)
    _siftdown(heap, 0, len(heap) - 1)

@njit
def heappop(heap):
    lastelt = heap.pop()
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt

@njit
def heapreplace(heap, item):
    returnitem = heap[0]
    heap[0] = item
    _siftup(heap, 0)
    return returnitem

@njit
def heappushpop(heap, item):
    if heap and heap[0] < item:
        item, heap[0] = heap[0], item
        _siftup(heap, 0)
    return item

@njit
def heapify(x):
    n = len(x)
    for i in range(n // 2 - 1, -1, -1):
        _siftup(x, i)

@njit
def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

@njit
def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)

njit_heappush = heappush
njit_heappop = heappop
njit_heapreplace = heapreplace
njit_heappushpop = heappushpop
njit_heapify = heapify

# %% [markdown]
# ##### Heapify test
# 

# %%
size = 1_000
item_size = 20
array = List.empty_list(Tuple((int64, int64, int64)))
for count in range(size):
    array.append(tuple(np.random.randint(0, capacity, size=item_size)))
njit_heapify(array)
output = [njit_heappop(array) for _ in range(size)]
assert sorted(output, key=lambda x: x) == output

# %%
@njit
def test_njit_heapify(array, size, item_size):
    for i in range(size):
        array.append(tuple(np.random.randint(0, capacity, size=item_size)))
    njit_heapify(array)
    output = []
    for count in range(size, 0, -1):
        output.append(njit_heappop(array))
    return output

# %%
size = 1_000_000
item_size = 20
array = List.empty_list(Tuple((int64, int64, int64)))
output = test_njit_heapify(array, size, item_size)
assert sorted(output, key=lambda x: x) == output

# %% [markdown]
# ##### Heappush test
# 

# %%
size = 1_000_000
item_size = 20
array = List.empty_list(Tuple((int64, int64, int64)))
for count in range(size):
    njit_heappush(array, tuple(np.random.randint(0, capacity, size=item_size)))
output = [njit_heappop(array) for _ in range(size)]
assert sorted(output, key=lambda x: x) == output

# %%
@njit
def test_njit_heappush(array, size, item_size):
    for count in range(size):
        njit_heappush(array, tuple(np.random.randint(0, capacity, size=item_size)))
    output = []
    for count in range(size, 0, -1):
        output.append(njit_heappop(array))
    return output

# %%
size = 1_000_000
item_size = 20
array = List.empty_list(Tuple((int64, int64, int64)))
output = test_njit_heappush(array, size, item_size)
assert sorted(output, key=lambda x: x) == output

# %% [markdown]
# ##### Interspersed push and pop tests
# 

# %%
capacity = 2_000
item_size = 20
iterations = capacity * 2_000
array = List.empty_list(Tuple((int64, int64, int64)))
size = capacity // 2
for count in range(size):
    njit_heappush(array, tuple(np.random.randint(0, capacity, size=item_size)))
for i in range(iterations):
    if np.random.randint(0, 2) == 0 and size != 0 or size == capacity:
        njit_heappop(array)
        size -= 1
        assert len(array) == size
    else:
        njit_heappush(array, tuple(np.random.randint(0, capacity, size=item_size)))
        size += 1
        assert len(array) == size
output = [njit_heappop(array) for _ in range(size)]
assert sorted(output, key=lambda x: x) == output
print(len(output))

# %%
@njit
def test_njit_interspersed(capacity, array, size, iterations, item_size):
    for count in range(size):
        njit_heappush(array, tuple(np.random.randint(0, capacity, size=item_size)))
    for i in range(iterations):
        if np.random.randint(0, 2) == 0 and size != 0 or size == capacity:
            njit_heappop(array)
            size -= 1
        else:
            njit_heappush(array, tuple(np.random.randint(0, capacity, size=item_size)))
            size += 1
    output = []
    for count in range(size, 0, -1):
        output.append(njit_heappop(array))
    return output

# %%
capacity = 2_000
item_size = 20
iterations = capacity * 2_000
array = List.empty_list(Tuple((int64, int64, int64)))
size = capacity // 2
output = test_njit_interspersed(capacity, array, size, iterations, item_size)
assert sorted(output, key=lambda x: x) == output
print(len(output))

# %% [markdown]
# ### Testing heapq, original Python
# 

# %%
from heapq import *

python_heappush = heappush
python_heappop = heappop
python_heapreplace = heapreplace
python_heappushpop = heappushpop
python_heapify = heapify

# %% [markdown]
# ##### Heapify test
# 

# %%
size = 1_000_000
item_size = 20
array = []
for count in range(size):
    array.append(tuple(np.random.randint(0, capacity, size=item_size)))
python_heapify(array)
output = []
for count in range(size, 0, -1):
    output.append(python_heappop(array))
assert sorted(output, key=lambda x: x) == output

# %%
@njit
def test_python_heapify(array, size, item_size):
    for i in range(size):
        array.append(tuple(np.random.randint(0, capacity, size=item_size)))
    python_heapify(array)
    output = []
    for count in range(size, 0, -1):
        output.append(python_heappop(array))
    return output

# %%
size = 1_000_000
item_size = 20
array = List.empty_list(Tuple((int64, int64, int64)))
output = test_python_heapify(array, size, item_size)
assert sorted(output, key=lambda x: x) == output

# %% [markdown]
# ##### Heappush test
# 

# %%
size = 1_000_000
item_size = 20
array = []
for count in range(size):
    python_heappush(array, tuple(np.random.randint(0, capacity, size=item_size)))
output = []
for count in range(size, 0, -1):
    output.append(python_heappop(array))
assert sorted(output, key=lambda x: x) == output

# %%
@njit
def test_python_heappush(array, size, item_size):
    for count in range(size):
        python_heappush(array, tuple(np.random.randint(0, capacity, size=item_size)))
    output = []
    for count in range(size, 0, -1):
        output.append(njit_heappop(array))
    return output

# %%
size = 1_000_000
item_size = 20
array = List.empty_list(Tuple((int64, int64, int64)))
output = test_python_heappush(array, size, item_size)
assert sorted(output, key=lambda x: x) == output

# %% [markdown]
# ##### Interspersed push and pop tests
# 

# %%
capacity = 2_000
item_size = 20
iterations = capacity * 2_000
array = []
size = capacity // 2
for count in range(size):
    python_heappush(array, tuple(np.random.randint(0, capacity, size=item_size)))
for i in range(iterations):
    if np.random.randint(0, 2) == 0 and size != 0 or size == capacity:
        elem = python_heappop(array)
        size -= 1
        assert len(array) == size
    else:
        python_heappush(array, tuple(np.random.randint(0, capacity, size=item_size)))
        size += 1
        assert len(array) == size
output = []
for count in range(size, 0, -1):
    output.append(python_heappop(array))
assert sorted(output, key=lambda x: x) == output
print(len(output))

# %%
@njit
def test_python_interspersed(capacity, array, size, iterations, item_size):
    for count in range(size):
        python_heappush(array, tuple(np.random.randint(0, capacity, size=item_size)))
    for i in range(iterations):
        if np.random.randint(0, 2) == 0 and size != 0 or size == capacity:
            python_heappop(array)
            size -= 1
        else:
            python_heappush(array, tuple(np.random.randint(0, capacity, size=item_size)))
            size += 1
    output = []
    for count in range(size, 0, -1):
        output.append(python_heappop(array))
    return output

# %%
capacity = 2_000
item_size = 20
iterations = capacity * 2_000
array = List.empty_list(Tuple((int64, int64, int64)))
size = capacity // 2
output = test_python_interspersed(capacity, array, size, iterations, item_size)
assert sorted(output, key=lambda x: x) == output
print(len(output))