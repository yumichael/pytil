from numba import njit


__all__ = ['array_heappush', 'array_heappop', 'array_heapify', 'array_heapreplace', 'array_heappushpop', 'array_is_less']

@njit
def is_less(a, b):
    m, n = len(a), len(b)
    assert m == n
    for i in range(m):
        if a[i] < b[i]:
            return True
    return False


@njit
def heappush(heap, heap_size_holder, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap_size = heap_size_holder[0]
    heap[heap_size] = item
    heap_size += 1
    heap_size_holder[0] = heap_size
    _siftdown(heap, 0, heap_size - 1)


@njit
def heappop(heap, heap_size_holder):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    heap_size = heap_size_holder[0]
    lastelt = heap[heap_size - 1].copy()  # raises appropriate IndexError if heap is empty
    heap_size -= 1
    heap_size_holder[0] = heap_size
    if heap_size:
        returnitem = heap[0].copy()
        heap[0] = lastelt
        _siftup(heap, 0, heap_size)
        return returnitem
    return lastelt


@njit
def heapreplace(heap, heap_size, item):
    """Pop and return the current smallest value, and add the new item.

    This is more efficient than heappop() followed by heappush(), and can be
    more appropriate when using a fixed-size heap.  Note that the value
    returned may be larger than item!  That constrains reasonable uses of
    this routine unless written as part of a conditional replacement:

        if item > heap[0]:
            item = heapreplace(heap, item)
    """
    returnitem = heap[0].copy()  # raises appropriate IndexError if heap is empty
    heap[0] = item
    _siftup(heap, 0, heap_size)
    return returnitem


@njit
def heappushpop(heap, heap_size, item):
    """Fast version of a heappush followed by a heappop."""
    if heap and heap[0] < item:
        item, heap[0] = heap[0], item
        _siftup(heap, 0, heap_size)
    return item


@njit
def heapify(x, heap_size_holder):
    """Transform list into a heap, in-place, in O(len(x)) time."""
    n = heap_size_holder[0]
    # Transform bottom-up.  The largest index there's any point to looking at
    # is the largest with a child index in-range, so must have 2*i + 1 < n,
    # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
    # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
    # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
    for i in range(n // 2 - 1, -1, -1):
        _siftup(x, i, n)


# 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
# is the index of a leaf with a possibly out-of-order value.  Restore the
# heap invariant.
@njit
def _siftdown(heap, startpos, pos):
    newitem = heap[pos].copy()
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if is_less(newitem, parent):
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


# The child indices of heap index pos are already heaps, and we want to make
# a heap at index pos too.  We do this by bubbling the smaller child of
# pos up (and so on with that child's children, etc) until hitting a leaf,
# then using _siftdown to move the oddball originally at index pos into place.
@njit
def _siftup(heap, pos, endpos):
    startpos = pos
    newitem = heap[pos].copy()
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2 * pos + 1  # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not is_less(heap[childpos], heap[rightpos]):
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)


array_is_less = is_less
array_heappush = heappush
array_heappop = heappop
array_heapreplace = heapreplace
array_heappushpop = heappushpop
array_heapify = heapify
