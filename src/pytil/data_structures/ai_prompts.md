# Workspace for drafts of AI prompts

### Nearest Neighbor Map

I want you to write a data structure class for me. It should be in Python, using Numba's njit environment as a jitclass. I will call the data structure class NearestNeighborMap. Let's specify the API.

```python
from collections.abc import Sequence
from functools import cache

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numpy.typing import NDArray

@cache
def get_nearest_neighbor_map_jitclass(count_type, coordinate_type, key_type):
    count_type_numba = nb.from_dtype(count_type) # This is the the go to type for integer indices that you might need to use
    coordinate_type_numba = nb.from_dtype(coordinate_type)
    key_type_numba = nb.from_dtype(key_type)
    nearest_neighbor_map_spec = (
        # fill in here
    )

    @jitclass(nearest_neighbor_map_spec)
    class NearestNeighborMap:
        def __init__(self, max_size: int, max_key_size: int):
            '''Initialize the data structure allowing for at most max_size number of points to be inserted. Parameter max_key_size is a bound for the key size that can be used to allocate an array for the mapping of keys.'''

        def __setitem__(self, key: key_type, point: Sequence[coordinate_type]):
            '''Store a 3-D point inside the data structure labelled with key.'''

        def __delitem__(self, key: key_type):
            '''Remove the point associated with key.'''

        def get_closest_points(self, reference_point: Sequence[coordinate_type], keys_buffer: NDArray[key_type]) -> int:
            '''Find the closest points (include all ties) to the reference_point in the data structure and place the corresponding keys in the keys_buffer starting at index 0 going up. Return the number of closest points placed in the buffer.'''

        def __len__(self) -> int:
            '''Return how many points there are in the data structure.'''

    return NearestNeighborMap
```

You should preinitialize all memory needed in `__init__` to save allocation time. I need the main operations to be sublinear time because I am going to be running the operations in a very long loop. So please optimize the speed as much as possible. Bonus if you can make the data structure support an arbitrary number of dimensions for the points, but don't worry if you can't do that without sacrificing speed.

Can you make a thorough test + benchmarking module for this data structure? You can use the file I am uploading here for reference on how thorough the testing needs to be: you should let the class go through a slew of various operations and test against a simple oracle implementation to ensure correctness. You can also implement your own ideas for checks and also benchmark tests.

#### Incremental updates

Some changes I made, including but possibly not only:

- Got rid of integer conversion on the key.
- Renamed get_closest_points to get_closest_points_assign.
- Raise IndexError when the number of closest points exceed the given buffer size.
- Renamed the data structure.

I have uploaded the changed file, please make edits on top of my changes.

Now, I want you to pay attention to efficiency. I believe that allocating numpy arrays during updates and queries are inefficient. Determine if this is really tha case. If it is true, then please find a way to remove all array allocations outside of `__init__`. Pass a new parameter `dimension_count` to `__init__` so you don't have to wait for the first inserted point to tell how many dimensions the points have. Give me the code difference in diff form so I can take a quick look.

Made more changes:

- Now I raise KeyError if trying to delete a key not already in the data structure.
- Got rid if dimension_count argument in the factory function since it's not needed there.

Ok I found the problem. It's because **delitem** isn't supported in this version of numba. I have replaced **delitem** with a method named remove. Please update on top of the changes I made in the files I am attaching now. Now, can you update the test module so that it is aligned with my remove behavior that raises a KeyError if the key is not in the structure.

Keep in mind. After testing it looks like Numba ignores the type hints. So type hints are NOT the problem. Don't try to change the type hints. Now, can you output the entire contents of the test module please instead of using diff syntax?

I made a change to the test module. It turns out numba doesn't allow catching specific exceptions. So I commented out the specific exceptions as just notes.

I have made some changes that I want to keep to the test module. I have uploaded the changes in this file. Please make your changes on top of mine.
The oracle_closest_points function scans the entire MAX_KEY_SIZE sized key space. This is unacceptable as I need to test a MAX_KEY_SIZE of 2^24. Please change the implementation of the oracle_closest_points to be faster, for example by keeping track of the points on the oracle side using a hash table.
Another thing, because I have such a large key space, it's unlikely if we are deleting random keys we will hit a key we actually already added. So I would like you to test only deleting existing keys. You might need to create another data structure to be able to select a random existing key, such as an array, in addition to the hash table earlier.
Another thing, you don't have to assign point coordinates in a loop over the dimensions. Numba is fine with you assigning a tuple to a Numpy array slice, as long as the sizes are the same.
Please give me the diff of your changes first so I can easily verify.

Let's revisit the AoS vs SoA. By AoS, I mean make the relevant array variables a single 2-D array intsead, and index into the second dimension with CONSTANTS to find which of the different values we want. I'm not familiar with the inner workings of the code, but isn't this at least possible to do with tree_left and tree_right? Maybe you can add tree_axis and tree_valid into the 2-D array as well. I don't know, maybe even you can add tree_keys if the logic makes sense, although you would have to unify the count_type with the key_type (which is fine for me by the way).

##########################

Can you make the following changes:

- Rename the concept of a key to that of a label everywhere
- Rename `get_closest_points_assign` to `get_all_nearest_labels_assign`
- Add this method:

```python
def get_nearest(reference_point: Sequence[coordinate_type]) -> tuple[Sequence[coordinate_type], label_type]:
    '''Return any one nearest point and its label as measured from the reference point.'''
```

- Reorder the parameters of `_build_tree_recursive_njit` as `_build_tree_recursive_njit(points, tree_data, indices, depth, dim, count_type, coordinate_type)`
- Rename `get_kd_tree_with_keys_jitclass` to `get_kd_tree_with_labeled_points_jitclass`
- Reorder the parameters of `get_kd_tree_with_labeled_points_jitclass` as `get_kd_tree_with_labeled_points_jitclass(count_type, coordinate_type, label_type)`
- Rename `nearest_neighbor_map_spec` to `kd_tree_with_labeled_points_spec`
- Rename `KdTreeWithKeys` to `KdTreeWithLabeledPoints`
- Note that the file should be renamed `kd_tree_with_labeled_points.py`
- Also let me know if you see anything else that should be changed to keep things consistent with the new changes I am asking for, and go ahead and make the adjustments you came up with.

#########################

Can you make the following changes:

- The concept "nearest neighbor map" should be renamed to "nearest neighbor index"
- The concept of "key" should be renamed to "label"
- The API `get_closest_points_assign` has been changed to `get_all_nearest_labels_assign`
- Support testing a new API in the nearest neighbor index data structure:

```python
def get_nearest(reference_point: Sequence[coordinate_type]) -> tuple[Sequence[coordinate_type], label_type]:
    '''Return any one nearest point and its label as measured from the reference point.'''
```

- Put `get_nearest` just before the `get_all_nearest_labels_assign` in the op flags list
- The pick ops weights should be `[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]` after the changes
- Note the file should be renamed `test_nearest_neighbor_index.py`
- Also let me know if you see anything else that should be changed to keep things consistent with the new changes I am asking for, and go ahead and make the adjustments you came up with.
