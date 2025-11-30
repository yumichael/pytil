import random

import numba as nb
import numpy as np
from numba import njit


@njit(inline='always')
def stringify_1d_array(array):
    return '[' + ', '.join([str(entry) for entry in array]) + ']'


@njit
def set_seed_njit(seed):
    random.seed(seed)
    np.random.seed(seed)


def set_seed(seed):
    set_seed_njit(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(0)
