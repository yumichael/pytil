import math

import matplotlib.pyplot as plt
import numpy as np

try:
    from theano.tensor.var import _tensor_py_operators as _TPO
except ImportError:
    _TPO = type("_TPO", (), {})
from pytil.object import Namespace as O
from pytil.utility import *


def canvas(width, height):
    dpi = 96
    from pylab import rcParams

    rcParams['figure.figsize'] = width, height


def draw(x, info=None, tile=2):
    self = draw

    # plt.axis('off')
    class draw_flags(O()):
        cmap = plt.cm.binary
        aspect = 'equal'
        interpolation = 'none'

    if isinstance(x, _TPO):
        if hasattr(x, A.get_value):
            x = x.get_value()
        else:
            x = x.eval()
    sh = list(x.shape)
    while sh and sh[0] == 1:
        del sh[0]
    while sh and sh[-1] == 1:
        del sh[-1]
    if not sh:
        print_("Input to draw is trivial")
    x.reshape(sh)
    tile11 = 0
    if tile == 1:
        for i in range(3):
            x = np.expand_dims(x, axis=-1)
        tile11 = 1
    sh = x.shape
    rcdims = []
    for i in range(1, math.ceil(len(sh) / 2)):
        odd = len(sh) == 2 * i + 1
        rcdims.append(slice(len(sh) - 2 * i - 2 + odd, len(sh) - 2 * i))
    rcsh = [sh[d] for d in rcdims]
    tile = sh[-2:]

    if info is not None:
        assert 2 <= len(sh) <= 4
        datsh = sh[:-2]
        info = np.asarray(info).reshape(datsh)
        if len(datsh) == 2:
            pltsh = datsh
        elif len(datsh) == 1:
            nhor = math.ceil(math.sqrt(datsh[0]))
            nnil = (nhor - datsh[0] % nhor) % nhor
            nver = (datsh[0] + nnil) // nhor
            if nver == 1:
                nhor = datsh[0]
            pltsh = (nver, nhor)
        elif len(datsh) == 0:
            pltsh = (1, 1)
        f, axt = plt.subplots(*pltsh)
        datit = np.nditer(np.ones(datsh), flags=['multi_index'])
        pltit = np.nditer(np.ones(pltsh), flags=['multi_index'])
        while not datit.finished:
            sub = axt[pltit.multi_index]
            sub.set_title(info[datit.multi_index])
            sub.imshow(x[datit.multi_index], **draw_flags)
            sub.axis('off')
            datit.iternext()
            pltit.iternext()
        while not pltit.finished:
            sub = axt[pltit.multi_index]
            sub.axis('off')
            pltit.iternext()
        return None

    for n, (d, s) in enumerate(zip(rcdims, rcsh)):
        d = [d.start, d.stop]
        thick = 1 - (n == 0 and tile11)  # n + 1 - tile11
        if thick > 0:
            xsh = list(x.shape)
            xsh[d[1]] = thick
            x = np.concatenate([np.ones(xsh), x], axis=d[1])
            xsh = list(x.shape)
            xsh[d[1] + 1] = thick
            x = np.concatenate([x, np.ones(xsh)], axis=d[1] + 1)
        #
        xsh = list(x.shape)

        def SUBROUTINE(nhor, side=0):
            nonlocal x
            dside = d[0] + side
            nadd = (nhor - s[side] % nhor) % nhor
            nver = (s[side] + nadd) // nhor
            if nver == 1:
                nhor = s[side]
            else:
                xsh[dside] = nadd
                x = np.concatenate([x, np.ones(xsh)], axis=dside)
            xsh[dside] = nver
            xsh.insert(dside + 1, nhor)
            x = x.reshape(xsh)
            return nver

        if d[0] + 1 == d[1]:
            nhor = math.ceil(math.sqrt(s[0] * tile[0] / tile[1]))
            nver = SUBROUTINE(nhor)
            d[1] += 1
        else:  # d[0] + 2 == d[1]
            ratio0 = (s[0] * tile[0]) / (s[1] * tile[1])
            if ratio0 > self.aspect:
                nhor = math.ceil(math.sqrt(ratio0))
                nver = SUBROUTINE(nhor, 0)
                x = x.swapaxes(d[0] + 1, d[0] + 2)
                xsh[d[0] + 1] = s[1] * nhor
                del xsh[d[0] + 2]
                x = x.reshape(xsh)
            elif 1 / ratio0 > self.aspect:
                nhor = math.ceil(math.sqrt(s[0] * s[1] * tile[0] / tile[1]))
                nver = SUBROUTINE(nhor, 1)
                xsh[d[0]] = s[0] * nver
                del xsh[d[0] + 1]
                x = x.reshape(xsh)
        # weave the x y dimensions iteratively:
        dsh = list(range(len(xsh)))
        dsh[d[0] + 1], dsh[d[1]] = dsh[d[1]], dsh[d[0] + 1]
        x = x.transpose(*dsh)
        xsh = list(x.shape)
        xsh[d[0]] = xsh[d[0]] * xsh[d[0] + 1]
        xsh[d[0] + 1] = xsh[d[1]] * xsh[d[1] + 1]
        del xsh[d[1]]
        del xsh[d[1]]
        x = x.reshape(xsh)
        tile = xsh[-2:]
    plt.imshow(x, **draw_flags)
    plt.colorbar()


draw.aspect = 4


def view(image, *, cmap=None):
    self = view
    plt.figure(figsize=view.figsize)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')


view.figsize = (10, 10)
