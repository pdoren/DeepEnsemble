import numpy as np
import theano.tensor as T

from deepensemble.utils.utils_functions import ITLFunctions


# noinspection PyUnusedLocal
def dVJ(Y, dydW, s):
    G = ITLFunctions.kernel_gauss
    DY = []
    for y in Y:
        dy = T.tile(y, (y.shape[0], 1, 1))
        dy = T.transpose(dy, axes=(1, 0, 2)) - dy
        DY.append(dy)

    # noinspection PyArgumentList
    dG = [T.mean((G(dy, T.sqrt(2) * s) / s ** 2) * dy * dydW) for dy in DY]

    return np.sum(dG)


# noinspection PyUnusedLocal
def dVM(X, Y, WX, WY, s):
    G = ITLFunctions.kernel_gauss
    DY = []
    for y in [X, Y]:
        dy = T.tile(y, (y.shape[0], 1, 1))
        dy = T.transpose(dy, axes=(1, 0, 2)) - dy
        DY.append(dy)

    # noinspection PyArgumentList
    dG = [T.mean((G(dy, T.sqrt(2) * s) / s ** 2) * dy) for dy in DY]

    return np.sum(dG)
