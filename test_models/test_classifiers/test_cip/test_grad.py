import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from theano import shared
import theano.tensor as T
from collections import OrderedDict

from deepensemble.utils import load_data, plot_pdf, load_data_segment
from deepensemble.utils.utils_classifiers import get_index_label_classes, translate_target
from deepensemble.utils.utils_functions import ActivationFunctions, ITLFunctions
from deepensemble.utils.utils_models import get_ensembleCIP_model
from deepensemble.utils.update_functions import sgd_cip

def dVJ(Y, dydW, s):
    G = ITLFunctions.kernel_gauss
    DY = []
    for y in Y:
        dy = T.tile(y, (y.shape[0], 1, 1))
        dy = T.transpose(dy, axes=(1, 0, 2)) - dy
        DY.append(dy)

    dG = [T.mean((G(dy, T.sqrt(2) * s) / s ** 2) * dy * dydw) for dy in DY]

    return np.sum(dG)


def dVM(X, Y, WX, WY, s):
    G = ITLFunctions.kernel_gauss
    DY = []
    for y in [X, Y]:
        dy = T.tile(y, (y.shape[0], 1, 1))
        dy = T.transpose(dy, axes=(1, 0, 2)) - dy
        DY.append(dy)

    dG = [T.mean((G(dy, T.sqrt(2) * s) / s ** 2) * dy) for dy in DY]

    return np.sum(dG)