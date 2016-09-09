import theano
import numpy as np

__all__ = ['translate_target']


def translate_target(_target, n_classes, target_labels):
    """ For each example you get a vector indicating the "index" from the vector labels class, this vector
    has all its elements in zero except the element of the position equals to "index" that it is 1.

    Parameters
    ----------
    _target: numpy.array
        Target sample.

    n_classes: int
        Number of classes.

    target_labels: list
        Target labels.

    Returns
    -------
    numpy.array
        Returns the '_target' translated according to target labels.
    """
    target = np.zeros(shape=(len(_target), n_classes), dtype=theano.config.floatX)
    for i, label in enumerate(_target):
        target[i, list(target_labels).index(label)] = 1.0
    return target
