import theano
import theano.tensor as T
from theano.tensor.extra_ops import to_one_hot
import numpy as np

__all__ = ['translate_target', 'translate_output', 'get_index_label_classes']


def translate_target(_target, n_classes, target_labels):
    """ For each example you get a vector indicating the "index" from the vector labels class, this vector
    has all its elements in zero except the element of the position equals to "index" that it is 1.

    Parameters
    ----------
    _target : numpy.array
        Target sample.

    n_classes : int
        Number of classes.

    target_labels : list
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


def get_index_label_classes(_output):
    """ Gets index labels.

    Parameters
    ----------
    _output : theano.function
        Output of classifier model.

    Returns
    -------
    numpy.array[int]
        Returns index of labels from output model.
    """
    return T.argmax(_output, axis=1).eval()


def translate_output(_output, n_classes):
    """ Gets matrix with one hot encoding where the 1 represent index of class.

    Parameters
    ----------
    _output : theano.tensor.matrix
        Output sample.

    n_classes : int
        Number of classes (or size of one hot encoding rows)

    Returns
    -------
    theano.tensor.matrix
        Returns one hot encoding.
    """
    return to_one_hot(T.argmax(_output, axis=1), n_classes)
