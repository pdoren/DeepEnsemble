import theano
import theano.tensor as T
from theano.tensor.extra_ops import to_one_hot
import numpy as np

__all__ = ['translate_target', 'translate_binary_target', 'translate_output', 'get_index_label_classes']


def translate_target(_target, target_labels):
    """ For each example you get a vector indicating the "index" from the vector labels class, this vector
    has all its elements in zero except the element of the position equals to "index" that it is 1.

    Parameters
    ----------
    _target : numpy.array
        Target sample.

    target_labels : list
        Target labels.

    Returns
    -------
    numpy.array
        Returns the '_target' translated according to target labels.
    """
    n_classes = len(target_labels)
    target = np.zeros(shape=(len(_target), n_classes), dtype=theano.config.floatX)
    for i, label in enumerate(_target):
        target[i, list(target_labels).index(label)] = 1.0
    return target


def translate_binary_target(_target, target_labels):
    """ Gets a vector with binary classes: +1 or -1.

    Parameters
    ----------
    _target : numpy.array
        Target sample.

    target_labels : list
        Target labels.

    Returns
    -------
    numpy.array
        Returns the '_target' translated according to target labels, also each class is encoding as: +1 and -1.
    """
    target = np.zeros(shape=(len(_target), 1), dtype=theano.config.floatX)
    dict_value = [0] * 2
    dict_value[0] = -1
    dict_value[1] = +1

    for i, label in enumerate(_target):
        target[i] = dict_value[list(target_labels).index(label)]
    return target


def get_index_label_classes(_output, is_binary_classification=False):
    """ Gets index labels.

    Parameters
    ----------
    _output : theano.function
        Output of classifier model.

    is_binary_classification : bool
        This flag means that model is for binary classification.

    Returns
    -------
    numpy.array[int]
        Returns index of labels from output model.
    """
    if is_binary_classification:
        return T.ge(_output, 0.0).eval()
    else:
        return T.argmax(_output, axis=1).eval()


def translate_output(_output, n_classes, is_binary_classification=False):
    """ Gets matrix with one hot encoding where the 1 represent index of class.

    Parameters
    ----------
    _output : theano.tensor.matrix
        Output sample.

    n_classes : int
        Number of classes (or size of one hot encoding rows)

    is_binary_classification : bool
        This flag means that model is for binary classification.

    Returns
    -------
    theano.tensor.matrix
        Returns one hot encoding.
    """
    if is_binary_classification:
        return T.sgn(_output)
    else:
        return to_one_hot(T.argmax(_output, axis=1), n_classes)
