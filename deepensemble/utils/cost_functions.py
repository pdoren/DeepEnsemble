import theano.tensor as T
from .utils_functions import ITLFunctions

__all__ = ['dummy_cost', 'mse', 'mcc', 'mee', 'neg_log_likelihood',
           'neg_corr', 'neg_correntropy', 'cross_entropy',
           'kullback_leibler', 'kullback_leibler_generalized',
           'itakura_saito']

eps = 0.00001


# noinspection PyUnusedLocal
def dummy_cost(model, _input, _target):
    """ Dummy cost function, this function only return zeros for each elements in _target.

    Parameters
    ----------
    model : Model
        Model.

    _input : theano.tensor.matrix
        Input Sample

    _target : theano.tensor.matrix
        Target Sample.

    Returns
    -------
    theano.tensor.matrix
        Returns only zeros for each elements in _target.
    """
    return T.zeros(_target.shape)


def information_potential(model, _input, _target, s=None, kernel=ITLFunctions.norm):
    """ Dummy cost function, this function only return zeros for each elements in _target.

    Parameters
    ----------
    model : Model
        Model.

    _input : theano.tensor.matrix
        Input Sample

    _target : theano.tensor.matrix
        Target Sample.

    s : float
        Size of kernel.

    kernel : theano.Op
        Kernel of compute information potential (Parzen Window).

    Returns
    -------
    theano.tensor.matrix
        Returns only zeros for each elements in _target.
    """
    if s is None:
        s = T.max(ITLFunctions.silverman(_target, _target.shape[0], model.get_dim_output()), eps)
    e = model.error(_input, _target)
    return -T.log(ITLFunctions.information_potential(e, kernel, s))


def itakura_saito(model, _input, _target):
    """ Itakura Saito distance.

    Parameters
    ----------
    model : Model
        Model for generating output for compare with target sample.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    Returns
    -------
    theano.tensor.matrix
        Return Itakura Saito divergence.
    """
    pt = _target
    pp = model.output(_input)
    return T.sum((pt + eps) / (pp + eps) - (T.log(pt + eps) - T.log(pp + eps)) - 1)


def kullback_leibler_generalized(model, _input, _target):
    """ Kullback Leilbler generalized divergence.

    Parameters
    ----------
    model : Model
        Model for generating output for compare with target sample.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    Returns
    -------
    theano.tensor.matrix
        Return Kullback Leilbler generalized divergence.
    """
    pt = _target
    pp = model.output(_input)
    return T.sum((pt + eps) * (T.log(pt + eps) - T.log(pp + eps)) - pt + pp)


def kullback_leibler(model, _input, _target):
    """ Kullback Leilbler divergence.

    Parameters
    ----------
    model : Model
        Model for generating output for compare with target sample.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    Returns
    -------
    theano.tensor.matrix
        Return Kullback Leilbler divergence.
    """
    pt = _target
    pp = model.output(_input)
    return T.sum((pt + eps) * (T.log(pt + eps) - T.log(pp + eps)))


def cross_entropy(model, _input, _target):
    """ Compute Cross Entropy between target and output prediction.

    Parameters
    ----------
    model : Model
        Model for generating output for compare with target sample.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    Returns
    -------
    theano.tensor.matrix
        Return Cross Entropy.
    """
    return T.nnet.categorical_crossentropy(model.output(_input), _target).mean()


def mse(model, _input, _target):
    """ Compute MSE error between target and output prediction.

    Parameters
    ----------
    model : Model
        Model for generating output for compare with target sample.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    Returns
    -------
    theano.tensor.matrix
        Return MSE error.
    """
    return T.mean(T.power(model.error(_input, _target), 2.0))


def mcc(model, _input, _target, s=None, kernel=ITLFunctions.kernel_gauss):
    """ Compute the MCC.

    Parameters
    ----------
    model : Model
        Model for generating output for compare with target sample.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    s : float
        Size of Kernel.

    kernel : callable
        Kernel for compute MCC.

    Returns
    -------
    theano.tensor.matrix
        Return MCC.
    """
    if s is None:
        s = T.max(ITLFunctions.silverman(_target, _target.shape[0], model.get_dim_output()), eps)
    return -T.mean(kernel(model.error(_input, _target, prob=True), s))


def mee(model, _input, _target, s=None, kernel=ITLFunctions.kernel_gauss):
    """ Compute the MEE.

    Parameters
    ----------
    model : Model
        Model for generating output for compare with target sample.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    s : float
        Size of Kernel.

    kernel : callable
        Kernel for compute MEE.

    Returns
    -------
    theano.tensor.matrix
        Return MEE.
    """
    if s is None:
        s = T.max(ITLFunctions.silverman(_target, _target.shape[0], model.get_dim_output()), eps)
    e = model.error(_input, _target)
    return -T.log(ITLFunctions.information_potential(e, kernel, s))


def neg_log_likelihood(model, _input, _target):
    """ Compute the negative means of errors between target and output prediction

    Parameters
    ----------
    model : Model
        Model for generating output for compare with target sample.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    Returns
    -------
    theano.tensor.matrix
        Return negative logarithm likelihood.
    """
    labels = T.argmax(_target, axis=-1)
    return -T.mean(T.log(model.output(_input))[T.arange(_target.shape[0]), labels])


############################################################################################################
#
# Cost Function only for Ensembles.
#
############################################################################################################


# noinspection PyUnusedLocal
def neg_corr(model, _input, _target, ensemble, lamb=0.5):
    """ Compute the Negative Correlation in Ensemble.

    Parameters
    ----------
    model : theano.tensor.matrix
        Current model that one would want to calculate the cost.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    ensemble : EnsembleModel
        Ensemble.

    lamb : float, 0.5 by default
        Ratio negative correlation.

    Returns
    -------
    theano.tensor.matrix
        Return Negative Correlation.
    """
    return T.mean(T.constant(-lamb) * T.power(model.output(_input) - ensemble.output(_input), 2.0))


# noinspection PyUnusedLocal
def neg_correntropy(model, _input, _target, ensemble, lamb=0.5, s=None, kernel=ITLFunctions.kernel_gauss):
    """ Compute the Correntropy regularization in Ensemble.

    Parameters
    ----------
    model : theano.tensor.matrix
        Current model that one would want to calculate the cost.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    ensemble : EnsembleModel
        Ensemble.

    lamb : float, 0.5 by default
        Ratio negative correlation.

    s : float
        Size of Kernel.

    kernel : callable
        Kernel for compute correntropy.

    Returns
    -------
    theano.tensor.matrix
        Return Negative Correntropy.
    """
    om = model.output(_input)
    oe = ensemble.output(_input)
    h = om - oe

    if s is None:
        s = T.max(ITLFunctions.silverman(oe, _target.shape[0], ensemble.get_dim_output()), eps)

    return T.mean(lamb * kernel(h, s))
