import theano.tensor as T
from ..models.model import Model

__all__ = ['mse', 'mcc', 'mee', 'neg_log_likelihood', 'neg_corr', 'correntropy_cost', 'cross_entropy']


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
    e = model.error(_input, _target)
    return T.mean(T.power(e, 2.0))


def mcc(model, _input, _target, s):
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
        Kernel's Parameter.

    Returns
    -------
    theano.tensor.matrix
        Return MCC.
    """
    e = model.error(_input, _target)
    return -T.mean(T.exp(-0.5 * T.power(e, 2.0) / s ** 2))


def mee(model, _input, _target, s):
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
        Kernel's Parameter.

    Returns
    -------
    theano.tensor.matrix
        Return MEE.
    """
    e = model.error(_input,  _target)
    de = T.tile(e, (e.shape[0], 1, 1))
    de = de - T.transpose(de, axes=(1, 0, 2))
    return -T.log(T.mean(T.exp(-0.5 * T.power(de, 2.0) / s ** 2)))


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
    labels = T.argmax(_target, axis=1)
    return -T.mean(T.log(T.power(model.output(_input), 2.0))[T.arange(_target.shape[0]), labels])

#
# Cost Function only for Ensembles.
#


# noinspection PyUnusedLocal
def neg_corr(model, _input, _target, index_current_model, ensemble, lamb_neg_corr=0.5):
    """ Compute the Negative Correlation in Ensemble.

    Parameters
    ----------
    model : theano.tensor.matrix
        Model.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    index_current_model : int
        Index of current model in ensemble.

    ensemble : EnsembleModel
        Ensemble.

    lamb_neg_corr : float, 0.5 by default
        Ratio negative correlation.

    Returns
    -------
    theano.tensor.matrix
        Return Negative Correlation.
    """
    return T.mean(-T.constant(lamb_neg_corr) * T.power(model.output(_input) - ensemble.output(_input), 2.0))


# noinspection PyUnusedLocal
def correntropy_cost(model, _input, _target, index_current_model, ensemble, lamb_corr=0.5, s=0.5):
    """ Compute the Correntropy regularization in Ensemble.

    Parameters
    ----------
    model : theano.tensor.matrix
        Model.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    index_current_model : int
        Index of current model in ensemble.

    ensemble : EnsembleModel
        Ensemble.

    lamb_corr : float, 0.5 by default
        Ratio negative correlation.

    s : float, 0.5 by default
        Ratio kernel.

    Returns
    -------
    theano.tensor.matrix
        Return Negative Correlation.
    """

    # error current model
    e = model.output(_input) - ensemble.output(_input)
    k = T.exp(- T.power(e, 2.0) / s)
    return T.mean(-T.constant(lamb_corr) * k)
