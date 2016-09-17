import theano.tensor as T
from ..models.model import Model

__all__ = ['mse', 'mcc', 'mee', 'neg_log_likelihood', 'neg_corr', 'corrpy_cost']


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
    theano.config.floatX
        Return MSE error.
    """
    e = model.output(_input) - _target
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
    theano.config.floatX
        Return MCC.
    """
    e = model.output(_input) - _target
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
    theano.config.floatX
        Return MEE.
    """
    e = model.output(_input) - _target
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
    theano.config.floatX
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
    theano.config.floatX
        Return Negative Correlation.
    """

    e = ensemble.list_models_ensemble[index_current_model].output(_input) \
        - ensemble.output(_input)  # error current model
    sum_ee = 0.0  # error sum of other models
    for i, model in enumerate(ensemble.list_models_ensemble):
        if i != index_current_model:
            sum_ee += model.output(_input) - ensemble.output(_input)
    return T.mean(T.constant(lamb_neg_corr) * e * sum_ee)


# noinspection PyUnusedLocal
def corrpy_cost(model, _input, _target, index_current_model, ensemble, lamb_corr=0.5, s=0.5):
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
    theano.config.floatX
        Return Negative Correlation.
    """

    # error current model
    # e = ensemble.list_models_ensemble[index_current_model].output(_input) - ensemble.output(_input)
    co = ensemble.list_models_ensemble[index_current_model].output(_input)
    eo = ensemble.output(_input)
    e = ensemble.output(_input) - _target
    sum_ee = 0.0  # error sum of other models
    for i, model in enumerate(ensemble.list_models_ensemble):
        if i != index_current_model:
            mo = model.output(_input)
            ec = (co - mo)
            sum_ee += T.power(ec, 2.0)
    return T.mean(e * sum_ee) * lamb_corr
