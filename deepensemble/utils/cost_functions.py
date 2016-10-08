import theano.tensor as T
from .utils_functions import ITLFunctions

__all__ = ['mse', 'mcc', 'mee', 'neg_log_likelihood',
           'neg_corr', 'correntropy_cost', 'cross_entropy', 'correntropy_silverman_cost',
           'kullback_leibler', 'kullback_leibler_generalized',
           'test_cost']


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
    eps = 0.0001
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
    eps = 0.0001
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


def mcc(model, _input, _target, s, kernel=ITLFunctions.norm):
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
    return -T.mean(kernel(model.error(_input, _target), s))


def mee(model, _input, _target, s, kernel=ITLFunctions.norm):
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
    e = model.error(_input, _target)
    de = T.tile(e, (e.shape[0], 1, 1))
    de = de - T.transpose(de, axes=(1, 0, 2))
    return -T.log(T.mean(kernel(de, s)))


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
    return -T.mean(T.log(model.output(_input))[T.arange(_target.shape[0]), labels])


#
# Cost Function only for Ensembles.
#


# noinspection PyUnusedLocal
def neg_corr(model, _input, _target, ensemble, lamb_neg_corr=0.5):
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

    lamb_neg_corr : float, 0.5 by default
        Ratio negative correlation.

    Returns
    -------
    theano.tensor.matrix
        Return Negative Correlation.
    """
    return T.mean(T.constant(-lamb_neg_corr) * T.power(model.output(_input) - ensemble.output(_input), 2.0))


# noinspection PyUnusedLocal
def correntropy_cost(model, _input, _target, ensemble, lamb_corr=0.5, s=0.5, kernel=ITLFunctions.norm):
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

    lamb_corr : float, 0.5 by default
        Ratio negative correlation.

    s : float, 0.5 by default
        Size of Kernel.

    kernel : callable
        Kernel for compute correntropy.

    Returns
    -------
    theano.tensor.matrix
        Return Negative Correntropy.
    """
    error = []
    output_current_model = model.output(_input)
    for model_j in ensemble.get_models():
        if model_j is not model:
            e = model_j.output(_input) - output_current_model
            error += [e]
    return T.mean(T.constant(lamb_corr) * kernel(T.concatenate(error), s))


def correntropy_silverman_cost(model, _input, _target, ensemble, lamb_corr=0.5, kernel=ITLFunctions.norm):
    """ Compute the Correntropy regularization in Ensemble where update size kernel with Silverman.

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

    lamb_corr : float, 0.5 by default
        Ratio negative correlation.

    kernel : callable
        Kernel for compute correntropy.

    Returns
    -------
    theano.tensor.matrix
        Return Negative Correntropy.
    """
    eo = ensemble.output(_input)
    e = model.output(_input) - eo
    s = T.max(ITLFunctions.silverman(eo, _target.shape[0], ensemble.get_dim_output()), 0.00001)
    return T.mean(T.constant(lamb_corr) * kernel(e, s))


# noinspection PyUnusedLocal
def test_cost_2(model, _input, _target, ensemble, lamb=1.0, s=1.0):
    sum_mcc = 0.0
    output_current_model = model.output(_input)
    s = T.std(_target) * s
    for model_j in ensemble.get_models():
        if model_j is not model:
            e = model_j.output(_input) - output_current_model
            sum_mcc += T.mean(ITLFunctions.norm(e, s))
    return T.constant(lamb) * sum_mcc


def test_cost(model, _input, _target, ensemble, lamb=1.0, s=1.0):
    sum_mcc = 0.0
    output_current_model = model.output(_input)
    ee = T.power(ensemble.output(_input) - _target, 2.0)
    s = T.std(_target) * s
    for model_j in ensemble.get_models():
        if model_j is not model:
            e = T.power(model_j.output(_input) - output_current_model, 2.0)
            sum_mcc += T.mean(e * ee)
    return T.constant(-lamb) * sum_mcc / T.sum(ee)


# noinspection PyUnusedLocal
def test_cost_1(model, _input, _target, ensemble, lamb=10):
    params_model = [i for i in model.get_params()]
    sum_d = 0.0
    for model_j in ensemble.get_models():
        params_model_j = [i for i in model_j.get_params()]
        e = [i - j for i, j in zip(params_model, params_model_j)]
        for e_i in e:
            sum_d += T.power(T.sum(e_i), 2.0)

    return T.constant(-lamb) * T.exp(-T.mean(sum_d))


"""# error current model
e = model.output(_input) - ensemble.output(_input)
k = T.exp(-T.power(e, 2.0) / s)
return T.mean(-T.constant(lamb_corr) * k)"""

"""mse_ensemble = T.mean(T.power(ensemble.error(_input, _target), 2.0))
k = T.exp(-s * mse_ensemble)
cost = 0.0
output_current_model = model.output(_input)
for model_j in ensemble.list_models_ensemble:
    cost += output_current_model - model_j.output(_input)
return T.mean(T.constant(lamb_corr) * T.power(cost, 2.0) * k)"""

"""eps = 0.0001
pt = model.output(_input)
sum_d = 0.0
for model_j in ensemble.list_models_ensemble:
    pp = model_j.output(_input)
    sum_d += T.sum((pt + eps) * (T.log((pt + eps) / (pp + eps)) - pt + pp))
return sum_d / ensemble.get_num_models()"""
