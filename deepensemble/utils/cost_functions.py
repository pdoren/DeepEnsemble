import theano.tensor as T
from theano import config
import numpy as np
from .utils_functions import ITLFunctions

__all__ = ['dummy_cost', 'mse', 'mcc', 'mee', 'neg_log_likelihood',
           'neg_corr', 'neg_correntropy', 'cross_entropy',
           'kullback_leibler', 'kullback_leibler_generalized',
           'itakura_saito', 'neg_mee', 'neg_klg', 'cauchy_schwarz_divergence',
           'cip_relevancy', 'cip_redundancy', 'cip_synergy', 'cip_full']

eps = 0.00001
sqrt2 = 1.41421356237


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


def cauchy_schwarz_divergence(model, _input, _target):
    """ Cauchy Schwarz divergence.

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
        Return Cauchy Schwarz divergence.
    """
    pt = _target
    pp = model.output(_input)

    return -2 * T.log(T.sum(pp * pt) + eps) + T.log(T.sum(pp * pp) + eps) + T.log(T.sum(pt * pt) + eps)


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
    return -T.log(ITLFunctions.information_potential(e, kernel, sqrt2 * s))


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


def cip_relevancy(model, _input, _target, s=None, dist='CS'):
    """ Cross Information Potential between model output and target.

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

    dist : str
        This string means if the CIP is compute with Euclidean or Cauchy-Schwarz divergence.

    Returns
    -------
    theano.tensor.matrix
        Return Cross Information Potential between model output and target.
    """
    kernel = ITLFunctions.kernel_gauss

    om = model.output(_input)
    if s is None:
        s = T.max(ITLFunctions.silverman(_target, _target.shape[0], model.get_dim_output()), eps)

    if dist == 'CS':
        return T.log(ITLFunctions.cross_information_potential([om, _target], kernel, sqrt2 * s))
    elif dist == 'CIP':
        return ITLFunctions.cross_information_potential([om, _target], kernel, sqrt2 * s)
    elif dist == 'ED':
        return -ITLFunctions.mutual_information_ed([om, _target], kernel, sqrt2 * s)
    else:
        raise ValueError('the distance must be CS or ED.')


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


# noinspection PyUnusedLocal
def neg_mee(model, _input, _target, ensemble, lamb=0.5, s=None, kernel=ITLFunctions.kernel_gauss):
    """ Compute the MEE regularization in Ensemble.

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
    om = model.error(_input, _target)
    oe = ensemble.error(_input, _target)
    h = om - oe

    if s is None:
        s = T.max(ITLFunctions.silverman(_target, _target.shape[0], model.get_dim_output()), eps)

    return lamb * T.log(ITLFunctions.information_potential(h, kernel, sqrt2 * s))


def neg_klg(model, _input, _target, ensemble, lamb=0.5):
    """

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
        Return Negative KLG.
    """
    pt = ensemble.error(_input, _target)
    pp = model.error(_input, _target)
    return -lamb * T.sum((pt + eps) * (T.log(pt + eps) - T.log(pp + eps)) - pt + pp)


def cip_redundancy(model, _input, _target, ensemble, beta=0.9, s=None, dist='CS'):
    """ Cross Information Potential Diversity.

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

    beta : float
        Ratio.

    s : float
        Size of Kernel.

    dist : str
        This string means if the CIP is compute with Euclidean or Cauchy-Schwarz divergence.

    Returns
    -------
    theano.tensor.matrix or float
        Return Cross Information Potential Diversity.
    """
    kernel = ITLFunctions.kernel_gauss

    if s is None:
        s = sqrt2 * T.max(ITLFunctions.silverman(_target, _target.shape[0], model.get_dim_output()), eps)

    redundancy = []
    om = model.output(_input)
    not_arrive = True
    for _model in ensemble.get_models():
        # jump until get current model
        if not_arrive and _model is not model:
            continue
        else:
            not_arrive = False

        # compute relevancy from next model
        if _model is not model:
            om_k = _model.output(_input)
            if dist == 'CS':
                cip2 = ITLFunctions.cross_information_potential([om, om_k], kernel, s)
                redundancy.append(T.log(cip2))
            elif dist == 'CIP':
                cip2 = ITLFunctions.cross_information_potential([om, om_k], kernel, s)
                redundancy.append(cip2)
            elif dist == 'ED':
                I2 = ITLFunctions.mutual_information_ed([om, om_k], kernel, s)
                redundancy.append(I2)
            else:
                raise ValueError('the distance must be CS or ED.')

    if len(redundancy) > 0:
        return beta * np.sum(redundancy)
    else:
        return T.constant(0.0, dtype=config.floatX)


def cip_synergy(model, _input, _target, ensemble, lamb=0.9, s=None, dist='CS'):
    """ Cross Information Potential Synergy.

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

    lamb : float
        Ratio.

    s : float
        Size of Kernel.

    dist : str
        This string means if the CIP is compute with Euclidean or Cauchy-Schwarz divergence.

    Returns
    -------
    theano.tensor.matrix
        Return Cross Information Potential Diversity.
    """
    kernel = ITLFunctions.kernel_gauss

    if s is None:
        s = sqrt2 * T.max(ITLFunctions.silverman(_target, _target.shape[0], model.get_dim_output()), eps)

    synergy = []
    om = model.output(_input)
    not_arrive = True
    for _model in ensemble.get_models():
        # jump until get current model
        if not_arrive and _model is not model:
            continue
        else:
            not_arrive = False

        # compute relevancy from next model
        if _model is not model:
            om_k = _model.output(_input)
            if dist == 'CS':
                cip2 = ITLFunctions.cross_information_potential([om, om_k], kernel, s)
                cip3 = ITLFunctions.cross_information_potential([om, om_k, _target], kernel, s)
                synergy.append(T.log(cip2) - T.log(cip3))
            elif dist == 'CIP':
                cip3 = ITLFunctions.cross_information_potential([om, om_k, _target], kernel, s)
                synergy.append(-cip3)
            elif dist == 'ED':
                I2 = ITLFunctions.mutual_information_ed([om, om_k], kernel, s)
                I3 = ITLFunctions.mutual_information_ed([om, om_k, _target], kernel, s)
                synergy.append(I2 - I3)
            else:
                raise ValueError('the distance must be CS or ED.')

    if len(synergy) > 0:
        return lamb * np.sum(synergy)
    else:
        return T.constant(0.0, dtype=config.floatX)


def cip_full(model, _input, _target, s=None):
    """ Cross Information Potential among all models ensemble.

    Parameters
    ----------
    model : theano.tensor.matrix
        Ensemble model.

    _input : theano.tensor.matrix
        Input sample.

    _target : theano.tensor.matrix
        Target sample.

    s : float
        Size of Kernel.

    Returns
    -------
    theano.tensor.matrix
        Return Cross Information Potential.
    """
    kernel = ITLFunctions.kernel_gauss

    if s is None:
        s = T.max(ITLFunctions.silverman(_target, _target.shape[0], model.get_dim_output()), eps)

    Y = [_model.output(_input) for _model in model.get_models()]
    Y.append(_target)

    return ITLFunctions.cross_information_potential(Y, kernel, sqrt2 * s)
