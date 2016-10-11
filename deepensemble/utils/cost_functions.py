import theano.tensor as T
from .utils_functions import ITLFunctions

__all__ = ['mse', 'mcc', 'mee', 'neg_log_likelihood',
           'neg_corr', 'correntropy_ensemble', 'cross_entropy',
           'kullback_leibler', 'kullback_leibler_generalized',
           'test_cost', 'dummy_cost']

eps = 0.0001


# noinspection PyUnusedLocal
def dummy_cost(model, _input, _target):
    return T.zeros(_target.shape)


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


def mcc(model, _input, _target, s=None, kernel=ITLFunctions.norm):
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
    return -T.mean(kernel(model.error(_input, _target), s))


def mee(model, _input, _target, s=None, kernel=ITLFunctions.norm):
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
    labels = T.argmax(_target, axis=1)
    return -T.mean(T.log(model.output(_input))[T.arange(_target.shape[0]), labels])


############################################################################################################
#
# Cost Function only for Ensembles.
#
############################################################################################################


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
def correntropy_ensemble(model, _input, _target, ensemble, lamb_corr=0.5, s=None, kernel=ITLFunctions.norm):
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

    if s is None:
        s = T.max(ITLFunctions.silverman(_target, _target.shape[0], ensemble.get_dim_output()), eps)

    for model_j in ensemble.get_models():
        if model_j is not model:
            e = model_j.output(_input) - output_current_model
            error += [e]
    return T.mean(T.constant(lamb_corr) * kernel(T.concatenate(error), s))


def test_cost(model, _input, _target, ensemble, **kwargs):
    return test_4(model, _input, _target, ensemble, **kwargs)

# noinspection PyUnusedLocal
def test_4(_model, _input, _target, ensemble, s=None, kernel=ITLFunctions.norm):
    N = len(_model.get_params())
    params_model = [_model.get_params()[i] for i in range(0, N, 2)]

    sum_d = []
    for model_j in ensemble.get_models():
        params_model_j = [model_j.get_params()[i] for i in range(0, N, 2)]
        e = [i - j for i, j in zip(params_model, params_model_j)]
        for i, e_i in enumerate(e):
            s = ITLFunctions.silverman(params_model[i], N, 1)
            sum_d += [-ITLFunctions.information_potential(e_i, kernel, s)]

    return T.mean(sum(sum_d))

# noinspection PyUnusedLocal
def test_ensemble_ambiguity(_model, _input, _target, ensemble, **kwargs):
    _output = ensemble.output(_input)
    err = [T.mean(T.sqr(model.output(_input, prob=False) - _output)) for model in ensemble.get_models()]
    return -sum(err) / ensemble.get_num_models()

# noinspection PyUnusedLocal
def mee_ensemble2(model, _input, _target, ensemble, s=None, kernel=ITLFunctions.norm):
    om = model.output(_input)
    oe = ensemble.output(_input)

    if s is None:
        s = T.max(ITLFunctions.silverman(oe, _target.shape[0], model.get_dim_output()), eps)

    e = om - oe
    de = T.tile(e, (e.shape[0], 1, 1))
    de = de - T.transpose(de, axes=(1, 0, 2))
    return -T.log(T.mean(kernel(de, s)))

# noinspection PyUnusedLocal
def mee_ensemble(model, _input, _target, ensemble, s=None, kernel=ITLFunctions.norm):
    if s is None:
        s = T.max(ITLFunctions.silverman(_target, _target.shape[0], model.get_dim_output()), eps)

    e = model.error(_input, _target)
    de = T.tile(e, (e.shape[0], 1, 1))
    de = de - T.transpose(de, axes=(1, 0, 2))
    return T.log(T.mean(kernel(de, s)))


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


def test_cost_3(model, _input, _target, ensemble, lamb=1.0, s=0.17, kernel=ITLFunctions.norm):
    om = model.output(_input)
    em2 = T.power(model.error(_input, _target), 2.0)

    es = []
    for model_j in ensemble.get_models():
        if model_j is not model:
            omj = model_j.output(_input)
            emj = omj - om
            es.append(em2 * kernel(emj, s))
    return T.constant(-lamb) * T.mean(T.concatenate(es))


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
