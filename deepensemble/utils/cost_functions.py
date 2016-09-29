import theano.tensor as T

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
    e = model.error(_input, _target)
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
    sum_err = 0.0
    output_current_model = model.output(_input)
    output_ensemble = ensemble.output(_input)
    for model_j in ensemble.get_models():
        if model_j != model:
            sum_err += model_j.output(_input) - output_ensemble
    return T.mean(T.constant(-lamb_neg_corr) * (output_current_model - output_ensemble) * sum_err)


# noinspection PyUnusedLocal
def correntropy_cost(model, _input, _target, ensemble, lamb_corr=0.5, s=0.5):
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
        Ratio kernel.

    Returns
    -------
    theano.tensor.matrix
        Return Negative Correntropy.
    """
    sum_err = 0.0
    output_current_model = model.output(_input)
    sqrt2pi = T.constant(2.50662827)  # sqrt(2 * pi)
    for model_j in ensemble.get_models():
        if model_j != model:
            e = model_j.output(_input) - output_current_model
            sum_err += T.exp(- T.power(e, 2.0) / (T.constant(2.0) * T.power(s, 2.0))) / (sqrt2pi * s)
    return T.mean(T.constant(-lamb_corr) * sum_err)


def correntropy_silverman_cost(model, _input, _target, ensemble, lamb_corr=0.5):
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

    Returns
    -------
    theano.tensor.matrix
        Return Negative Correntropy.
    """
    d = T.constant(ensemble.n_output)
    N = T.sum(T.ones_like(_target)) / d
    e = model.output(_input) - ensemble.output(_input)
    k = T.power(T.constant(4.0) / (N * (T.constant(2.0) * d + T.constant(1.0))), T.constant(1.0) / d + T.constant(4.0))
    s = T.std(_target) * k
    sqrt2pi = T.constant(2.50662827)  # sqrt(2 * pi)
    return T.mean(
        T.constant(-lamb_corr) * T.exp(- T.power(e, 2.0) / (T.constant(2.0) * T.power(s, 2.0))) / (sqrt2pi * s))


def test_cost(model, _input, _target, ensemble, lamb=10):
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
