import theano.tensor as T
from theano import shared
from collections import OrderedDict
import numpy as np

__all__ = ['adagrad', 'sgd', 'sgd_momentum', 'adadelta']


def adagrad(cost_function, params, initial_learning_rate=0.1, fudge_factor=1e-6):
    """

    Parameters
    ----------
    cost_function
    params
    initial_learning_rate
    fudge_factor

    Returns
    -------

    """
    gparams = [T.grad(cost_function, param) for param in params]
    updates = OrderedDict()

    for param, grad in zip(params, gparams):
        value = param.get_value(borrow=True)
        accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (initial_learning_rate * grad / T.sqrt(accu_new + fudge_factor))

    return updates


def sgd(cost_function, params, learning_rate=0.1):
    """

    Parameters
    ----------
    cost_function
    params
    learning_rate

    Returns
    -------

    """
    gparams = [T.grad(cost_function, param) for param in params]
    updates = OrderedDict()

    for param, grad in zip(params, gparams):
        updates[param] = param - learning_rate * grad

    return updates


def sgd_momentum(cost_function, params, learning_rate=0.1, momentum_rate=0.9):
    """

    Parameters
    ----------
    cost_function
    params
    learning_rate
    momentum_rate

    Returns
    -------

    """
    gparams = [T.grad(cost_function, param) for param in params]
    updates = OrderedDict()

    for param, grad in zip(params, gparams):
        value = param.get_value(borrow=True)
        velocity = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        updates[param] = param - learning_rate * grad
        x = momentum_rate * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x

    return updates


def adadelta(cost_function, params, initial_learning_rate=0.1, rho=0.95, fudge_factor=1e-6):
    """

    Parameters
    ----------
    cost_function
    params
    initial_learning_rate
    rho
    fudge_factor

    Returns
    -------

    """
    gparams = [T.grad(cost_function, param) for param in params]
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, gparams):
        value = param.get_value(borrow=True)
        accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        delta_accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        update = (grad * T.sqrt(delta_accu + fudge_factor) / T.sqrt(accu_new + fudge_factor))
        updates[param] = param - initial_learning_rate * update
        delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates
