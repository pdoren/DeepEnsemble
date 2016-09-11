import theano.tensor as T
from theano import shared
from collections import OrderedDict
import numpy as np

# Thanks to the library Lasagne.

__all__ = ['adagrad', 'sgd', 'sgd_momentum', 'adadelta']


def adagrad(cost_function, params, initial_learning_rate=0.1, epsilon=1e-6):
    """ Adagrad updates.

    Parameters
    ----------
    cost_function : theano.function
        Function cost.

    params : theano.share
        List of params model.

    initial_learning_rate : float, 0.1 by default
        The learning rate controlling the size of update steps.

    epsilon : float, 1e-6 by default
        Small value added for numerical stability.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression.

    References
    ----------
    .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
           Adaptive subgradient methods for online learning and stochastic
           optimization. JMLR, 12:2121-2159.

    .. [2] Chris Dyer:
           Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
    """
    gparams = [T.grad(cost_function, param) for param in params]
    updates = OrderedDict()

    for param, grad in zip(params, gparams):
        value = param.get_value(borrow=True)
        accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (initial_learning_rate * grad / T.sqrt(accu_new + epsilon))

    return updates


def sgd(cost_function, params, learning_rate=0.1):
    """ Stochastic Gradient Descent (SGD).

    Parameters
    ----------
    cost_function : theano.function
        Function cost.

    params : theano.share
        List of params model.

    learning_rate : float, 0.1 by default
        The learning rate controlling the size of update steps.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression.
    """
    gparams = [T.grad(cost_function, param) for param in params]
    updates = OrderedDict()

    for param, grad in zip(params, gparams):
        updates[param] = param - learning_rate * grad

    return updates


def sgd_momentum(cost_function, params, learning_rate=0.1, momentum_rate=0.9):
    """ Stochastic Gradient Descent (SGD) updates with momentum.

    Parameters
    ----------
    cost_function : theano.function
        Function cost.

    params : theano.share
        List of params model.

    learning_rate : float, 0.1 by default
        The learning rate controlling the size of update steps.

    momentum_rate : float, 0.9 by default
        The Momentum rate smoothing over more update steps.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression.
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
    """ Adadelta updates

    Parameters
    ----------
    cost_function : theano.function
        Function cost.

    params : theano.share
        List of params model.

    initial_learning_rate : float, 0.1 by default
        The learning rate controlling the size of update steps.

    rho : float, 0.95 by default
        Squared gradient moving average decay factor.

    fudge_factor : float, 1e-6 by default
        Small value added for numerical stability.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression.

    References
    ----------
    .. [1] Zeiler, M. D. (2012):
           ADADELTA: An Adaptive Learning Rate Method.
           arXiv Preprint arXiv:1212.5701.
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
