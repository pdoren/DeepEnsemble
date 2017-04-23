import theano.tensor as T
from theano import shared
from collections import OrderedDict
import numpy as np

# Thanks to the library Lasagne.

__all__ = ['dummy_update', 'adagrad', 'sgd', 'sgd_momentum', 'adadelta', 'rmsprop', 'adam', 'sgd_cip', 'count_epoch']


# noinspection PyUnusedLocal,PyUnusedLocal
def dummy_update(cost_function, params):
    """ Dummy update function.

    Parameters
    ----------
    cost_function : callable
        Cost function.

    params : dict
        Parameters of cost function.

    Returns
    -------
    OrderedDict
        Returns an empty dictionary.
    """
    return OrderedDict()


def adagrad(cost_function, params, learning_rate=0.1, epsilon=1e-6):
    """ Adagrad updates.

    Parameters
    ----------
    cost_function : theano.function
        Function cost.

    params : theano.share
        List of params model.

    learning_rate : float, 0.1 by default
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
        updates[param] = param - (learning_rate * grad / T.sqrt(accu_new + epsilon))

    return updates


# noinspection PyUnusedLocal
def sgd_cip(cost_function, params, learning_rate=0.1, error=1):
    """ Stochastic Gradient Descent (SGD).

    Parameters
    ----------
    cost_function : theano.function
        Function cost.

    params : theano.share
        List of params model.

    learning_rate : float, 0.1 by default
        The learning rate controlling the size of update steps.
    
    error
        Error of model.       

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression.
    """
    updates = OrderedDict()
    error_curr = shared(np.float32(0.0))
    error_curr_new = T.mean(T.sqr(error))
    delta_error = error_curr_new - error_curr
    gparams = [T.grad(cost_function, param) for param in params]
    for param, grad in zip(params, gparams):
        updates[param] = param - learning_rate * grad

    updates[error_curr] = error_curr_new
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


def adadelta(cost_function, params, learning_rate=0.1, rho=0.95, fudge_factor=1e-6):
    """ Adadelta updates

    Parameters
    ----------
    cost_function : theano.function
        Function cost.

    params : theano.share
        List of params model.

    learning_rate : float, 0.1 by default
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
        updates[param] = param - learning_rate * update
        delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates


def rmsprop(cost_function, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """ RMSProp updates

    Parameters
    ----------
    cost_function : theano.function
        Function cost.

    params : theano.share
        List of params model.

    learning_rate : float, 0.1 by default
        The learning rate controlling the size of update steps.

    rho : float, 0.9 by default
        Squared gradient moving average decay factor.

    epsilon : float, 1e-6 by default
        Small value added for numerical stability.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.

    Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
    learning rate :math:`\\eta_t` is calculated as:

    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}

    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """
    gparams = [T.grad(cost_function, param) for param in params]
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, gparams):
        value = param.get_value(borrow=True)
        accu = shared(np.zeros(value.shape, dtype=value.dtype),
                      broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates


def adam(cost_function, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    """ Adam updates

    Parameters
    ----------
    cost_function : theano.function
        Function cost.

    params : theano.share
        List of params model.

    learning_rate : float
        Learning rate

    beta1 : float
        Exponential decay rate for the first moment estimates.

    beta2 : float
        Exponential decay rate for the second moment estimates.

    epsilon : float
        Constant for numerical stability.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    The paper [1]_ includes an additional hyperparameter lambda. This is only
    needed to prove convergence of the algorithm and has no practical use
    (personal communication with the authors), it is therefore omitted here.

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """
    gparams = [T.grad(cost_function, param) for param in params]
    t_prev = shared(0.)
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate * T.sqrt(one - beta2 ** t) / (one - beta1 ** t)

    for param, g_t in zip(params, gparams):
        value = param.get_value(borrow=True)
        m_prev = shared(np.zeros(value.shape, dtype=value.dtype),
                        broadcastable=param.broadcastable)
        v_prev = shared(np.zeros(value.shape, dtype=value.dtype),
                        broadcastable=param.broadcastable)

        m_t = beta1 * m_prev + (one - beta1) * g_t
        v_t = beta2 * v_prev + (one - beta2) * g_t ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t

    return updates


# noinspection PyUnusedLocal
def count_epoch(error, _i=None):
    updates = OrderedDict()
    updates[_i] = _i + 1
    return updates
