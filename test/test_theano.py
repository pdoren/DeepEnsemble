import numpy as np
from theano import function, shared, config, scan
import theano.tensor as T
from collections import OrderedDict
import matplotlib.pyplot as plt


def mse(a1, a2):
    return T.mean(T.power(a1-a2, 2.0))


def _output(_in, params):
    return T.tanh(T.dot(_in, params))


def test1():
    p = np.array(np.random.uniform(0, 1, (3, 1)), dtype=config.floatX)
    params = [shared(p, name='W', borrow=True)]
    learning_rate = 0.5
    _input = T.matrix('input_1')
    _target = T.matrix('target_1')

    cost = mse(_output(_input, params), _target)

    gparams = [T.grad(cost, param) for param in params]
    updates = OrderedDict()

    for param, grad in zip(params, gparams):
        updates[param] = param - learning_rate * grad

    fun_train = function([_input, _target], cost, updates=updates)

    batch_size = 32

    t = np.linspace(0, 6, 1000, dtype=config.floatX)
    x1 = np.sin(t)
    x2 = t
    x3 = t**2
    y = np.tanh(1 * x1 + 0.1 * x3 - 0.8 * x2)
    x = np.vstack((x1, x2, x3)).T
    c = fun_train(x, y[:, np.newaxis])
    o = np.squeeze(_output(x, params).eval())
    plt.hold(True)
    plt.plot(t, y)
    plt.plot(t, o)
    plt.hold(False)
    plt.show()


def test2():
    coefficients = T.vector("coefficients")

    x = T.scalar("x")

    max_coefficients_supported = 10000

    # Generate the components of the polynomial
    components, updates = scan(fn=lambda coefficient, power, free_variable: coefficient * (free_variable ** power),
                                      outputs_info=None,
                                      sequences=[coefficients, T.arange(max_coefficients_supported)],
                                      non_sequences=x)
    # Sum them up
    polynomial = components.sum()

    # Compile a function
    calculate_polynomial = function(inputs=[coefficients, x], outputs=polynomial)

    # Test
    test_coefficients = np.asarray([1, 0, 2], dtype=np.float32)
    test_value = 3
    print(calculate_polynomial(test_coefficients, test_value))
    print(1.0 * (3 ** 0) + 0.0 * (3 ** 1) + 2.0 * (3 ** 2))


def test3():
    import numpy
    import time

    import theano
    from theano import function
    from theano import config
    from theano import tensor as T

    def numpy_one_hot(random_integers, n_dim, max_integer):
        np_M = numpy.zeros((n_dim, max_integer + 1))
        np_M[numpy.arange(n_dim), random_integers] = 1
        return np_M

    def pylearn2_one_hot(random_integers, n_dim, max_integer):
        np_M = numpy.zeros((n_dim, max_integer + 1))
        np_M.flat[numpy.arange(0, np_M.size, np_M.shape[1]) + random_integers] = 1
        return np_M

    num_tests = 100
    n_dim = 1000
    max_integer = 100

    random_integers = numpy.random.random_integers(0, max_integer, (n_dim,))
    # print(random_integers)

    th_random_integers = T.lvector()

    # Only subtensor
    th_M_subtensor = shared(numpy.zeros((n_dim, max_integer + 1)), config.floatX)
    th_M_subtensor = T.set_subtensor(th_M_subtensor[T.arange(n_dim), th_random_integers], 1.)
    f_theano_one_hot_subtensor = function(inputs=[th_random_integers], outputs=[th_M_subtensor])

    # Flatten subtensor
    th_M_subtensor_flat = shared(numpy.zeros((n_dim, max_integer + 1)), config.floatX)
    th_M_subtensor_flat = T.set_subtensor(
        th_M_subtensor_flat.flatten()[T.arange(0, n_dim * (max_integer + 1), max_integer + 1) + th_random_integers],
        1.).reshape((n_dim, max_integer + 1))
    f_theano_one_hot_subtensor_flat = function(inputs=[th_random_integers], outputs=[th_M_subtensor_flat])

    # Padding
    ranges = T.shape_padleft(T.arange(max_integer + 1, dtype=config.floatX), th_random_integers.ndim)
    th_M_one_hot = T.eq(ranges, T.shape_padright(th_random_integers, 1)).astype(config.floatX)
    f_theano_one_hot_padding = function(inputs=[th_random_integers], outputs=[th_M_one_hot])

    # print(numpy_one_hot(random_integers, n_dim, max_integer))
    # print(pylearn2_one_hot(random_integers, n_dim, max_integer))
    # print(f_theano_one_hot_subtensor(random_integers))
    # print(f_theano_one_hot_padding(random_integers))

    time_np = 0
    time_pylearn2 = 0
    time_th_subtensor = 0
    time_th_padding = 0
    time_th_subtensor_flat = 0

    for t in range(num_tests):
        random_integers = numpy.random.random_integers(0, max_integer, (n_dim,))

        t1 = time.clock()
        current = numpy_one_hot(random_integers, n_dim, max_integer)
        t2 = time.clock()
        time_np += t2 - t1

        t1 = time.clock()
        current = pylearn2_one_hot(random_integers, n_dim, max_integer)
        t2 = time.clock()
        time_pylearn2 += t2 - t1

        t1 = time.clock()
        current = f_theano_one_hot_padding(random_integers)
        t2 = time.clock()
        time_th_padding += t2 - t1

        t1 = time.clock()
        current = f_theano_one_hot_subtensor(random_integers)
        t2 = time.clock()
        time_th_subtensor += t2 - t1

        t1 = time.clock()
        current = f_theano_one_hot_subtensor_flat(random_integers)
        t2 = time.clock()
        time_th_subtensor_flat += t2 - t1

    print(["time numpy : ", time_np])
    print(["time theano padding : ", time_th_padding])
    print(["time theano subtensor : ", time_th_subtensor])
    print(["time theano subtensor flat : ", time_th_subtensor_flat])

test3()
