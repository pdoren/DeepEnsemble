from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets.mldata import fetch_mldata
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

import theano
import collections
import numpy as np

__all__ = ['load_data', 'load_data_iris', 'mackey_glass', 'mso', 'lorentz', 'load_data_cancer', 'load_ionosphere',
           'load_data_segment',
           'jacobs', 'friendman',
           'add_noise']


def load_data(db_name, classes_labels=None, normalize=True, data_home='data'):
    """ Load data from mldata.org.

    Parameters
    ----------
    db_name : str
        Name of data base.

    classes_labels : list[]
        Labels of classes.

    normalize : bool
        Flag for indicate if it necessary to normalize the data.

    data_home : str
        String with path of data bases directory.

    Returns
    -------
    tuple
        Returns a tuple with data as follow:
        (input data, target data, labels classes, name data base, description data base, list with feature names)
    """
    db = fetch_mldata(db_name, data_home=data_home)
    if isinstance(db.data, csr_matrix):
        data_input = db.data.todense()
    else:
        data_input = db.data

    if normalize:
        scaler = StandardScaler()
        scaler.fit(data_input)
        data_input = scaler.transform(data_input)
    data_input = np.asarray(data_input, dtype=theano.config.floatX)

    if hasattr(db, 'target_names'):
        classes_labels = db.target_names

    if classes_labels is None:
        classes_labels = ['Class %s' % i for i in set(db.target)]

    classes_labels = np.asarray(classes_labels, dtype='<U10')
    if len(classes_labels) > 2:
        if np.min(db.target) > 0:
            db.target -= 1
    else:
        db.target[db.target == -1] = 0

    data_target = classes_labels[np.asarray(db.target, dtype=int)]

    return data_input, data_target, classes_labels, db_name, db.DESCR, db.COL_NAMES


def load_data_segment(normalize=True, data_home='data'):
    """ Load segment data set from mldata.org.

    Parameters
    ----------
    normalize : bool
        Flag for indicate if it necessary to normalize the data.

    data_home : str
        String with path of data bases directory.

    Returns
    -------
    tuple
        Returns a tuple with data as follow:
        (input data, target data, labels classes, name data base, description data base, list with feature names)
    """
    db_name = 'uci 20070111 segment'
    db = fetch_mldata(db_name, data_home=data_home)
    if isinstance(db.data, csr_matrix):
        data_input = db.data.todense()
    else:
        data_input = db.data

    if normalize:
        scaler = StandardScaler()
        scaler.fit(data_input)
        data_input = scaler.transform(data_input)
    data_input = np.asarray(data_input, dtype=theano.config.floatX)

    if hasattr(db, 'target_names'):
        classes_labels = db.target_names

    if classes_labels is None:
        classes_labels = ['Class %s' % i for i in set(db.target)]

    classes_labels = np.asarray(classes_labels, dtype='<U10')
    if len(classes_labels) > 2:
        if np.min(db.target) > 0:
            db.target -= 1
    else:
        db.target[db.target == -1] = 0

    data_target = classes_labels[np.asarray(db.target, dtype=int)]

    return data_input, data_target, classes_labels, db_name, db.DESCR, db.COL_NAMES


def load_ionosphere(classes_labels=None, normalize=True, data_home='data'):
    """ Load data UCI Ionosphere from mldata.org.

    Parameters
    ----------
    classes_labels : list[]
        Labels of classes.

    normalize : bool
        Flag for indicate if it necessary to normalize the data.

    data_home : str
        String with path of data bases directory.

    Returns
    -------
    tuple
        Returns a tuple with data as follow:
        (input data, target data, labels classes, name data base, description data base, list with feature names)
    """
    db_name = 'datasets uci ionosphere'
    db = fetch_mldata(db_name, data_home=data_home)
    if isinstance(db.data, csr_matrix):
        data_input = db.data.todense()
    else:
        data_input = db.data

    if normalize:
        scaler = StandardScaler()
        scaler.fit(data_input)
        data_input = scaler.transform(data_input)
    data_input = np.asarray(data_input, dtype=theano.config.floatX)

    if hasattr(db, 'target_names'):
        classes_labels = db.target_names

    db.target = db.target[0]

    if classes_labels is None:
        classes_labels = ['Class %s' % i for i in set(db.target)]

    classes_labels = np.asarray(classes_labels, dtype='<U10')
    db.target[db.target == -1] = 0
    data_target = classes_labels[np.asarray(db.target, dtype=int)]

    return data_input, data_target, classes_labels, db_name, db.DESCR, db.COL_NAMES


def load_data_iris():
    """ Load data Iris.

    Returns
    -------
    tuple
        Returns a tuple with data as follow:
        (input data, target data, labels classes, name data base)
    """
    iris = load_iris()
    data_input = np.asarray(iris.data, dtype=theano.config.floatX)
    data_target = iris.target_names[iris.target]
    classes_labels = iris.target_names

    return data_input, data_target, classes_labels, 'Iris'


def load_data_cancer(normalize=True):
    """ Load data Iris.

    Returns
    -------
    tuple
        Returns a tuple with data as follow:
        (input data, target data, labels classes, name data base)
    """
    data = load_breast_cancer()
    data_target = data.target_names[data.target]
    classes_labels = data.target_names

    data_input = data.data
    if normalize:
        scaler = StandardScaler()
        scaler.fit(data_input)
        data_input = scaler.transform(data_input)
    data_input = np.asarray(data_input, dtype=theano.config.floatX)

    return data_input, data_target, classes_labels, 'Breast Cancer', data.DESCR, data.feature_names


# ===============================================================================

def mackey_glass(sample_len=1000, tau=17, seed=None, n_samples=1):
    """
    mackey_glass(sample_len=1000, tau=17, seed = None, n_samples = 1) -> input
    Generate the Mackey Glass time-series. Parameters are:
        - sample_len: length of the time-series in timesteps. Default is 1000.
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild
          chaos) and tau=30 (moderate chaos). Default is 17.
        - seed: to seed the random generator, can be used to generate the same
          timeseries at each invocation.
        - n_samples : number of samples to generate
    """
    delta_t = 10
    history_len = tau * delta_t
    # Initial conditions for the history of the system
    timeseries = 1.2

    if seed is not None:
        np.random.seed(seed)

    samples = []

    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * (np.random.rand(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((sample_len, 1))

        for timestep in range(sample_len):
            # noinspection PyAssignmentToLoopOrWithParameter
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - 0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries

        # Squash timeseries through tanh
        inp = np.tanh(inp - 1)
        samples.append(inp)
    return samples


def mso(sample_len=1000, n_samples=1):
    """
    mso(sample_len=1000, n_samples = 1) -> input
    Generate the Multiple Sinewave Oscillator time-series, a sum of two sines
    with incommensurable periods. Parameters are:
        - sample_len: length of the time-series in timesteps
        - n_samples: number of samples to generate
    """
    signals = []
    for _ in range(n_samples):
        phase = np.random.rand()
        x = np.atleast_2d(np.arange(sample_len)).T
        signals.append(np.sin(0.2 * x + phase) + np.sin(0.311 * x + phase))
    return signals


def lorentz(sample_len=1000, sigma=10, rho=28, beta=8 / 3, step=0.01):
    """This function generates a Lorentz time series of length sample_len,
    with standard parameters sigma, rho and beta.
    """

    x = np.zeros([sample_len])
    y = np.zeros([sample_len])
    z = np.zeros([sample_len])

    # Initial conditions taken from 'Chaos and Time Series Analysis', J. Sprott
    x[0] = 0
    y[0] = -0.01
    z[0] = 9

    for t in range(sample_len - 1):
        x[t + 1] = x[t] + sigma * (y[t] - x[t]) * step
        y[t + 1] = y[t] + (x[t] * (rho - z[t]) - y[t]) * step
        z[t + 1] = z[t] + (x[t] * y[t] - beta * z[t]) * step

    x.shape += (1,)
    y.shape += (1,)
    z.shape += (1,)

    return np.concatenate((x, y, z), axis=1)


# ===============================================================================

def jacobs(sample_len=1000, seed=13):
    np.random.seed(seed=0)
    np.random.seed(seed=seed)

    X = np.random.rand(sample_len, 5)
    y = (1.0 / 13.0) * (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) +
                        20 * np.power((X[:, 2] - 0.5), 2) +
                        10 * X[:, 3] + 5 * X[:, 4]) - 1

    return X, y


def friendman(sample_len=1000, seed=13):
    np.random.seed(seed=0)
    np.random.seed(seed=seed)

    X = np.random.rand(sample_len, 10)
    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + \
        20 * np.power((X[:, 2] - 0.5), 2) + 10 * X[:, 3] + 5 * X[:, 4]

    return X, y


# ===============================================================================

def add_noise(y, snr=1.0, seed=13):
    np.random.seed(seed=0)
    np.random.seed(seed=seed)

    noise = np.random.normal(loc=0.0, scale=1.0 / snr, size=y.shape)

    return y + noise
