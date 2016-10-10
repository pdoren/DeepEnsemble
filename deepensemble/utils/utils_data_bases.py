from sklearn.datasets import load_iris
from sklearn.datasets.mldata import fetch_mldata
from scipy.sparse import csr_matrix

import theano
import numpy as np

__all__ = ['load_data', 'load_data_iris']

DATA_HOME = 'data'


def load_data(db_name, classes_labels=None, normalize=True):
    db = fetch_mldata(db_name, data_home=DATA_HOME)
    if isinstance(db.data, csr_matrix):
        data_input = np.asarray(db.data.todense(), dtype=theano.config.floatX)
    else:
        data_input = np.asarray(db.data, dtype=theano.config.floatX)

    if hasattr(db, 'target_names'):
        classes_labels = db.target_names

    if classes_labels is None:
        classes_labels = ['Class %s' % i for i in set(db.target)]

    classes_labels = np.asarray(classes_labels, dtype='<U10')
    db.target[db.target == -1] = 0
    data_target = classes_labels[np.asarray(db.target, dtype=int)]

    if normalize:
        data_input = (data_input - np.mean(data_input, axis=0)) / np.var(data_input, axis=0)

    return data_input, data_target, classes_labels, db_name, db.DESCR, db.COL_NAMES


def load_data_iris():
    iris = load_iris()
    data_input = np.asarray(iris.data, dtype=theano.config.floatX)
    data_target = iris.target_names[iris.target]
    classes_labels = iris.target_names

    return data_input, data_target, classes_labels, 'Iris'
