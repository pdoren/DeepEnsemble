import time
import os

import matplotlib.pylab as plt
import numpy as np
import theano

from sklearn import cross_validation
from sklearn.datasets import load_iris
from sklearn.datasets.mldata import fetch_mldata

from scipy.sparse import csr_matrix

from deepensemble.combiner import *
from deepensemble.layers import *
from deepensemble.metrics import *
from deepensemble.models import *
from deepensemble.utils import *
from deepensemble.utils.utils_functions import *


data_home = 'data'
seed = 8014

def load_data(db_name, classes_labels):
    db = fetch_mldata(db_name, data_home=data_home)
    if isinstance(db.data, csr_matrix):
        data_input = np.asarray(db.data.todense(), dtype=theano.config.floatX)
    else:
        data_input = np.asarray(db.data, dtype=theano.config.floatX)

    if hasattr(db, 'target_names'):
        classes_labels = db.target_names

    classes_labels = np.asarray(classes_labels, dtype='<U10')
    db.target[db.target == -1] = 0
    data_target = classes_labels[np.asarray(db.target, dtype=int)]

    return data_input, data_target, classes_labels, db_name


def load_data_iris():
    iris = load_iris()
    data_input = np.asarray(iris.data, dtype=theano.config.floatX)
    data_target = iris.target_names[iris.target]
    classes_labels = iris.target_names

    return data_input, data_target, classes_labels, 'Iris'


def test_classifier(dir, cls, input_train, target_train, input_test, target_test, folds=25, max_epoch=300, **kwargs):
    """ Test on classifier.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

    metrics = FactoryMetrics.get_metric(cls)
    Logger().reset()

    best_params = None
    best_score = 0
    for i in range(folds):
        metric = cls.fit(input_train, target_train, max_epoch=max_epoch, **kwargs)

        # Compute metrics
        score = metrics.append_prediction(target_test, cls.predict(input_test))
        metrics.append_metric(metric)

        # Save the best params
        if score > best_score:
            best_params = cls.save_params()
            best_score = score
        elif score == best_params:
            score_curr = metrics.get_score_prediction(target_train, cls.predict(input_train))
            params_curr = cls.save_params()
            cls.load_params(best_params)
            score_best = metrics.get_score_prediction(target_train, cls.predict(input_train))

            if score_curr > score_best:
                best_params = params_curr

        # Reset parameters
        cls.reset()

    Logger().print('wait ... ', end='')
    # Load the best params
    if best_params is not None:
        cls.load_params(best_params)

    # Save classifier
    cls.save(dir + '%s_classifier.pkl' % cls.get_name())

    # Compute and Show metrics
    plt.style.use('ggplot')
    metrics.classification_report()
    if isinstance(cls, EnsembleModel):
        metrics.diversity_report()

    Logger().print('The best score: %.4f' % best_score)

    fig_ = []
    fig_.append((metrics.plot_confusion_matrix(), 'confusion_matrix'))
    fig_.append((metrics.plot_confusion_matrix_prediction(target_train, cls.predict(input_train)),
                 'confusion_matrix_best_train'))
    fig_.append((metrics.plot_confusion_matrix_prediction(target_test, cls.predict(input_test)),
                 'confusion_matrix_best_test'))
    fig_.append((metrics.plot_cost(max_epoch), 'Cost'))
    fig_.append((metrics.plot_costs(max_epoch), 'Costs'))
    fig_.append((metrics.plot_scores(max_epoch), 'Scores'))

    if isinstance(cls, EnsembleModel):
        for key in metrics.get_models_metric():
            model = metrics.get_models_metric()[key].get_model()
            fig_.append((metrics.get_models_metric()[key].plot_costs(max_epoch), 'Cost_' + model.get_name()))
            fig_.append((metrics.get_models_metric()[key].plot_scores(max_epoch), 'Cost_' + model.get_name()))

    for fig, name in fig_:
        fig.savefig(dir + name + '.pdf', format='pdf', dpi=1200)
        fig.clf()

    Logger().save_buffer(dir + 'info.txt')

    print(':) OK')

    return best_score, cls


def mlp_australian(n_input, n_output, classes_labels):
    net = Sequential("mlp", "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_input, n_output=10,
                        activation=ActivationFunctions.sigmoid))
    net.add_layer(Dense(n_output=n_output, activation=ActivationFunctions.sigmoid))
    net.append_cost(mse, name="MSE")
    net.append_reg(L1, name='Regularization L1', lamb=0.005)
    net.append_reg(L2, name='Regularization L2', lamb=0.001)
    net.append_score(score_accuracy, name='Accuracy')
    net.set_update(adagrad, name="Adagrad", initial_learning_rate=0.1)

    net.compile(fast=False)

    return net


def mlp_diabetes(n_input, n_output, classes_labels):
    net = Sequential("mlp", "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_input, n_output=8,
                        activation=ActivationFunctions.tanh))
    net.add_layer(Dense(n_output=1, activation=ActivationFunctions.tanh))
    net.append_cost(mse, name="MSE")
    # net.append_cost(mcc, name="MCC")
    net.append_reg(L1, name='Regularization L1', lamb=0.05)
    net.append_reg(L2, name='Regularization L2', lamb=0.01)
    net.append_score(score_accuracy, name='Accuracy')
    net.set_update(adagrad, name="Adagrad", initial_learning_rate=0.1)
    # net.set_update(sgd, name="SGD", learning_rate=0.1)

    net.compile(fast=False)

    return net


def mlp_german(n_input, n_output, classes_labels):
    net = Sequential("mlp", "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_input, n_output=18,
                        activation=ActivationFunctions.tanh))
    net.add_layer(Dense(n_output=2, activation=ActivationFunctions.sigmoid))
    net.append_cost(mse, name="MSE")
    # net.append_cost(mcc, name="MCC")
    net.append_reg(L1, name='Regularization L1', lamb=0.005)
    net.append_reg(L2, name='Regularization L2', lamb=0.001)
    net.append_score(score_accuracy, name='Accuracy')
    net.set_update(adagrad, name="Adagrad", initial_learning_rate=0.1)
    # net.set_update(sgd, name="SGD", learning_rate=0.1)

    net.compile(fast=False)

    return net


def load_all_data():
    dbs = []

    data_dbs = {
        # 'australian_scale': ['yes', 'no'],
        # 'diabetes_scale' : ['yes', 'no'],
        'german.numer_scale' : ['class 1', 'class 2'],
        # 'a1a' : ['class 1', 'class 2'],
        # 'Breast Cancer IDA' : ['yes', 'no'],
        # 'datasets-UCI ionosphere' : ['class 1', 'class 2'],
        # 'datasets-UCI splice' : ['class 1', 'class 2'],
        # 'w1a' : ['class 1', 'class 2']
    }

    for name_db in data_dbs:
        dbs.append(load_data(name_db, data_dbs[name_db]))

    return dbs


def test_mlp():

    dbs = load_all_data()

    for data_input, data_target, classes_labels, name_db in dbs:

        n_input = data_input.shape[1]
        n_output = len(classes_labels)

        batch_size = 32

        dir_db = name_db + '/'

        input_train, input_test, target_train, target_test = cross_validation.train_test_split(
            data_input, data_target, stratify=data_target, test_size=0.3, random_state=seed)

        net = mlp_german(n_input, n_output, classes_labels)

        # Print Info Data and Training
        Logger().print('Model %s | in: %d, out: %d | classes: %s' % (name_db, n_input, n_output, classes_labels))
        Logger().print('Examples: %d | train: %d, validation: %d ' %
                       (data_input.shape[0], input_train.shape[0], input_test.shape[0]))

        dir = dir_db + net.get_name() + '/'
        test_classifier(dir, net, input_train, target_train, input_test, target_test,
                        folds=25, max_epoch=300,
                        batch_size=batch_size, early_stop=False,
                        improvement_threshold=0.99995, update_sets=False)


test_mlp()