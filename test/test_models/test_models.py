import time
import os

import matplotlib.pylab as plt
import numpy as np
import theano

from sklearn import cross_validation, clone
from sklearn.datasets import load_iris
from sklearn.datasets.mldata import fetch_mldata

from scipy.sparse import csr_matrix

from deepensemble.combiner import *
from deepensemble.layers import *
from deepensemble.metrics import *
from deepensemble.models import *
from deepensemble.utils import *
from deepensemble.utils.utils_functions import *


DATA_HOME = 'data'
RANDOM_SEED = 13
np.random.seed(RANDOM_SEED)

def load_data(db_name, classes_labels, normalize=True):
    db = fetch_mldata(db_name, data_home=DATA_HOME)
    if isinstance(db.data, csr_matrix):
        data_input = np.asarray(db.data.todense(), dtype=theano.config.floatX)
    else:
        data_input = np.asarray(db.data, dtype=theano.config.floatX)

    if hasattr(db, 'target_names'):
        classes_labels = db.target_names

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


def test_classifier(dir, cls, input_train, target_train, input_test, target_test, folds=25, max_epoch=300, **kwargs):
    """ Test on classifier.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

    metrics = FactoryMetrics.get_metric(cls)

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
    net.append_reg(L1, name='Regularization L1', lamb=0.05)
    net.append_reg(L2, name='Regularization L2', lamb=0.01)
    net.append_score(score_accuracy, name='Accuracy')
    net.set_update(adagrad, name="Adagrad", initial_learning_rate=0.1)

    net.compile(fast=False)

    return net


def mlp_german(n_input, n_output, classes_labels):
    net = Sequential("mlp", "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_input, n_output=12,
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


def mlp_a1a(n_input, n_output, classes_labels):
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
        'australian_scale': ['yes', 'no'],
        # 'diabetes_scale' : ['yes', 'no'],
        # 'german.numer_scale' : ['class 1', 'class 2'],
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

    for data_input, data_target, classes_labels, name_db, desc, col_names in dbs:

        n_input = data_input.shape[1]
        n_output = len(classes_labels)

        batch_size = 32
        max_epoch = 300
        folds = 25

        dir_db = name_db + '/'

        # Generate Cross validation data
        input_train, input_valid, target_train, target_valid = cross_validation.train_test_split(
            data_input, data_target, stratify=data_target, test_size=0.3, random_state=RANDOM_SEED)

        # Select Classifier
        net = mlp_australian(n_input, n_output, classes_labels)

        # Print Info Data and Training
        Logger().reset()
        Logger().print('Model:\n %s | in: %d, out: %d' % (net.get_name(), n_input, n_output))
        Logger().print('Data (%s):\n DESC: %s.\n Features: %d\n Classes(%d): %s' %
                       (name_db, desc, n_input, n_output, classes_labels))
        Logger().print('Training:\n total data: %d | train: %d, validation: %d ' %
                       (data_input.shape[0], input_train.shape[0], input_valid.shape[0]))
        Logger().print(' folds: %d | Epoch: %d, Batch Size: %d ' %
                       (folds, max_epoch, batch_size))

        dir = dir_db + net.get_name() + '/'
        test_classifier(dir, net, input_train, target_train, input_valid, target_valid,
                        folds=folds, max_epoch=max_epoch,
                        batch_size=batch_size, early_stop=False,
                        improvement_threshold=0.99995, update_sets=False)


def test_random_forest():
    from sklearn.ensemble import RandomForestClassifier

    dbs = load_all_data()

    for data_input, data_target, classes_labels, name_db, desc, col_names in dbs:
        n_input = data_input.shape[1]
        n_output = len(classes_labels)

        batch_size = 32
        max_epoch = 300
        folds = 25

        dir_db = name_db + '/'

        # Generate Cross validation data
        input_train, input_valid, target_train, target_valid = cross_validation.train_test_split(
            data_input, data_target, stratify=data_target, test_size=0.3, random_state=RANDOM_SEED)

        # Select Classifier
        n_estimators = 30
        rf = RandomForestClassifier(n_estimators=n_estimators)
        model = Wrapper(rf, name='Random Forest', type_model='classifier', target_labels=classes_labels)

        # Print Info Data and Training
        Logger().reset()
        Logger().print('Model:\n %s | in: %d, out: %d' % ('Random Forest', n_input, n_output))
        Logger().print('Data (%s):\n DESC: %s.\n Features: %d\n Classes(%d): %s' %
                       (name_db, desc, n_input, n_output, classes_labels))
        Logger().print('Training:\n total data: %d | train: %d, validation: %d ' %
                       (data_input.shape[0], input_train.shape[0], input_valid.shape[0]))
        Logger().print(' folds: %d | Epoch: %d, Batch Size: %d ' %
                       (folds, max_epoch, batch_size))

        dir = dir_db + 'Random Forest/'
        test_classifier(dir, model, input_train, target_train, input_valid, target_valid,
                        folds=folds, max_epoch=max_epoch,
                        batch_size=batch_size, early_stop=False,
                        improvement_threshold=0.99995, update_sets=False)

#test_mlp()
test_random_forest()