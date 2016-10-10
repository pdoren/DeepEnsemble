import numpy as np

from sklearn import cross_validation

from deepensemble.layers import *
from deepensemble.models import *
from deepensemble.utils import *
from deepensemble.utils.utils_functions import *

RANDOM_SEED = 13
np.random.seed(RANDOM_SEED)

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