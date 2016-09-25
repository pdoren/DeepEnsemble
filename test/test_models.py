import time

import matplotlib.pylab as plt
import numpy as np
import theano
import theano.tensor as T
from sklearn import cross_validation
from sklearn.cross_validation import ShuffleSplit
from sklearn.datasets import load_iris

from deepensemble.combiner.weightaveragecombiner import WeightAverageCombiner
from deepensemble.layers.dense import Dense
from deepensemble.models.ensemblemodel import EnsembleModel
from deepensemble.models.sequential import Sequential
from deepensemble.utils import *


def load_data():
    """ Load Iris data base.

    Returns
    -------
    tuple
        Returns Iris data base.
    """
    # Load Data
    iris = load_iris()
    data_input = np.asarray(iris.data, dtype=theano.config.floatX)
    data_target = iris.target_names[iris.target]
    classes_names = iris.target_names

    return data_input, data_target, classes_names


def test_mlp():
    """ Test MLP classifier with Iris data base.
    """

    data_input, data_target, classes_names = load_data()

    # Create model MLP
    reg = 0.001
    lr = 0.1

    mlp1 = Sequential(classes_names, "classifier", "net1")
    mlp1.add_layer(Dense(n_input=data_input.shape[1], n_output=8, activation=T.tanh))
    mlp1.add_layer(Dense(n_output=len(classes_names), activation=T.nnet.softmax))
    mlp1.append_cost(mse)
    mlp1.append_reg(L1, lamb=reg)
    mlp1.append_reg(L2, lamb=reg)
    mlp1.set_update(adagrad, initial_learning_rate=lr)
    mlp1.compile()

    folds = 2
    sss = ShuffleSplit(data_input.shape[0], n_iter=folds, test_size=None, train_size=0.6, random_state=0)
    max_epoch = 400
    validation_jump = 5

    metrics_mlp = ClassifierMetrics(mlp1)

    for i, (train_set, test_set) in enumerate(sss):
        # data train and test
        input_train = data_input[train_set]
        input_test = data_input[test_set]
        target_train = data_target[train_set]
        target_test = data_target[test_set]

        tic = time.time()
        metrics_mlp.append_metric(mlp1.fit(input_train, target_train,
                                           max_epoch=max_epoch, batch_size=32,
                                           validation_jump=validation_jump, improvement_threshold=4))
        toc = time.time()
        # Compute metrics
        metrics_mlp.append_prediction(target_test, mlp1.predict(input_test))

        # Reset parameters
        mlp1.reset()

        print("%d Elapsed time [s]: %f" % (i, toc - tic))

    # Compute and Show metrics
    metrics_mlp.plot_confusion_matrix()
    metrics_mlp.plot_cost(max_epoch, "Cost MLP")
    metrics_mlp.plot_score(max_epoch, "Score MLP")
    plt.show()
    print('TEST MLP OK')


def test_ensemble():
    """ Test Ensemble Neural Network classifier with Iris data base.
    """

    data_input, data_target, classes_names = load_data()

    # Generate data train and test
    input_train, input_test, target_train, target_test = cross_validation.train_test_split(
        data_input, data_target, test_size=0.4, random_state=0)

    # Create models for ensemble
    reg = 0.001
    lr = 0.1

    mlp1 = Sequential(classes_names, "classifier", "net1")
    mlp1.add_layer(Dense(n_input=data_input.shape[1], n_output=8, activation=T.tanh))
    mlp1.add_layer(Dense(n_output=len(classes_names), activation=T.nnet.softmax))
    mlp1.append_cost(mse)
    mlp1.append_reg(L1, lamb=reg)
    mlp1.append_reg(L2, lamb=reg)
    mlp1.set_update(sgd_momentum, learning_rate=lr, momentum_rate=0.9)

    mlp2 = Sequential(classes_names, "classifier", "net2")
    mlp2.add_layer(Dense(n_input=data_input.shape[1], n_output=3, activation=T.tanh))
    mlp2.add_layer(Dense(n_output=len(classes_names), activation=T.nnet.softmax))
    mlp2.append_cost(mse)
    mlp2.append_reg(L1, lamb=reg)
    mlp2.append_reg(L2, lamb=reg)
    mlp2.set_update(sgd_momentum, learning_rate=lr, momentum_rate=0.9)

    mlp3 = Sequential(classes_names, "classifier", "net3")
    mlp3.add_layer(Dense(n_input=data_input.shape[1], n_output=3, activation=T.tanh))
    mlp3.add_layer(Dense(n_output=len(classes_names), activation=T.nnet.softmax))
    mlp3.append_cost(mse)
    mlp3.append_reg(L1, lamb=reg)
    mlp3.append_reg(L2, lamb=reg)
    mlp3.set_update(sgd_momentum, learning_rate=lr, momentum_rate=0.9)

    mlp4 = Sequential(classes_names, "classifier", "net4")
    mlp4.add_layer(Dense(n_input=data_input.shape[1], n_output=3, activation=T.tanh))
    mlp4.add_layer(Dense(n_output=len(classes_names), activation=T.nnet.softmax))
    mlp4.append_cost(mse)
    mlp4.append_reg(L1, lamb=reg)
    mlp4.append_reg(L2, lamb=reg)
    mlp4.set_update(sgd_momentum, learning_rate=lr, momentum_rate=0.9)

    # Create Ensemble
    ensemble = EnsembleModel()

    ensemble.append_model(mlp1)
    ensemble.append_model(mlp2)
    ensemble.append_model(mlp3)
    ensemble.append_model(mlp4)

    ensemble.set_combiner(WeightAverageCombiner(4))

    # compile ensemble: update cost and update function

    ensemble.add_cost_ensemble(fun_cost=neg_corr, lamb_neg_corr=0.3)  # adds neg correlation in all models
    ensemble.compile()

    # Training

    max_epoch = 200
    validation_jump = 5

    classifier_metrics = ensemble.fit(input_train, target_train,
                                      max_epoch=max_epoch, batch_size=32,
                                      validation_jump=validation_jump, improvement_threshold=4, verbose=True)

    # Compute and Show metrics
    classifier_metrics.append_prediction(target_test, ensemble.predict(input_test))
    classifier_metrics.plot_confusion_matrix()
    classifier_metrics.plot_cost(max_epoch, "Cost ensemble")
    classifier_metrics.plot_score(max_epoch, "Score ensemble")
    plt.show()

test_mlp()

