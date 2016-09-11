import theano
import theano.tensor as T
import numpy as np
import matplotlib.pylab as plt

from sklearn import cross_validation
from sklearn.datasets import load_iris
from theano.sandbox import cuda

from libml.ensemble.ensemblemodel import EnsembleModel
from libml.ensemble.combiner import *
from libml.models.sequential import Sequential
from libml.layers.dense import Dense

from libml.utils import *


def test1():
    # Load Data

    iris = load_iris()

    data_input = np.asarray(iris.data, dtype=theano.config.floatX)
    data_target = iris.target_names[iris.target]
    classes_names = iris.target_names

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
                                      validation_jump=validation_jump, early_stop_th=4, verbose=True)

    # Compute and Show metrics
    classifier_metrics.append_prediction(target_test, ensemble.predict(input_test))
    classifier_metrics.plot_confusion_matrix()
    classifier_metrics.plot_cost(max_epoch, "Cost ensemble")
    classifier_metrics.plot_score(max_epoch, "Score ensemble")

    plt.show()

    print('TEST 1 OK')


if __name__ == "__main__":
    theano.config.floatX = 'float32'
    cuda.use('gpu')
    theano.config.compute_test_value = 'off'

    test1()
