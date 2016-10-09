import time

import matplotlib.pylab as plt
import numpy as np
import theano
import os
import theano.tensor as T
from sklearn import cross_validation
from sklearn.cross_validation import ShuffleSplit
from sklearn.datasets import load_iris

from deepensemble.combiner import *
from deepensemble.layers import *
from deepensemble.metrics import *
from deepensemble.models import *
from deepensemble.utils import *
from deepensemble.utils.utils_functions import *


def load_data_iris():
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
    classes_labels = iris.target_names

    return data_input, data_target, classes_labels


def test_classifier(cls, input_train, target_train, input_test, target_test, folds=25, max_epoch=300, **kwargs):
    """ Test on classifier.
    """
    path = cls.get_name() + '/'
    os.mkdir(path)

    metrics = FactoryMetrics.get_metric(cls)
    Logger().reset()
    for i in range(folds):
        metric = cls.fit(input_train, target_train, max_epoch=max_epoch, **kwargs)

        # Compute metrics
        metrics.append_prediction(target_test, cls.predict(input_test))
        metrics.append_metric(metric)

        cls.save(path + '%s_epoch_%d.dat' % (cls.get_name(), i))

        # Reset parameters
        cls.reset()

    # Compute and Show metrics
    plt.style.use('ggplot')
    metrics.classification_report()
    if isinstance(cls, EnsembleModel):
        metrics.diversity_report()

    fig_ = []
    fig_.append((metrics.plot_confusion_matrix(), 'confusion_matrix'))
    fig_.append((metrics.plot_cost(max_epoch), 'Cost'))
    fig_.append((metrics.plot_costs(max_epoch), 'Costs'))
    fig_.append((metrics.plot_scores(max_epoch), 'Scores'))

    if isinstance(cls, EnsembleModel):
        for key in metrics.get_models_metric():
            model = metrics.get_models_metric()[key].get_model()
            fig_.append((metrics.get_models_metric()[key].plot_costs(max_epoch), 'Cost_' + model.get_name()))
            fig_.append((metrics.get_models_metric()[key].plot_scores(max_epoch), 'Cost_' + model.get_name()))

    for fig, name in fig_:
        fig.savefig(path + name + '.pdf', format='pdf', dpi=1200)

    Logger().save_buffer(path + 'info.txt')

    print(':) OK')

def mlp_classifier(n_input, n_output, classes_labels):
    net = Sequential("mlp", "classifier", classes_labels)
    net.add_layer(Dense(n_input=n_input, n_output=5,
                        activation=ActivationFunctions.tanh))
    net.add_layer(Dense(n_output=n_output, activation=ActivationFunctions.softmax))
    net.append_cost(mse, name="MSE")
    net.append_reg(L1, name='Regularization L1', lamb=0.005)
    net.append_reg(L2, name='Regularization L2', lamb=0.001)
    net.append_score(score_accuracy, name='Accuracy')
    net.set_update(adagrad, name="Adagrad", initial_learning_rate=0.1)

    net.compile(fast=False)

    return net


def test():

    clrs = []
    dbs = []

    dbs.append(load_data_iris())

    for data_input, data_target, classes_labels in dbs:

        n_input = data_input.shape[1]
        n_output = len(classes_labels)

        input_train, input_test, target_train, target_test = cross_validation.train_test_split(
            data_input, data_target, stratify=data_target, test_size=0.3, random_state=32)

        clrs.append(mlp_classifier(n_input=n_input, n_output=n_output, classes_labels=classes_labels))

        for cls in clrs:
            test_classifier(cls, input_train, target_train, input_test, target_test, folds=20, max_epoch=300,
                            batch_size=24, early_stop=False,
                            improvement_threshold=0.99995, update_sets=False)

        plt.show()

test()