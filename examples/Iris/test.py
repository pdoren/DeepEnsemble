import theano
import theano.tensor as T
import numpy as np
from sklearn.datasets import load_iris
from theano.sandbox import cuda

theano.config.floatX = 'float32'
cuda.use('gpu')
theano.config.compute_test_value = 'off'

iris = load_iris()
data_input = np.asarray(iris.data, dtype=theano.config.floatX)
data_target = iris.target_names[iris.target]
classes_names = iris.target_names

import sys
import time

sys.path.insert(0, r'../../')

from libml.nnet.mlpclassifier import MLPClassifier
from libml.trainers.trainermlp import TrainerMLP

from sklearn.cross_validation import ShuffleSplit
from utils.classifiermetrics import ClassifierMetrics

classifier = MLPClassifier(data_input.shape[1], [5], classes_names,
                           output_activation=T.tanh,
                           hidden_activation=[T.tanh])

trainerMLP = TrainerMLP(classifier, cost="MSE", lr_adapt="CONS",
                        initial_learning_rate=0.05, initial_momentum_rate=0.9, regularizer="L2+L1")

folds = 2
sss = ShuffleSplit(data_input.shape[0], n_iter=folds, test_size=None, train_size=0.5, random_state=0)
max_epoch = 200
validation_jump = 5

# Initialize metrics
metrics = ClassifierMetrics(classes_names)

for i, (train_set, test_set) in enumerate(sss):
    # data train and test
    input_train = data_input[train_set]
    input_test = data_input[test_set]
    target_train = data_target[train_set]
    target_test = data_target[test_set]

    # training
    tic = time.time()
    train_cost, test_cost, best_test_predict = trainerMLP.trainer(input_train, target_train, input_test, target_test,
                                                                  max_epoch=max_epoch, reg_L1=1e-2, reg_L2=1e-3,
                                                                  batch_size=32,
                                                                  validation_jump=validation_jump, early_stop_th=4)
    toc = time.time()

    # Reset parameters
    classifier.reset()

    # Compute metrics
    metrics.append_pred(target_test, best_test_predict)
    metrics.append_cost(train_cost, test_cost)

    print("%d Elapsed time [s]: %f" % (i, toc - tic))

print("FINISHED!")

import matplotlib.pylab as plt

metrics.print()
metrics.plot_confusion_matrix()
metrics.plot_cost(max_epoch)

plt.show()

