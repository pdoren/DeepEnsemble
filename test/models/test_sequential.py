from unittest import TestCase
import unittest
import logging
import sys
import numpy as np
import theano.tensor as T
from theano import config

from deepensemble.utils.utils_data_bases import load_data_iris
from deepensemble.utils import mse, sgd, translate_target, translate_binary_target
from deepensemble.metrics import FactoryMetrics, ClassifierMetrics
from deepensemble.utils.utils_functions import ActivationFunctions
from deepensemble.layers import Dense
from deepensemble.models import Sequential

__author__ = 'pdoren'
__project__ = 'DeepEnsemble'

data_input, data_target, classes_labels, name_db = load_data_iris()

seq = None
name = "MLP classifier"
info = 'Neural Network type MLP for test'

class TestSequential(TestCase):

    def test_initialization(self):
        seq = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq.set_info(info)

        self.assertEqual(info, seq.get_info(), 'Problem with info')
        self.assertEqual((classes_labels == seq.get_target_labels()).all(), True,
                         'The elements of classes labels are different')
        self.assertEqual(name, seq.get_name(), 'Problem with name of class')
        self.assertEqual(seq.is_classifier(), True, 'Problem identified its own type of model')

    def test_add_layer(self):
        seq = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq.set_info(info)

        seq.add_layer(Dense(n_input=data_input.shape[1], n_output=3,
                                 activation=ActivationFunctions.sigmoid))
        seq.add_layer(Dense(n_output=3, activation=ActivationFunctions.sigmoid))

        self.assertEqual(seq.get_dim_input(), len(data_input.shape) - 1, 'Problem with input dimension')
        self.assertEqual(seq.get_dim_output(), 1, 'Problem with output dimension')
        self.assertEqual(seq.get_fan_in(), data_input.shape[1], 'Problem with input shape')
        self.assertEqual(seq.get_fan_out(), 3, 'Problem with output shape')

    def test_get_layers(self):
        seq = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq.set_info(info)

        layers = [Dense(n_input=data_input.shape[1], n_output=3,
                            activation=ActivationFunctions.sigmoid),
                  Dense(n_output=3, activation=ActivationFunctions.sigmoid)]

        for layer in layers:
            seq.add_layer(layer)

        self.assertEqual(seq.get_layers() == layers, True, 'Problem with append layers')

    def test_get_new_metric(self):
        seq = Sequential(name=name, type_model="classifier", target_labels=classes_labels)

        metric1 = FactoryMetrics.get_metric(seq)
        metric2 = seq.get_new_metric()

        self.assertEqual(isinstance(metric1, ClassifierMetrics), True, 'Problem with metric from FactoryMetrics')
        self.assertEqual(isinstance(metric2, ClassifierMetrics), True, 'Problem with metric from Sequential Model')
        self.assertEqual(type(metric1) is type(metric2), True, 'Problem with type of metrics')

    def test__compile(self):
        pass

    def test_fit_fast(self):
        seq = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq.set_info(info)

        seq.add_layer(Dense(n_input=data_input.shape[1], n_output=3,
                            activation=ActivationFunctions.sigmoid))
        seq.add_layer(Dense(n_output=3, activation=ActivationFunctions.sigmoid))
        seq.append_cost(mse, name='MSE')
        seq.compile(fast=True)

        seq.fit(data_input, data_target)

    def test_fit_not_fast(self):
        seq = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq.set_info(info)

        seq.add_layer(Dense(n_input=data_input.shape[1], n_output=3,
                            activation=ActivationFunctions.sigmoid))
        seq.add_layer(Dense(n_output=3, activation=ActivationFunctions.sigmoid))
        seq.append_cost(mse, name='MSE')
        seq.compile(fast=False)

        seq.fit(data_input, data_target)

    def test_output(self):
        seq = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq.set_info(info)

        seq.add_layer(Dense(n_input=data_input.shape[1], n_output=4,
                            activation=ActivationFunctions.sigmoid))
        seq.add_layer(Dense(n_output=3, activation=ActivationFunctions.sigmoid))
        seq.append_cost(mse, name='MSE')
        seq.compile()

        seq.fit(data_input, data_target)

        din = data_input[0:10]
        dta = data_target[0:10]

        pred_prob =seq.output(din, prob=True)
        pred_crisp = seq.output(din, prob=False)

        log = logging.getLogger("Sequential.test")
        log.debug("pred prob= %s", pred_prob.eval())
        log.debug("pred crisp= %s", pred_crisp.eval())
        log.debug("Target= %s", translate_target(dta, classes_labels))

        self.assertEquals((T.argmax(pred_prob, axis=1).eval() == T.argmax(pred_crisp, axis=1).eval()).all(), True,
                          'Problem with difference between output probability and crisp')

    def test_output_bin(self):
        classes_labels = ['class_1', 'class_2']
        N = 100
        data_input = np.random.random(size=(N, 5)).astype(dtype=config.floatX)
        data_target = ['class_1'] * (N // 2) + ['class_2'] * (N // 2)
        seq = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq.set_info(info)

        seq.add_layer(Dense(n_input=data_input.shape[1], n_output=4,
                            activation=ActivationFunctions.tanh))
        seq.add_layer(Dense(n_output=1, activation=ActivationFunctions.tanh))
        seq.append_cost(mse, name='MSE')
        seq.compile()

        seq.fit(data_input, data_target)

        din = data_input[0:5]
        dta = data_target[0:5]

        pred_prob = seq.output(din, prob=True)
        pred_crisp = seq.output(din, prob=False)

        log = logging.getLogger("Sequential.test")
        log.debug("pred prob= %s", pred_prob.eval())
        log.debug("pred crisp= %s", pred_crisp.eval())
        log.debug("Target= %s", translate_binary_target(dta, classes_labels))

        din = data_input[N//2:N//2+5]
        dta = data_target[N//2:N//2+5]

        pred_prob =seq.output(din, prob=True)
        pred_crisp = seq.output(din, prob=False)

        log = logging.getLogger("Sequential.test")
        log.debug("pred prob= %s", pred_prob.eval())
        log.debug("pred crisp= %s", pred_crisp.eval())
        log.debug("Target= %s", translate_binary_target(dta, classes_labels))

        self.assertEquals((T.argmax(pred_prob, axis=1).eval() == T.argmax(pred_crisp, axis=1).eval()).all(), True,
                          'Problem with difference between output probability and crisp')

    def test_reset(self):
        seq = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq.set_info(info)

        seq.add_layer(Dense(n_input=data_input.shape[1], n_output=3,
                            activation=ActivationFunctions.sigmoid))
        seq.add_layer(Dense(n_output=3, activation=ActivationFunctions.sigmoid))

        params1 = seq.save_params()

        seq.reset()

        params2 = seq.save_params()

        cmp_values = np.array([(p1 == p2).all() for p1, p2 in zip(params1, params2)])

        self.assertEquals(cmp_values.all(), False, 'Problem with reset parameters on Sequential Model')

if __name__ == "__main__":
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "Sequential.test" ).setLevel( logging.DEBUG )
    unittest.main()