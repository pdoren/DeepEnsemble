import logging
import sys
import unittest
from unittest import TestCase

import numpy as np
import theano.tensor as T
from theano import config

from deepensemble.layers import Dense
from deepensemble.metrics import FactoryMetrics, ClassifierMetrics
from deepensemble.models import Sequential
from deepensemble.utils import mse, translate_target, translate_binary_target
from deepensemble.utils.utils_data_bases import load_data_iris
from deepensemble.utils.utils_functions import ActivationFunctions

__author__ = 'pdoren'
__project__ = 'DeepEnsemble'

data_input, data_target, classes_labels, name_db = load_data_iris()

seq = None
name = "MLP classifier"
info = 'Neural Network type MLP for test'


class TestSequential(TestCase):
    def test_initialization(self):
        seq1 = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq1.append_comment(info)

        self.assertEqual((classes_labels == seq1.get_target_labels()).all(), True,
                         'The elements of classes labels are different')
        self.assertEqual(name, seq1.get_name(), 'Problem with name of class')
        self.assertEqual(seq1.is_classifier(), True, 'Problem identified its own type of model')

    def test_add_layer(self):
        seq2 = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq2.append_comment(info)

        seq2.add_layer(Dense(n_input=data_input.shape[1], n_output=3,
                             activation=ActivationFunctions.sigmoid))
        seq2.add_layer(Dense(n_output=3, activation=ActivationFunctions.sigmoid))

        self.assertEqual(seq2.get_dim_input(), len(data_input.shape), 'Problem with input dimension')
        self.assertEqual(seq2.get_dim_output(), 2, 'Problem with output dimension')
        self.assertEqual(seq2.get_fan_in(), data_input.shape[1], 'Problem with input shape')
        self.assertEqual(seq2.get_fan_out(), 3, 'Problem with output shape')

    def test_get_layers(self):
        seq3 = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq3.append_comment(info)

        layers = [Dense(n_input=data_input.shape[1], n_output=3,
                        activation=ActivationFunctions.sigmoid),
                  Dense(n_output=3, activation=ActivationFunctions.sigmoid)]

        for layer in layers:
            seq3.add_layer(layer)

        self.assertEqual(seq3.get_layers() == layers, True, 'Problem with append layers')

    def test_get_new_metric(self):
        seq4 = Sequential(name=name, type_model="classifier", target_labels=classes_labels)

        metric1 = FactoryMetrics.get_metric(seq4)
        metric2 = seq4.get_new_metric()

        self.assertEqual(isinstance(metric1, ClassifierMetrics), True, 'Problem with metric from FactoryMetrics')
        self.assertEqual(isinstance(metric2, ClassifierMetrics), True, 'Problem with metric from Sequential Model')
        self.assertEqual(type(metric1) is type(metric2), True, 'Problem with type of metrics')

    def test__compile(self):
        pass

    def test_fit_fast(self):
        seq5 = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq5.append_comment(info)

        seq5.add_layer(Dense(n_input=data_input.shape[1], n_output=3,
                             activation=ActivationFunctions.sigmoid))
        seq5.add_layer(Dense(n_output=3, activation=ActivationFunctions.sigmoid))
        seq5.append_cost(mse, name='MSE')
        seq5.compile(fast=True)

        seq5.fit(data_input, data_target)

    def test_fit_not_fast(self):
        seq6 = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq6.append_comment(info)

        seq6.add_layer(Dense(n_input=data_input.shape[1], n_output=3,
                             activation=ActivationFunctions.sigmoid))
        seq6.add_layer(Dense(n_output=3, activation=ActivationFunctions.sigmoid))
        seq6.append_cost(mse, name='MSE')
        seq6.compile(fast=False)

        seq6.fit(data_input, data_target)

    def test_output(self):
        seq7 = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq7.append_comment(info)

        seq7.add_layer(Dense(n_input=data_input.shape[1], n_output=4,
                             activation=ActivationFunctions.sigmoid))
        seq7.add_layer(Dense(n_output=3, activation=ActivationFunctions.sigmoid))
        seq7.append_cost(mse, name='MSE')
        seq7.compile()

        seq7.fit(data_input, data_target)

        din = data_input[0:10]
        dta = data_target[0:10]

        pred_prob = seq7.output(din, prob=True)
        pred_crisp = seq7.output(din, prob=False)

        log = logging.getLogger("Sequential.test")
        log.debug("pred prob= %s", pred_prob.eval())
        log.debug("pred crisp= %s", pred_crisp.eval())
        log.debug("Target= %s", translate_target(dta, classes_labels))

        self.assertEquals((T.argmax(pred_prob, axis=1).eval() == T.argmax(pred_crisp, axis=1).eval()).all(), True,
                          'Problem with difference between output probability and crisp')

    def test_output_bin(self):
        _classes_labels = ['class_1', 'class_2']
        N = 100
        _data_input = np.random.random(size=(N, 5)).astype(dtype=config.floatX)
        _data_target = ['class_1'] * (N // 2) + ['class_2'] * (N // 2)
        seq8 = Sequential(name=name, type_model="classifier", target_labels=_classes_labels)
        seq8.append_comment(info)

        seq8.add_layer(Dense(n_input=_data_input.shape[1], n_output=4,
                             activation=ActivationFunctions.tanh))
        seq8.add_layer(Dense(n_output=1, activation=ActivationFunctions.tanh))
        seq8.append_cost(mse, name='MSE')
        seq8.compile()

        seq8.fit(_data_input, _data_target)

        din = _data_input[0:5]
        dta = _data_target[0:5]

        pred_prob = seq8.output(din, prob=True)
        pred_crisp = seq8.output(din, prob=False)

        log = logging.getLogger("Sequential.test")
        log.debug("pred prob= %s", pred_prob.eval())
        log.debug("pred crisp= %s", pred_crisp.eval())
        log.debug("Target= %s", translate_binary_target(dta, _classes_labels))

        din = _data_input[N // 2:N // 2 + 5]
        dta = _data_target[N // 2:N // 2 + 5]

        pred_prob = seq8.output(din, prob=True)
        pred_crisp = seq8.output(din, prob=False)

        log = logging.getLogger("Sequential.test")
        log.debug("pred prob= %s", pred_prob.eval())
        log.debug("pred crisp= %s", pred_crisp.eval())
        log.debug("Target= %s", translate_binary_target(dta, _classes_labels))

        self.assertEquals((T.argmax(pred_prob, axis=1).eval() == T.argmax(pred_crisp, axis=1).eval()).all(), True,
                          'Problem with difference between output probability and crisp')

    def test_reset(self):
        seq9 = Sequential(name=name, type_model="classifier", target_labels=classes_labels)
        seq9.append_comment(info)

        seq9.add_layer(Dense(n_input=data_input.shape[1], n_output=3,
                             activation=ActivationFunctions.sigmoid))
        seq9.add_layer(Dense(n_output=3, activation=ActivationFunctions.sigmoid))

        params1 = seq9.save_params()

        seq9.reset()

        params2 = seq9.save_params()

        cmp_values = np.array([(p1 == p2).all() for p1, p2 in zip(params1, params2)])

        self.assertEquals(cmp_values.all(), False, 'Problem with reset parameters on Sequential Model')


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("Sequential.test").setLevel(logging.DEBUG)
    unittest.main()
