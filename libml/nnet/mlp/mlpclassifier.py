import numpy as np
import theano
import theano.tensor as T
from libml.nnet.mlp.mlpmodel import MLPModel


class MLPClassifier(MLPModel):
    def __init__(self, n_input, n_hidden, target_labels=None,
                 output_activation=T.nnet.sigmoid, hidden_activation=None):
        """ Multi-Layer Perceptron (MLP) for classification

        A multilayer perceptron is a feedforward artificial neural network model
        that has one layer or more of hidden units and nonlinear activations.

        Parameters
        ----------
        n_input
        n_hidden
        target_labels
        output_activation
        hidden_activation
        """
        if target_labels is None:
            raise ValueError("Incorrect labels target")

        n_classes = len(target_labels)
        if n_classes == 0:
            raise ValueError("Incorrect labels target")

        super(MLPClassifier, self).__init__(n_input=n_input, n_hidden=n_hidden, n_output=n_classes,
                                            output_activation=output_activation, hidden_activation=hidden_activation,
                                            type_model='classifier')

        self.target_labels = np.array(target_labels)

    def translate_target(self, _target):
        """ For each example you get a vector indicating the "index" from the vector labels class, this vector
        has all its elements in zero except the element of the position equals to "index" that it is 1.

        Parameters
        ----------
        _target

        Returns
        -------

        """
        target = np.zeros(shape=(len(_target), self.n_output), dtype=theano.config.floatX)
        for i, label in enumerate(_target):
            target[i, list(self.target_labels).index(label)] = 1.0
        return target

    def output(self, _input):
        """ Output of MLP

        Parameters
        ----------
        _input

        Returns
        -------

        """
        for layer in self.layers:
            _input = layer.output(_input)
        return _input

    def translate_output(self, _output):
        """ Evaluating _output of MLP

        Parameters
        ----------
        _output

        Returns
        -------

        """
        return self.target_labels[T.argmax(_output, axis=1).eval()]
