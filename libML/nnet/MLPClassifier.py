import theano
import theano.tensor as T
import numpy as np
from libML.nnet.MLPRegressor import MLPRegressor


class MLPClassifier(MLPRegressor):
    def __init__(self, n_input, n_hidden, target_labels=None,
                 output_activation=T.nnet.sigmoid, hidden_activation=None):

        if target_labels is None:
            raise ValueError("Incorrect labels target")

        self.n_classes = len(target_labels)
        if self.n_classes == 0:
            raise ValueError("Incorrect labels target")

        super(MLPClassifier, self).__init__(n_input=n_input, n_hidden=n_hidden, n_output=self.n_classes,
                                            output_activation=output_activation, hidden_activation=hidden_activation)

        self.target_labels = target_labels
        self.type_learner = "classifier"

    def translate_output(self, _output):
        return self.target_labels[T.argmax(_output, axis=1).eval()]

    def translate_target(self, _target):
        target = np.zeros(shape=(len(_target), self.n_classes), dtype=theano.config.floatX)
        for i, label in enumerate(_target):
            target[i, list(self.target_labels).index(label)] = 1.0
        return target
