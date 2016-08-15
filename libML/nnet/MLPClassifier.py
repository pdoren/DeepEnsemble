import theano
import theano.tensor as T
import numpy as np
from libML.nnet.MLPRegressor import MLPRegressor


class MLPClassifier(MLPRegressor):
    def __init__(self, n_input, n_hidden, target_labels=[],
                 output_activation=T.nnet.sigmoid, hidden_activation=None):

        self.N_classes = len(target_labels)
        self.target_labels = target_labels
        if self.N_classes == 0:
            raise ValueError("Incorrect labels target")

        super(MLPClassifier, self).__init__(n_input=n_input, n_hidden=n_hidden, n_output=self.N_classes,
                                            output_activation=output_activation, hidden_activation=hidden_activation)

        self.type_learner = "classifier"

    def predict(self, _input):
        output = super(MLPClassifier, self).output(_input)
        return self.translate_output(output)

    def translate_output(self, _output):
        return self.target_labels[T.argmax(_output, axis=1).eval()]

    def get_target(self, _target):
        target = np.zeros(shape=(len(_target), self.N_classes), dtype=theano.config.floatX)
        for i, label in enumerate(_target):
            target[i, list(self.target_labels).index(label)] = 1.0
        return target

    def fit(self, _input, _target):
        super(MLPClassifier, self).fit(_input, self.get_target(_target))
