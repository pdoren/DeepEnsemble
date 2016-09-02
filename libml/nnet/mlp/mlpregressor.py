import theano.tensor as T
from libml.model import Model
from libml.nnet.layer import Layer


class MLPRegressor(Model):
    def __init__(self, n_input, n_hidden, n_output,
                 output_activation=T.nnet.sigmoid, hidden_activation=None):
        """ Multi-Layer Perceptron (MLP) for regression

        A multilayer perceptron is a feedforward artificial neural network model
        that has one layer or more of hidden units and nonlinear activations.

        Parameters
        ----------
        n_input
        n_hidden
        n_output
        output_activation
        hidden_activation
        """
        super(MLPRegressor, self).__init__(n_input=n_input, n_hidden=n_hidden, n_output=n_output,
                                           output_activation=output_activation, hidden_activation=hidden_activation,
                                           type_model='regressor')

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
        return _output.eval()