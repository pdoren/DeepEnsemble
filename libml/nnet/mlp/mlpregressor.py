import theano.tensor as T
from .mlpmodel import MLPModel


class MLPRegressor(MLPModel):
    def __init__(self, n_input, n_hidden, n_output,
                 output_activation=T.nnet.sigmoid, hidden_activation=None):
        """ Multi-Layer Perceptron (MLP) for regression.

        A multilayer perceptron is a feedforward artificial neural network model
        that has one layer or more of hidden units and nonlinear activations.

        Parameters
        ----------
        n_input: int
            Number of input for MLP net.

        n_hidden: int or list
            Number of hidden Layers.

        n_output: int
            Number of output for MLP net.

        output_activation: theano.tensor
            Function of activation for output layer.

        hidden_activation: theno.tensor or list
            Functions of activation for hidden layers.

        """
        super(MLPRegressor, self).__init__(n_input=n_input, n_hidden=n_hidden, n_output=n_output,
                                           output_activation=output_activation, hidden_activation=hidden_activation,
                                           type_model='regressor')

    def output(self, _input):
        """ Output of MLP.

        Parameters
        ----------
        _input: theano.tensor.matrix
            Input sample.

        Returns
        -------
        theano.tensor.matrix
        Returns the prediction of MLP model.

        """
        for layer in self.layers:
            _input = layer.output(_input)
        return _input

    def translate_output(self, _output):
        """ Evaluating _output of MLP.

        Parameters
        ----------
        _output: theano.tensor.matrix
            Output MLP.

        Returns
        -------
        numpy.array
        Returns the same '_output' array but evaluated.

        """
        return _output.eval()
