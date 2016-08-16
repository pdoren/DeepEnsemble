import theano.tensor as T
from libml.model import Model
from libml.nnet.layer import Layer


class MLPRegressor(Model):
    def __init__(self, n_input, n_hidden, n_output,
                 output_activation=T.nnet.sigmoid, hidden_activation=None):
        """ Multi-Layer Perceptron (MLP)

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
        super(MLPRegressor, self).__init__(n_input=n_input, n_output=n_output, type_model='regressor')

        self.layers = []

        if hidden_activation is None:
            functions_activation = []
        else:
            functions_activation = hidden_activation

        while len(functions_activation) != len(n_hidden):
            functions_activation.append(T.tanh)  # default activation function

        # input layer
        self.layers.append(Layer(self.n_input, n_hidden[0], functions_activation[0]))

        # hidden layers
        for i in range(1, len(n_hidden)):
            self.layers.append(Layer(n_hidden[i - 1], n_hidden[i], functions_activation[i]))

        # output layer
        self.layers.append(Layer(n_hidden[len(n_hidden) - 1], n_output, output_activation))

        self.params = []
        for i in range(0, len(self.layers)):
            self.params += self.layers[i].params

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

    def reset(self):
        """ Reset weights of MLP

        Returns
        -------

        """
        for layer in self.layers:
            layer.initialize_parameters()

    def sqr_L2(self, lamb):
        """ Compute regularization square L2

        Parameters
        ----------
        lamb

        Returns
        -------

        """
        sqrL2W = 0.0
        for layer in self.layers:
            sqrL2W += T.sum(T.power(layer.W, 2.0))
        return lamb * sqrL2W

    def L1(self, lamb):
        """ Compute regularization L1

        Parameters
        ----------
        lamb

        Returns
        -------

        """
        L1W = 0.0
        for layer in self.layers:
            L1W += T.sum(T.abs_(layer.W))
        return lamb * L1W
