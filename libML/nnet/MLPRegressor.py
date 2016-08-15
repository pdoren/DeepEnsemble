import theano.tensor as T
from libML.MLearn import MLearn
from libML.nnet.Layer import Layer


class MLPRegressor(MLearn):
    def __init__(self, n_input, n_hidden, n_output, output_activation=T.nnet.sigmoid, hidden_activation=None):
        """
        Multi-Layer Perceptron (MLP)

        A multilayer perceptron is a feedforward artificial neural network model
        that has one layer or more of hidden units and nonlinear activations.

        :type n_input: int
        :param n_input: Dimensionality of input
        :type n_hidden: array
        :param n_hidden: Array with number of each hidden layers
        :type n_output: int
        :param n_output: Dimensionality of output
        :type output_activation: tensor.Op or function
        :param output_activation: Non linearity to be applied in the output Layers
        :type hidden_activation: array
        :param hidden_activation: Non linearity to be applied in the hidden Layers
        """
        super(MLPRegressor, self).__init__(n_input=n_input, n_output=n_output, type_learner='regressor')

        self.layers = []

        if hidden_activation is None:
            functions_activation = []
        else:
            functions_activation = hidden_activation

        while len(functions_activation) != len(n_hidden):
            functions_activation.append(T.tanh)  # default tanh

        # input layer
        self.layers.append(Layer(self.N_input, n_hidden[0], functions_activation[0]))

        # hidden layers
        for i in range(1, len(n_hidden)):
            self.layers.append(Layer(n_hidden[i - 1], n_hidden[i], functions_activation[i]))

        # output layer
        self.layers.append(Layer(n_hidden[len(n_hidden) - 1], n_output, output_activation))

        self.params = []
        for i in range(0, len(self.layers)):
            self.params += self.layers[i].params

    def sqr_L2(self, lamb):
        """
        Compute regularization square L2

        :type lamb: float
        :param lamb: Ratio for regularization
        :return: square L2-norm's weight of MLP
        """
        sqrL2W = 0.0
        for layer in self.layers:
            sqrL2W += T.sum(T.power(layer.W, 2.0))
        return lamb * sqrL2W

    def L1(self, lamb):
        """
        Compute regularization L1

        :type lamb: float
        :param lamb: Ratio for regularization
        :return: L1-norm's weight of MLP
        """
        L1W = 0.0
        for layer in self.layers:
            L1W += T.sum(T.abs_(layer.W))
        return lamb * L1W

    def output(self, _input):
        """
        Output of MLP

        :type _input: theano.tensor.dmatrix
        :param _input: input
        :return: direct output of MLP
        """
        for layer in self.layers:
            _input = layer.output(_input)
        return _input

    def predict(self, _input):
        """
        Prediction of MLP

        :type _input: theano.tensor.dmatrix
        :param _input: input
        :return: direct output of MLP
        """
        return self.translate_output(self.output(_input))

    def translate_output(self, _output):
        return self.output.eval()

    def reset(self):
        """
        Reset weights of MLP
        """
        for layer in self.layers:
            layer.initialize_parameters()
