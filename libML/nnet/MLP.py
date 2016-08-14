import theano.tensor as T
import libML.MLearn as MLearn
from libML.nnet.Layer import Layer


class MLP(MLearn):
    def __init__(self, n_input, n_hidden, n_output, output_activation=T.nnet.sigmoid, hidden_activation=None,
                 type_output="classifier"):
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
        super().__init__(type_output=type_output)
        self.layers = []
        self.N_output = n_output

        if hidden_activation is None:
            functions_activation = []
        else:
            functions_activation = hidden_activation

        while len(functions_activation) != len(n_hidden):
            functions_activation.append(T.tanh)  # default tanh

        # input layer
        self.layers.append(Layer(n_input, n_hidden[0], functions_activation[0]))

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
        Prediction of MLP

        :type _input: theano.tensor.dmatrix
        :param _input: input
        :return: direct output of MLP
        """
        for layer in self.layers:
            _input = layer.output(_input)
        return _input

    def predict_classifier(self, _input, threshold=0.5):
        return (self.predict_binary(_input, threshold) if self.N_output < 2
                else self.predict_multiclass(_input))

    def reset(self):
        """
        Reset weights of MLP
        """
        for layer in self.layers:
            layer.initialize_parameters()

    def predict_binary(self, _input, threshold=0.5):
        """
        Binary prediction with MLP (2 classes)

        :type _input: theano.tensor.dmatrix
        :param _input: input
        :type threshold: float
        :param threshold: Threshold for get to classification
        :return: Return true or false or number of class (1 or 0)
        """
        for layer in self.layers:
            _input = layer.output(_input)
        return _input > threshold

    def predict_multiclass(self, _input):
        """
        Multiclass prediction with MLP

        :type _input: theano.tensor.dmatrix
        :param _input: input
        :return: Return number of class
        """
        for layer in self.layers:
            _input = layer.output(_input)
        return T.argmax(_input, axis=1)
