import theano.tensor as T
from libml.model import Model
from libml.nnet.layer import Layer


class MLPModel(Model):

    def __init__(self, n_input, n_hidden, n_output,
                 output_activation=T.nnet.sigmoid, hidden_activation=None, type_model="regressor"):
        """ Base class for Multi-Layer Perceptron (MLP).

        A base class for multilayer perceptron is a feedforward artificial neural network model
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

        type_model: str
            Type of MLP model: classifier or regressor.

        """
        super(MLPModel, self).__init__(n_input=n_input, n_output=n_output, type_model=type_model)

        self.layers = []
        default_hidden_fun = T.tanh  # default activation function

        if hidden_activation is None:
            functions_activation = []
        elif type(hidden_activation) != list:
            functions_activation = []
            default_hidden_fun = hidden_activation
        else:
            functions_activation = hidden_activation

        while len(functions_activation) != len(n_hidden):
            functions_activation.append(default_hidden_fun)

        if type(n_hidden) == int:
            n_hidden_aux = n_hidden
            n_hidden = []
            n_hidden[0] = n_hidden_aux

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
        """ Get output of MLP model.

        Parameters
        ----------
        _input: theano.tensor.matrix
            Input sample.

        Returns
        -------
        theano.tensor.matrix
        Returns the output or prediction of MLP model.

        """
        raise NotImplementedError

    def translate_output(self, _output):
        """ Translate '_output' according to labels in case of the classifier MLP,
        for regressor MLP return '_output' without changes.

        Parameters
        ----------
        _output: theano.tensor.matrix
            Prediction or output of model.

        Returns
        -------
        numpy.array
        It will depend of the type model:

         - Classifier models: the translation of '_output' according to target labels.
         - Regressor models: the same '_output' array (evaluated).

        """
        raise NotImplementedError

    def reset(self):
        """ Reset weights of MLP
        """
        for layer in self.layers:
            layer.initialize_parameters()

    def sqr_L2(self, lamb):
        """ Compute regularization square L2.

        Parameters
        ----------
        lamb: float or double
            Ratio regularization L2.

        Returns
        -------
        float or double
        Return regularization square L2.

        """
        sqrL2W = 0.0
        for layer in self.layers:
            sqrL2W += T.sum(T.power(layer.W, 2.0))
        return lamb * sqrL2W

    def L1(self, lamb):
        """ Compute regularization L1.

        Parameters
        ----------
        lamb float or double
            Ratio regularization L1.

        Returns
        -------
        float or double
        Return regularization L1.

        """
        L1W = 0.0
        for layer in self.layers:
            L1W += T.sum(T.abs_(layer.W))
        return lamb * L1W