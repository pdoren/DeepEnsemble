import theano.tensor as T
import pickle


class Model:
    def __init__(self, n_input, n_output, type_model='classifier'):
        """ Base class for models.

        Parameters
        ----------
        n_input: int
            Number of inputs of the model.

        n_output: int
            Number of output of the model.

        type_model: str
            Type of model: classifier or regressor.

        """
        self.n_input = n_input
        self.n_output = n_output
        self.type_model = type_model
        self.params = []
        self.target_labels = []

    def __eq__(self, other):
        """ Evaluate if 'other' model has the same form. The items for the comparison are:

        - Number of input
        - Number of output
        - Type of model: classifier or regressor
        - Target labels

        Parameters
        ----------
        other: Model
            Model for compare oneself.

        Returns
        -------
        Return True if the 'other' model has the same form, False otherwise.

        """
        if isinstance(other, Model):
            return (self.n_input == other.n_input) and (self.n_output == other.n_output) and (
                self.type_model is other.type_model) and (list(self.target_labels) == list(other.target_labels))
        else:
            return False

    def __ne__(self, other):
        """ Evaluate if 'other' model doesn't have the same form. The items for the comparison are:

        - Number of input
        - Number of output
        - Type of model: classifier or regressor
        - Target labels

        Parameters
        ----------
        other: Model
            Model for compare oneself.

        Returns
        -------
        Return True if the 'other' model doesn't have the same form, False otherwise.

        """
        return not self.__eq__(other)

    def reset(self):
        """ Reset params
        """
        raise NotImplementedError

    def translate_target(self, _target):
        """ Translate values of '_target' for the classifiers using information of the labels,
         in case the regressor models '_target' doesn't change.

        Parameters
        ----------
        _target
            Target or output of model.

        Returns
        -------
        numpy.array
        For the classifiers return the '_target' translated according to labels,
        in case the regressor models return the same '_target'.
        """
        return _target

    def output(self, _input):
        """ Output model

        Parameters
        ----------
        _input: theano.tensor.matrix
            Input sample.

        Returns
        -------
        theano.tensor.matrix
        Prediction of model.

        """
        raise NotImplementedError

    def translate_output(self, _output):
        """ Translate '_output' according to labels in case of the classifier model,
        for regressor model return '_output' without changes.

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

    def predict(self, _input):
        """ Compute the prediction of model.

        Parameters
        ----------
        _input: theano.tensor.matrix
            Input sample.

        Returns
        -------
        numpy.array
        Return the prediction of model.

        """
        output = self.output(_input)
        return self.translate_output(output)

    def get_cost_function(self, cost, _input, _target, kernel_size):
        """ Get cost function implemented.

        Parameters
        ----------
        cost: str
            Type of cost.

        _input: theano.tensor.matrix
            Input sample.

        _target: theano.tensor.matrix
            Target sample.

        kernel_size: float or double
            Kernel size for some cost functions.

        Raises
        ------
        If it isn't implemented cost function it is generated an error.

        Returns
        -------
        theano.tensor
        Return cost function.

        """
        if cost == "MSE":
            return self.mse(_input, _target)
        elif cost == "MCC":
            return self.mcc(_input, _target, kernel_size)
        elif cost == "MEE":
            return self.mee(_input, _target, kernel_size)
        elif cost == "NLL":
            return self.neg_log_likelihood(_input, _target)
        else:
            raise ValueError("Incorrect cost function, options are MSE, MCC, MEE or NLL")

    def mse(self, _input, _target):
        """ Compute MSE error between target and output prediction.

        Parameters
        ----------
        _input: theano.tensor.matrix
            Input sample.

        _target: theano.tensor.matrix
            Target sample.

        Returns
        -------
        theano.config.floatX
        Return MSE error.

        """
        e = self.output(_input) - _target
        return T.mean(T.power(e, 2.0))

    def mcc(self, _input, _target, s):
        """ Compute the MCC.

        Parameters
        ----------
        _input: theano.tensor.matrix
        Input sample.

        _target: theano.tensor.matrix
        Target sample.

        s: float or double
        Kernel's Parameter.

        Returns
        -------
        theano.config.floatX
        Return MCC.

        """
        e = self.output(_input) - _target
        return -T.mean(T.exp(-0.5 * T.power(e, 2.0) / s ** 2))

    def mee(self, _input, _target, s):
        """ Compute the MEE.

        Parameters
        ----------
         _input: theano.tensor.matrix
            Input sample.

        _target: theano.tensor.matrix
            Target sample.

        s: float or double
            Kernel's Parameter.

        Returns
        -------
        theano.config.floatX
        Return MEE.

        """
        e = self.output(_input) - _target
        de = T.tile(e, (e.shape[0], 1, 1))
        de = de - T.transpose(de, axes=(1, 0, 2))
        return -T.log(T.mean(T.exp(-0.5 * T.power(de, 2.0) / s ** 2)))

    def neg_log_likelihood(self, _input, _target):
        """ Compute the negative means of errors between target and output prediction

        Parameters
        ----------
         _input: theano.tensor.matrix
            Input sample.

        _target: theano.tensor.matrix
            Target sample.

        Returns
        -------
        theano.config.floatX
        Return negative logarithm likelihood.

        """
        labels = T.argmax(_target, axis=1)
        return -T.mean(T.log(T.power(self.output(_input), 2.0))[T.arange(_target.shape[0]), labels])

    def load(self, filename):
        """ Load model from file.

        Parameters
        ----------
        filename: str
            Path of file where recovery data of model.

        """
        file_model = open(filename, 'rb')
        tmp_dict = pickle.load(file_model)
        file_model.close()
        self.__dict__.update(tmp_dict)

    def save(self, filename):
        """ Save data to file.

        Parameters
        ----------
        filename: str
            Path of file where storage data of model.

        """
        file_model = open(filename, 'wb')
        pickle.dump(self.__dict__, file_model, 2)
        file_model.close()
