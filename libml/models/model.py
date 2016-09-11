import theano.tensor as T
from theano import config
import numpy as np
import pickle
from libml.utils.utils_classifiers import *
from libml.utils.metrics.classifiermetrics import *
from libml.utils.metrics.regressionmetrics import *


class Model:
    def __init__(self, n_input=None, n_output=None, target_labels=None, type_model='classifier', name="model"):
        """ Base class for models.

        Parameters
        ----------
        n_input : int
            Number of inputs of the model.

        n_output : int
            Number of output of the model.

        target_labels: list or numpy.array
            Target labels.

        type_model : str, "classifier" by default
            Type of model: classifier or regressor.

        name : str, "model" by default
            Name of model.
        """
        self.n_input = n_input
        self.n_output = n_output
        self.type_model = type_model
        self.target_labels = np.array(target_labels)

        self.model_input = T.matrix('model_input')
        self.model_target = T.matrix('model_target')

        self.params = []
        self.cost_function = None
        self.cost_function_list = []
        self.reg_function = None
        self.reg_function_list = []
        self.updates = None
        self.score = None

        self.name = name

        self.batch_reg_ratio = T.scalar('batch_reg_ratio', dtype=config.floatX)

    def __eq__(self, other):
        """ Evaluate if 'other' model has the same form. The items for the comparison are:

        - Number of input
        - Number of output
        - Type of model: classifier or regressor
        - Target labels

        Parameters
        ----------
        other : Model
            Model for compare oneself.

        Returns
        -------
        bool
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
        other : Model
            Model for compare oneself.

        Returns
        -------
        bool
            Return True if the 'other' model doesn't have the same form, False otherwise.
        """
        return not self.__eq__(other)

    def reset(self):
        """ Reset params
        """
        raise NotImplementedError

    def output(self, _input):
        """ Output model

        Parameters
        ----------
        _input : theano.tensor.matrix
            Input sample.

        Returns
        -------
        theano.tensor.matrix
            Prediction of model.
        """
        raise NotImplementedError

    def predict(self, _input):
        """ Compute the prediction of model.

        Parameters
        ----------
        _input : theano.tensor.matrix
            Input sample.

        Returns
        -------
        numpy.array
            Return the prediction of model.
        """
        output = self.output(_input)
        if self.type_model is 'regressor':
            return output.eval()
        else:
            return self.target_labels[get_index_label_classes(output)]

    def fit(self, _input, _target, max_epoch, validation_jump, **kwargs):
        """ Training model.

        Parameters
        ----------
        _input : theano.tensor.matrix
            Training Input sample.

        _target : theano.tensor.matrix
            Training Target sample.

        max_epoch: int
            Number of epoch for training.

        validation_jump: int
            Number of times until doing validation jump.

        kwargs
            Other parameters.

        Returns
        -------
        numpy.array[float]
            Returns training cost for each batch.
        """
        raise NotImplementedError

    def compile(self):
        """ Prepare training.
        """
        if self.type_model is "classifier":
            self.score = score_accuracy(translate_output(self.output(self.model_input), self.n_output),
                                        self.model_target)
        else:
            self.score = score_rms(self.output(self.model_input), self.model_target)

    def append_cost(self, fun_cost, **kwargs):
        """ Adds an extra item in the cost function.

        Parameters
        ----------
        fun_cost : theano.function
            Function of cost

        **kwargs
            Extra parameters of function cost.
        """
        c = fun_cost(model=self, _input=self.model_input, _target=self.model_target, **kwargs)
        self.cost_function_list.append(c)
        if self.cost_function is None:
            self.cost_function = c
        else:
            self.cost_function += c

    def append_reg(self, fun_reg, **kwargs):
        """ Adds an extra item in the cost function.

        Parameters
        ----------
        fun_reg : theano.function
            Function of regularization

        **kwargs
            Extra parameters of regularization function.
        """
        c = fun_reg(model=self, batch_reg_ratio=self.batch_reg_ratio, **kwargs)
        self.reg_function_list.append(c)
        if self.reg_function is None:
            self.reg_function = c
        else:
            self.reg_function += c

    def set_update(self, fun_update, **kwargs):
        """ Adds an extra item in the cost function.

        Parameters
        ----------
        fun_update : theano.function
            Function of update parameters of models.

        **kwargs
            Extra parameters of update function.
        """
        self.updates = fun_update(self.cost_function, self.params, **kwargs)

    def load(self, filename):
        """ Load model from file.

        Parameters
        ----------
        filename : str
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
        filename : str
            Path of file where storage data of model.
        """
        file_model = open(filename, 'wb')
        pickle.dump(self.__dict__, file_model, 2)
        file_model.close()
