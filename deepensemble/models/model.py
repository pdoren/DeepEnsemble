import theano.tensor as T
from theano import config, shared
import numpy as np
import pickle
from ..utils.utils_classifiers import *
from ..utils.metrics.classifiermetrics import *
from ..utils.metrics.regressionmetrics import *


class Model:
    """ Base class for models.

    Attributes
    ----------
    n_input : int
        Number of inputs of the model.

    n_output : int
        Number of output of the model.

    type_model : str
        Type of model: classifier or regressor.

    target_labels : numpy.array
        Labels of classes.

    params : list
        List of model's parameters.

    cost_function_list : list
        List for saving the cost functions.

    reg_function_list : list
        List for saving the regularization functions.

    score_function_list : list
        This is a list of function for compute a score to models, for classifier model is accuracy by default
         and for regressor model is RMS by default.

    update_function : theano.function
        This function allow to update the model's parameters.

    update_function_args
        Arguments of update function.

    name : str
        This model's name is useful to identify it later.

    _output : TensorVariable
        Output model (Theano).

    Parameters
    ----------
    target_labels: list or numpy.array
        Target labels.

    type_model : str, "classifier" by default
        Type of model: classifier or regressor.

    name : str, "model" by default
        Name of model.

    n_input : int
        Number of inputs of the model.

    n_output : int
        Number of output of the model.
    """

    # static variables
    model_input = T.matrix('model_input')  # Attribute for save input model.
    model_target = T.matrix('model_target')  # Attribute for save target model.
    batch_reg_ratio = T.scalar('batch_reg_ratio', dtype=config.floatX)  # Attribute related with regularization

    def __init__(self, target_labels, type_model, name="model", n_input=None, n_output=None):
        self.n_input = n_input
        self.n_output = n_output
        self.type_model = type_model
        self.target_labels = np.array(target_labels)

        self.params = []
        self.cost_function_list = []
        self.reg_function_list = []
        self.score_function_list = []
        self.update_function = None
        self.update_function_args = None

        self.minibatch_train_eval = None
        self.minibatch_test_eval = None
        self.scan_minibatch = None
        self.share_data_input_train = shared(np.zeros((1, 1), dtype=config.floatX))
        self.share_data_input_test = shared(np.zeros((1, 1), dtype=config.floatX))
        self.share_data_target_train = shared(np.zeros((1, 1), dtype=config.floatX))
        self.share_data_target_test = shared(np.zeros((1, 1), dtype=config.floatX))

        self.name = name

        self._output = None
        self._error = None

    def error(self, _input, _target):
        """ Compute the error prediction of model.

        Parameters
        ----------
        _input : theano.tensor.matrix or numpy.array
            Input sample.

        _target : theano.tensor.matrix or numpy.array
            Target sample.

        Returns
        -------
        theano.tensor.matrix or numpy.array
            Returns error of model prediction.

        """
        if _input == self.model_input and _target == self.model_target:
            if self._error is None:
                self._error = self.output(_input) - _target
            return self._error
        else:
            return self.output(_input) - _target

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
                self.type_model is other.type_model) and (
                   self.type_model is "regressor" or (list(self.target_labels) == list(other.target_labels)))
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
        self._output = None

    def output(self, _input):
        """ Output model

        Parameters
        ----------
        _input : theano.tensor.matrix
            Input sample.

        Returns
        -------
        theano.tensor.matrix or numpy.array
            Prediction of model.
        """
        raise NotImplementedError

    def predict(self, _input):
        """ Compute the prediction of model.

        Parameters
        ----------
        _input : theano.tensor.matrix or numpy.array
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

    def minibatch_eval(self, n_input, batch_size=32, train=True):
        """ Evaluate cost and score in mini batch.

        Parameters
        ----------
        n_input: int
            Dimension Input.

        batch_size: int
            Size of batch.

        train: bool
            Flag for knowing if the evaluation of batch is for training or testing.

        Returns
        -------
        numpy.array
            Returns evaluation cost and score in mini batch.
        """
        if n_input > batch_size:
            t_data = []
            for (start, end) in zip(range(0, n_input, batch_size), range(batch_size, n_input, batch_size)):
                r = (end - start) / n_input
                if train:
                    t_data.append(self.minibatch_train_eval(start, end, r))
                else:
                    t_data.append(self.minibatch_test_eval(start, end, r))
            return np.mean(t_data, axis=0)
        else:
            if train:
                return self.minibatch_train_eval(0, n_input, 1.0)
            else:
                return self.minibatch_test_eval(0, n_input, 1.0)

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
        self.set_default_score()

    def prepare_data(self, _input, _target):
        target_train = _target
        input_train = _input
        if self.type_model is 'classifier':
            target_train = translate_target(_target=_target, n_classes=self.n_output, target_labels=self.target_labels)

        self.share_data_input_train.set_value(input_train)
        self.share_data_target_train.set_value(target_train)

    def set_default_score(self):
        """ Setting default score in model.
        """
        if self.type_model is "classifier":
            self.score_function_list.append(
                score_accuracy(translate_output(self.output(self.model_input), self.n_output), self.model_target))
        else:
            self.score_function_list.append(score_rms(self.output(self.model_input), self.model_target))

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

    def set_update(self, fun_update, **kwargs):
        """ Adds an extra item in the cost function.

        Parameters
        ----------
        fun_update : theano.function
            Function of update parameters of models.

        **kwargs
            Extra parameters of update function.
        """
        self.update_function = fun_update
        self.update_function_args = kwargs

    def get_cost_functions(self):
        """ Gets cost function of model.

        Returns
        -------
        theano.tensor.TensorVariable
            Returns cost model include regularization.
        """
        cost = 0.0
        for c in self.cost_function_list:
            cost += c

        for r in self.reg_function_list:
            cost += r

        return cost

    def get_score_functions(self):
        """ Gets score function of model.

        Returns
        -------
        theano.tensor.TensorVariable
            Returns score model.
        """
        score = 0.0
        for s in self.score_function_list:
            score += s

        return score

    def get_update_function(self, cost):
        """ Gets dict for update model parameters.

        Parameters
        ----------
        cost : theano.tensor.TensorVariable
            Cost function.

        Returns
        -------
        OrderedDict
            A dictionary mapping each parameter to its update expression.
        """
        return self.update_function(cost, self.params, **self.update_function_args)

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
