import pickle
import numpy as np
import theano.tensor as T
from sklearn import cross_validation

from theano import config, shared
from ..metrics import *
from ..utils import *

__all__ = ['Model']


class Model(object):
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

    _params : list
        List of model's parameters.

    _cost_function_list : list
        List for saving the cost functions.

    _reg_function_list : list
        List for saving the regularization functions.

    _score_function_list : list
        This is a list of function for compute a score to models, for classifier model is accuracy by default
        and for regressor model is RMS by default.

    _update_function : theano.function
        This function allow to update the model's parameters.

    _update_function_args
        Arguments of update function.

    name : str
        This model's name is useful to identify it later.

    _output : theano.tensor.TensorVariable
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
        self._n_input = n_input
        self._n_output = n_output
        self._type_model = type_model
        self._target_labels = np.array(target_labels)
        self._params = []

        self._cost_function_list = []
        self._reg_function_list = []
        self._score_function_list = []
        self._update_function = None
        self._update_function_args = None

        self._minibatch_train_eval = None
        self._minibatch_test_eval = None
        self._share_data_input_train = shared(np.zeros((1, 1), dtype=config.floatX))
        self._share_data_input_test = shared(np.zeros((1, 1), dtype=config.floatX))
        self._share_data_target_train = shared(np.zeros((1, 1), dtype=config.floatX))
        self._share_data_target_test = shared(np.zeros((1, 1), dtype=config.floatX))

        self._name = name

        self._output = None
        self._error = None

    def get_name(self):
        """ Getter name.

        Returns
        -------
        str
            Returns name of model.
        """
        return self._name

    def get_params(self):
        """ Getter model parameters.

        Returns
        -------
        theano.shared
            Returns model parameters.
        """
        return self._params

    def get_score_function_list(self):
        """ Getter list score.

        Returns
        -------
        list[theano.Op]
            Returns list of score functions.
        """
        return self._score_function_list

    def get_target_labels(self):
        """ Getter target labels.

        Returns
        -------
        list
            Returns one list with target labels of this model.
        """
        return self._target_labels

    def get_new_metric(self):
        """ Get metrics for respective model.

        .. note:: This function is necessary implemented for uses FactoryMetrics.

        See Also
        --------
        FactoryMetrics
        """
        raise NotImplementedError

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
            return (self._n_input == other._n_input) and (self._n_output == other._n_output) and (
                self._type_model is other._type_model) and (
                       self._type_model == "regressor" or (list(self._target_labels) == list(other._target_labels)))
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
        if self._type_model is 'regressor':
            return output.eval()
        else:
            return self._target_labels[get_index_label_classes(output)]

    def batch_eval(self, n_input, batch_size=32, train=True):
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
                    t_data.append(self._minibatch_train_eval(start, end, r))
                else:
                    t_data.append(self._minibatch_test_eval(start, end, r))
            return np.mean(t_data, axis=0)
        else:
            if train:
                return self._minibatch_train_eval(0, n_input, 1.0)
            else:
                return self._minibatch_test_eval(0, n_input, 1.0)

    def fit(self, _input, _target, max_epoch=100, batch_size=32, early_stop=True,
            improvement_threshold=0.995, minibatch=True):
        """ Function for training sequential model.

        Parameters
        ----------
        _input : theano.tensor.matrix
            Input training samples.

        _target : theano.tensor.matrix
            Target training samples.

        max_epoch : int, 100 by default
            Number of epoch for training.

        batch_size : int, 32 by default
            Size of batch.

        early_stop : bool, True by default
            Flag for enabled early stop.

        improvement_threshold : int, 0.995 by default

        minibatch : bool
            Flag for indicate training with minibatch or not.

        Returns
        -------
        numpy.array[float]
            Returns training cost for each batch.
        """
        # create a specific metric
        metric_model = FactoryMetrics().get_metric(self)

        # save data in shared variables
        self.prepare_data(_input, _target)
        n_train = self._share_data_target_train.get_value(borrow=True).shape[0]
        n_test = self._share_data_target_test.get_value(borrow=True).shape[0]

        # parameters early stopping
        best_params = None
        best_validation_cost = np.inf
        patience = max(max_epoch * n_train // 5, 5000)
        validation_jump = min(patience // 100, max_epoch // 50)
        patience_increase = 2

        for epoch, _ in enumerate(Logger().progressbar_training(max_epoch, self)):

            if minibatch:  # Train minibatches
                t_data_train = self.batch_eval(n_input=n_train, batch_size=batch_size, train=True)
            else:
                t_data_train = self.batch_eval(n_input=n_train, batch_size=n_train, train=True)

            metric_model.append_data(t_data_train, epoch, type_set_data="train")

            iteration = epoch * n_train

            if epoch % validation_jump == 0:
                # Evaluate test set
                t_data_test = self.batch_eval(n_input=n_test, batch_size=n_test, train=False)
                metric_model.append_data(t_data_test, epoch, type_set_data="test")
                validation_cost = t_data_test[0]

                if validation_cost < best_validation_cost:

                    if validation_cost < best_validation_cost * improvement_threshold:
                        patience = max(patience,  iteration * patience_increase)

                    best_params = self._save_params()
                    best_validation_cost = validation_cost

            if early_stop and patience <= iteration:
                Logger().print()
                break

        if best_params is not None:
            self._load_params(best_params)

        return metric_model

    def _save_params(self):
        return [i.get_value() for i in self.get_params()]

    def _load_params(self, params):
        for p, value in zip(self.get_params(), params):
            p.set_value(value)

    def _compile(self, fast=True, **kwargs):
        """ Prepare training.

        Parameters
        ----------
        fast : bool
            Compiling cost and regularization items without separating them.
        """
        raise NotImplementedError

    def compile(self, fast=True, **kwargs):
        """ Prepare training.
        """
        Logger().start_measure_time("Start Compile %s" % self._name)

        if len(self._score_function_list) <= 0:
            self.set_default_score()

        self._compile(fast=fast, **kwargs)

        Logger().stop_measure_time()

    def prepare_data(self, _input, _target, test_size=0.3):
        """

        Parameters
        ----------
        _input
        _target
        test_size
        """
        input_train, input_test, target_train, target_test = \
            cross_validation.train_test_split(_input, _target, test_size=test_size, random_state=0)

        if self._type_model is 'classifier':
            target_train = translate_target(_target=target_train, n_classes=self._n_output,
                                            target_labels=self._target_labels)
            target_test = translate_target(_target=target_test, n_classes=self._n_output,
                                           target_labels=self._target_labels)

        self._share_data_input_train.set_value(input_train)
        self._share_data_target_train.set_value(target_train)
        self._share_data_input_test.set_value(input_test)
        self._share_data_target_test.set_value(target_test)

    def set_default_score(self):
        """ Setting default score in model.
        """
        if self._type_model == "classifier":
            self._score_function_list.append(
                score_accuracy(translate_output(self.output(self.model_input), self._n_output), self.model_target))
        else:
            self._score_function_list.append(score_rms(self.output(self.model_input), self.model_target))

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
        self._cost_function_list.append(c)

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
        self._reg_function_list.append(c)

    def set_update(self, fun_update, **kwargs):
        """ Adds an extra item in the cost function.

        Parameters
        ----------
        fun_update : theano.function
            Function of update parameters of models.

        **kwargs
            Extra parameters of update function.
        """
        self._update_function = fun_update
        self._update_function_args = kwargs

    def get_cost_functions(self):
        """ Gets cost function of model.

        Returns
        -------
        theano.tensor.TensorVariable
            Returns cost model include regularization.
        """
        cost = 0.0
        for c in self._cost_function_list:
            cost += c

        for r in self._reg_function_list:
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
        for s in self._score_function_list:
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
        return self._update_function(cost, self._params, **self._update_function_args)

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
