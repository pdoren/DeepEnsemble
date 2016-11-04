import inspect

import numpy as np
import theano.tensor as T
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from theano import config, function

from ..metrics import FactoryMetrics
from ..utils import Logger, score_accuracy, score_rms
from ..utils.serializable import Serializable
from ..utils.update_functions import sgd
from ..utils.utils_classifiers import *

__all__ = ['Model']


class Model(Serializable):
    """ Base class for models.

    Attributes
    ----------
    __input_shape : tuple
        Shape of inputs of the model.

    __output_shape : tuple
        Shape of output of the model.

    type_model : str
        Type of model: classifier or regressor.

    target_labels : numpy.array
        Labels of classes.

    _params : list
        List of model's parameters.

    _cost_function_list : dict
        List for saving the cost functions.

    _reg_function_list : list
        List for saving the regularization functions.

    _score_function_list : dict
        This is a list of function for compute a score to models, for classifier model is accuracy by default
        and for regressor model is RMS by default.

    _update_function : tuple
        This function allow to update the model's parameters.

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

    input_shape : tuple[]
        Number of inputs of the model.

    output_shape : tuple[]
        Number of output of the model.
    """

    def __init__(self, target_labels, type_model, name="model", input_shape=None, output_shape=None):
        super(Model, self).__init__()

        self._model_input = None  # input model.
        self._model_target = None  # target model.
        self._batch_reg_ratio = T.fscalar('batch_reg_ratio')  # related with regularization

        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__type_model = type_model
        self.__target_labels = np.array(target_labels)
        self._params = []

        # Default score
        self._score_function_list = {'list': [], 'changed': True, 'result': []}

        if self.is_classifier():
            self.append_score(score_accuracy, 'Accuracy')
        else:
            self.append_score(score_rms, 'Root Mean Square')

        self._cost_function_list = {'list': [], 'changed': True, 'result': []}
        self._reg_function_list = []

        self._update_function = None
        self.set_update(sgd, name='SGD', learning_rate=0.1)  # default training algorithm

        self._train_eval = None
        self._valid_eval = None
        self._data_train = {'input': None, 'target': None}
        self._data_valid = {'input': None, 'target': None}

        self._name = name

        self._output = {'prob': {'changed': True, 'result': None},
                        'crisp': {'changed': True, 'result': None}}
        self._error = None
        self._labels_result_train = None

        self._current_data_train = None
        self._current_data_valid = None
        self.__binary_classification = False
        self._info_model = {'info': 'Nothing.', 'comment': None}
        self._is_compiled = False
        self._is_fast_compile = False

    def _define_input(self):
        """ Generate a shared variable for input.

        Returns
        -------
        None
        """
        tmp_data = [False] * self.get_dim_input()

        self._model_input = T.TensorType(config.floatX, tmp_data)('model_input')

    def _define_output(self):
        """ Generate a shared variable for output.

        Returns
        -------
        None
        """
        tmp_data = [False] * self.get_dim_output()

        self._model_target = T.TensorType(config.floatX, tmp_data)('model_target')

    def is_binary_classification(self):
        """ Gets True if this model is a binary classifier, False otherwise.

        Returns
        -------
        bool
            Returns True if this model is a binary classifier, False otherwise.
        """
        return self.__binary_classification

    def is_classifier(self):
        """ Asks if the model is a classifier.

        Returns
        -------
        bool
            Return True if the model is a classifier, False otherwise.
        """
        return self.__type_model == "classifier"

    def is_compiled(self):
        """ Indicate if the model was compiled.

        Returns
        -------
        bool
            Returns True if the model was compiled, False otherwise.
        """
        return self._is_compiled

    def is_fast_compiled(self):
        """ Indicate if the model was compiled in fast mode.

        Returns
        -------
        bool
            Returns True if the model was compiled in fast mode, False otherwise.
        """
        return self._is_fast_compile

    def is_multi_label(self):
        """ Indicate if this model is a multi-class classifier model.

        Returns
        -------
        bool
            Returns True if the number of classes is great than 2, False in otherwise.
        """
        return len(self.__target_labels) > 2

    # noinspection PyNoneFunctionAssignment
    def get_info(self):
        """ Gets model info.

        Returns
        -------
        str
            Returns info.
        """
        self.__generate_info()
        if self._info_model['comment'] is None:
            info_model = self._info_model['info']
        else:
            info_model = self._info_model['comment'] + '\n' + self._info_model['info']
        size = max(len(self.get_name()), max([len(i) for i in info_model.splitlines()]))
        line = '-' * size + '\n'
        extra = self._get_extra_info()
        if extra is None:
            return line + self.get_name() + '\n' + line + info_model + line
        else:
            # noinspection PyTypeChecker
            return line + self.get_name() + '\n' + line + info_model + line + extra + line

    def append_comment(self, comment):
        """ Set model info.

        Parameters
        ----------
        comment : str
            Information for model.
        """
        self._info_model['comment'] = comment

    def get_result_labels(self):
        """ Gets list with labels of data training.

        Returns
        -------
        list[]
            Returns list of labels of data training.
        """
        return self._labels_result_train

    def get_input_shape(self):
        """ Gets input shape.

        Returns
        -------
        tuple
            Returns input shape.
        """
        return self.__input_shape

    def set_input_shape(self, shape):
        """ Set input shape.

        Parameters
        ----------
        shape : tuple
            Input shape.
        """
        self.__input_shape = shape

    def get_output_shape(self):
        """ Gets output shape.

        Returns
        -------
        tuple
            Returns output shape.
        """
        return self.__output_shape

    def set_output_shape(self, shape):
        """ Set output shape.

        Parameters
        ----------
        shape : tuple
            Output shape.
        """
        self.__output_shape = shape

    def get_type_model(self):
        """ Gets type of model.

        Returns
        -------
        str
            Returns type od model (regressor or classifier).
        """
        return self.__type_model

    def copy_kind_of_model(self, model):
        """ Copy important data from model.

        This data is:
            - Input shape.
            - Output Shape.
            - Type of model.
            - Target labels.

        Parameters
        ----------
        model : Model
            Source data model.
        """
        self.set_input_shape(model.get_input_shape())
        self.set_output_shape(model.get_output_shape())
        self.__type_model = model.get_type_model()
        self.__target_labels = model.get_target_labels()

    def _copy_input(self, model):
        """ Copy input.
        """
        # noinspection PyProtectedMember
        self._model_input = model._model_input

    def _copy_output(self, model):
        """ Copy output.
        """
        # noinspection PyProtectedMember
        self._model_target = model._model_target

    def _copy_batch_ratio(self, model):
        """ Copy output.
        """
        # noinspection PyProtectedMember
        self._batch_reg_ratio = model._batch_reg_ratio

    def get_dim_input(self):
        """ Gets input dimension.

        Returns
        -------
        int
            Returns dimension output.
        """
        return len(self.__input_shape)

    def get_dim_output(self):
        """ Gets output dimension.

        Returns
        -------
        int
            Returns dimension output.
        """
        return len(self.__output_shape)

    def get_fan_in(self):
        """ Gets number of input.

        Returns
        -------
        int
            Returns number of input.
        """
        input_shape = list(self.__input_shape)
        input_shape[input_shape is None] = 1
        return int(np.prod(input_shape))

    def get_fan_out(self):
        """ Gets number of output.

        Returns
        -------
        int
            Returns number of output.
        """
        output_shape = list(self.__output_shape)
        output_shape[output_shape is None] = 1
        return int(np.prod(output_shape))

    def get_test_cost(self):
        """ Gets current testing cost.

        Returns
        -------
        float
            Returns testing cost.
        """
        if self._current_data_valid is None:
            return 0.0
        else:
            return self._current_data_valid[1]

    def get_train_error(self):
        """ Gets current training error.

        Returns
        -------
        float
            Returns average training error.
        """
        if self._current_data_train is None:
            return 0.0
        else:
            return self._current_data_train[0]

    def get_train_cost(self):
        """ Gets current training cost.

        Returns
        -------
        float
            Returns training cost.
        """
        if self._current_data_train is None:
            return 0.0
        else:
            return self._current_data_train[1]

    def get_test_score(self):
        """ Gets current testing score.

        Returns
        -------
        float
            Returns testing score.
        """
        if self._current_data_valid is None or len(self._score_function_list) <= 0:
            return 0.0
        else:
            return self._current_data_valid[self._index_score()]

    def get_train_score(self):
        """ Gets current training score.

        Returns
        -------
        float
            Returns training score.
        """
        if self._current_data_train is None or len(self._score_function_list) <= 0:
            return 0.0
        else:
            return self._current_data_train[self._index_score()]

    def _index_score(self):
        """ Gets index in vector of training result where start score.

        Returns
        -------
        int
            Returns index in vector of training result where start score.
        """
        if self._is_fast_compile:
            return 2
        else:
            return len(self._cost_function_list['list']) + len(self._reg_function_list) + 2

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

    def get_target_labels(self):
        """ Getter target labels.

        Returns
        -------
        list
            Returns one list with target labels of this model.
        """
        return self.__target_labels

    def get_new_metric(self):
        """ Get metrics for respective model.

        .. note:: This function is necessary implemented for uses FactoryMetrics.

        See Also
        --------
        FactoryMetrics
        """
        raise NotImplementedError

    def error(self, _input, _target, prob=True):
        """ Compute the error prediction of model.

        Parameters
        ----------
        _input : theano.tensor.matrix or numpy.array
            Input sample.

        _target : theano.tensor.matrix or numpy.array
            Target sample.

        prob :  bool
            In the case of classifier if is True the output is probability, for False means the output is translated.
            Is recommended hold True for training because the translate function is non-differentiable.

        Returns
        -------
        theano.tensor.matrix or numpy.array
            Returns error of model prediction.

        """
        if _input == self._model_input and _target == self._model_target:
            if self._error is None:
                self._error = self.output(_input, prob=prob) - _target
            return self._error
        else:
            return self.output(_input, prob=prob) - _target

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
            return (self.get_input_shape() == other.get_input_shape()) and \
                   (self.get_output_shape() == other.get_output_shape()) and \
                   (self.get_type_model() == other.get_type_model()) and \
                   (not self.is_classifier() or
                    (list(self.get_target_labels()) == list(other.get_target_labels())))
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
        pass

    def output(self, _input, prob=True):
        """ Output model

        Parameters
        ----------
        _input : theano.tensor.matrix
            Input sample.

        prob : bool
            In the case of classifier if is True the output is probability, for False means the output is translated.
            Is recommended hold True for training because the translate function is non-differentiable.

        Returns
        -------
        theano.tensor.matrix or numpy.array
            Raw output of model.
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
        output = self.output(_input, prob=False)

        if self.is_classifier():
            return np.squeeze(self.__target_labels[get_index_label_classes(output, self.is_binary_classification())])
        else:
            return output.eval()

    def batch_eval(self, data, batch_size=32, train=True, shuffle=False):
        """ Evaluate cost and score in mini batch.

        Parameters
        ----------
        data : dict
            Dictionary with 'input' and 'output' data.

        batch_size: int
            Size of batch.

        train: bool
            Flag for knowing if the evaluation of batch is for training or testing.

        shuffle : bool


        Returns
        -------
        numpy.array
            Returns evaluation cost and score in mini batch.
        """
        _input = data['input']
        _target = data['target']

        n_input = len(_input)
        indices = []
        if shuffle:
            indices = np.arange(n_input)
            np.random.shuffle(indices)

        fun_eval = self._train_eval if train else self._valid_eval

        if n_input > batch_size:
            t_data = []
            for (start, end) in zip(range(0, n_input, batch_size), range(batch_size, n_input, batch_size)):
                r = (end - start) / n_input
                if shuffle:
                    batch_index = indices[start:end]
                else:
                    batch_index = slice(start, end)

                t_data.append(fun_eval(_input[batch_index], _target[batch_index], r))

            return np.mean(t_data, axis=0)
        else:
            return fun_eval(_input, _target, 1.0)

    def fit(self, _input, _target, max_epoch=100, batch_size=32, early_stop=True,
            improvement_threshold=0.995, minibatch=True, update_sets=True):
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

        improvement_threshold : float, 0.995 by default

        minibatch : bool
            Flag for indicate training with minibatch or not.

        update_sets : bool
            Flag for update sets or not.

        Returns
        -------
        numpy.array[float]
            Returns training cost for each batch.
        """
        # create a specific metric
        if not self.is_compiled():
            raise AssertionError('The model need to be compiled before to be used.')

        metric_model = FactoryMetrics().get_metric(self)

        # save data in shared variables
        n_train, n_valid = self.prepare_data(_input, _target)

        # parameters early stopping
        best_params = None
        best_validation_cost = np.inf
        patience = max(max_epoch * n_train // 5, 5000)
        validation_jump = max(min(patience // 100, max_epoch // 50), 1)
        patience_increase = 2

        for epoch, _ in enumerate(Logger().progressbar_training(max_epoch, self)):

            if minibatch:  # Train minibatches
                self._current_data_train = self.batch_eval(self._data_train, batch_size=batch_size, train=True,
                                                           shuffle=update_sets)
            else:
                self._current_data_train = self.batch_eval(self._data_train, batch_size=n_train, train=True,
                                                           shuffle=update_sets)

            metric_model.append_data(self._current_data_train, epoch, type_set_data="train")

            iteration = epoch * n_train

            if epoch % validation_jump == 0 or epoch == max_epoch:
                # Evaluate test set
                self._current_data_valid = self.batch_eval(self._data_valid, batch_size=batch_size, train=False,
                                                           shuffle=update_sets)
                metric_model.append_data(self._current_data_valid, epoch, type_set_data="test")
                validation_cost = self.get_test_cost()

                if validation_cost < best_validation_cost:

                    if validation_cost < best_validation_cost * improvement_threshold:
                        patience = max(patience, iteration * patience_increase)

                    best_params = self.save_params()
                    best_validation_cost = validation_cost

            if early_stop and patience <= iteration:
                Logger().log()
                break

        if best_params is not None:
            self.load_params(best_params)

        return metric_model

    def save_params(self):
        """ Save parameter of model.

        Returns
        -------
        list[]
            Returns a list with the parameters model.
        """
        return [i.get_value() for i in self.get_params()]

    def load_params(self, params):
        """ Load parameters.

        Parameters
        ----------
        params : list[]
            List of parameters.
        """
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
        """ Prepare training (compile function of Theano).

        Parameters
        ----------
        fast : bool
            Compile model only necessary.

        kwargs

        Raises
        ------
        If exist an inconsistency between output and count classes
        """
        Logger().start_measure_time("Start Compile %s" % self._name)
        self._is_fast_compile = fast

        # review possibles mistakes
        self.review_is_binary_classifier()
        self.review_shape_output()

        cost, updates, extra_results, labels_extra_results = self._compile(fast=fast, **kwargs)

        if self.is_classifier():
            error = T.mean(T.neq(self.output(self._model_input, prob=False), self._model_target))
        else:
            error = T.mean(self.error(self._model_input, self._model_target))

        self._labels_result_train = []
        result = [error, cost]
        self._labels_result_train += ['Error', 'Cost']

        if not fast:
            costs = self.get_costs()
            scores = self.get_scores()
            result += costs + scores + extra_results
            self._labels_result_train += self.get_labels_costs()
            self._labels_result_train += self.get_labels_scores()
            self._labels_result_train += labels_extra_results
        else:
            # append only first score (default score)
            result += [self.get_scores()[0]]
            self._labels_result_train += [self.get_labels_scores()[0]]

        _inputs = [self._model_input, self._model_target, self._batch_reg_ratio]

        self._train_eval = function(inputs=_inputs, outputs=result, updates=updates,
                                    on_unused_input='ignore', allow_input_downcast=True)
        self._valid_eval = function(inputs=_inputs, outputs=result,
                                    on_unused_input='ignore', allow_input_downcast=True)

        Logger().stop_measure_time()
        self._is_compiled = True

    @staticmethod
    def __extract_info(list_func):
        """ Extract info of parameters from a function list.

        Returns
        -------
        str
            Returns string with functions info.
        """
        str_info = ''
        for fun, name, kwargs in list_func:
            args_fun = inspect.getargspec(fun)
            comment = fun.__doc__.splitlines()[0] if fun.__doc__ is not None else None

            if comment is not None:
                str_info += ' %s: %s\n' % (name, comment)
            else:
                str_info += ' %s:\n' % name

            if args_fun.defaults is not None:
                params_fun = dict(zip(args_fun.args[-len(args_fun.defaults):], args_fun.defaults))
                params_fun.update(kwargs)
                str_info += ' params: %s\n' % params_fun

        return str_info

    def _get_spec_model(self):
        """ Gets specific info about this model.
        """
        return 'Info model:\n Type model: %s\n inputs: %d\n outputs: %d\n' % \
               (self.__type_model, self.get_fan_in(), self.get_fan_out())

    # noinspection PyMethodMayBeStatic
    def _get_extra_info(self):
        """ Gets extra info about this model.
        """
        return None

    def __generate_info(self):
        """ Generate information about this model.

        Information about:

            - Specific information about model.
            - Cost functions.
            - Regularization.
            - Score functions.

        Returns
        -------
        None
        """
        self._info_model['info'] = ''  # Reset info model
        self._info_model['info'] += self._get_spec_model() + '\n'

        self._info_model['info'] += 'Update params:\n' + self.__extract_info([self._update_function]) + '\n'

        self._info_model['info'] += 'Cost functions:\n' + self.__extract_info(self._cost_function_list['list']) + '\n'

        if len(self._reg_function_list) > 0:
            self._info_model['info'] += \
                'Regularization functions:\n' + self.__extract_info(self._reg_function_list) + '\n'

        self._info_model['info'] += 'Score functions:\n' + self.__extract_info(self._score_function_list['list']) + '\n'

    def review_shape_output(self):
        """ Review if this model its dimension output is wrong.

        Raises
        ------
        If exist an inconsistency in output.
        """
        if self.is_classifier() and len(self.__target_labels) != self.get_fan_out() and \
                not self.__binary_classification:  # no is binary classifier
            raise ValueError("Output model is not equals to number of classes.")  # TODO: review translation

    def review_is_binary_classifier(self):
        """ Review this model is binary classifier
        """
        if self.is_classifier() and len(self.__target_labels) == 2 and self.get_fan_out() == 1:
            self.__binary_classification = True

    def prepare_data(self, _input, _target, test_size=0.20):
        """

        Parameters
        ----------
        _input
        _target
        test_size
        """
        input_train, input_valid, target_train, target_valid = \
            cross_validation.train_test_split(_input, _target, test_size=test_size, stratify=_target)

        if self.__type_model == 'classifier':
            target_train = self.translate_target(_target=target_train)
            target_valid = self.translate_target(_target=target_valid)

        self._data_train['input'] = input_train
        self._data_train['target'] = target_train
        self._data_valid['input'] = input_valid
        self._data_valid['target'] = target_valid

        return len(input_train), len(input_valid)

    def translate_target(self, _target):
        """ Translate target.

        Parameters
        ----------
        _target : numpy.array
            Target sample.

        Returns
        -------
        numpy.array
            Returns the '_target' translated according to target labels.
        """
        if self.is_binary_classification():
            return translate_binary_target(_target=_target, target_labels=self.__target_labels)
        else:
            return translate_target(_target=_target, target_labels=self.__target_labels)

    def append_score(self, fun_score, name, **kwargs):
        """ Adds an extra item in the score functions.

        Parameters
        ----------
        fun_score : theano.function
            Function of score.

        name : str
            This string identify score function, is useful for plot metrics.

        **kwargs
            Extra parameters of score function.
        """
        self._score_function_list['list'].append((fun_score, name, kwargs))
        self._score_function_list['changed'] = True

    def append_cost(self, fun_cost, name, **kwargs):
        """ Adds an extra item in the cost functions.

        Parameters
        ----------
        fun_cost : theano.function
            Function of cost.

        name : str
            This string identify cost function, is useful for plot metrics.

        **kwargs
            Extra parameters of cost function.
        """
        self._cost_function_list['list'].append((fun_cost, name, kwargs))
        self._cost_function_list['changed'] = True

    def append_reg(self, fun_reg, name, **kwargs):
        """ Adds an extra item in the cost functions.

        Parameters
        ----------
        fun_reg : theano.function
            Function of regularization.

        name : str
            This string identify regularization function, is useful for plot metrics.

        **kwargs
            Extra parameters of regularization function.
        """
        self._reg_function_list.append((fun_reg, name, kwargs))
        self._cost_function_list['changed'] = True

    def set_update(self, fun_update, name, **kwargs):
        """ Adds an extra item in the cost function.

        Parameters
        ----------
        fun_update : theano.function
            Function of update parameters of models.

        name : str
            This string identify regularization function, is useful for plot metrics.

        **kwargs
            Extra parameters of update function.
        """
        self._update_function = (fun_update, name, kwargs)

    def get_cost(self):
        """ Get cost function.

        Returns
        -------
        theano.tensor.TensorVariable
            Returns cost function.
        """
        return sum(self.get_costs())

    def get_costs(self):
        """ Gets cost function of model.

        Returns
        -------
        list[]
            Returns cost model list that include regularization.
        """
        if self._cost_function_list['changed']:
            self._cost_function_list['result'] = []
            for fun_cost, _, params in self._cost_function_list['list']:
                self._cost_function_list['result'].append(fun_cost(model=self,
                                                                   _input=self._model_input,
                                                                   _target=self._model_target, **params))

            for fun_reg, _, params in self._reg_function_list:
                self._cost_function_list['result'].append(fun_reg(model=self,
                                                                  batch_reg_ratio=self._batch_reg_ratio, **params))
            self._cost_function_list['changed'] = False

        return self._cost_function_list['result']

    def get_scores(self):
        """ Gets score function of model.

        Returns
        -------
        list[]
            Returns score model list.
        """
        if self._score_function_list['changed']:
            self._score_function_list['result'] = []
            if self.is_classifier():
                output = self.output(self._model_input, prob=False)

                if output is not None:  # TODO: for Wrapper model, changed in the future
                    output = translate_output(output, self.get_fan_out(), self.is_binary_classification())

                for fun_score, _, params in self._score_function_list['list']:
                    self._score_function_list['result'].append(fun_score(_output=output,
                                                                         _input=self._model_input,
                                                                         _target=self._model_target,
                                                                         model=self,
                                                                         **params))
            else:
                output = self.output(self._model_input)

                for fun_score, _, params in self._score_function_list['list']:
                    self._score_function_list['result'].append(fun_score(_output=output,
                                                                         _input=self._model_input,
                                                                         _target=self._model_target,
                                                                         model=self,
                                                                         **params))

            self._score_function_list['changed'] = False

        return self._score_function_list['result']

    def get_labels_costs(self):
        """ Gets list of cost functions.

        Returns
        -------
        list[]
            Returns a list cost functions.
        """
        return [l for _, l, _ in self._cost_function_list['list'] + self._reg_function_list]

    def get_labels_scores(self):
        """ Gets list of score functions.

        Returns
        -------
        list[]
            Returns a list score functions.
        """
        return [l for _, l, _ in self._score_function_list['list']]

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
        return self._update_function[0](cost, self._params, **self._update_function[2])

    def score(self, _input, _target):
        """ Gets score prediction.

        Parameters
        ----------
        _input : numpy.array
            Input sample.

        _target : numpy.array
            Target sample.

        Returns
        -------
        float
            Returns score prediction.
        """
        return accuracy_score(np.squeeze(_target), np.squeeze(self.predict(_input)))
