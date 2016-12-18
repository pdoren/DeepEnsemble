from sklearn import cross_validation, clone
from copy import deepcopy

from .model import Model
from ..metrics import FactoryMetrics, ClassifierMetrics, RegressionMetrics
from ..utils.cost_functions import dummy_cost
from ..utils.logger import Logger
from ..utils.score_functions import dummy_score
from ..utils.update_functions import dummy_update

__all__ = ['Wrapper']


class Wrapper(Model):
    """ Wrapper Model class.

    This model is a wrapper for model from sklearn (scikit).

    Attributes
    ----------

    __model
        Model from sklearn.

    __clf
        Training model.

    """

    def __init__(self, model, name, input_shape=None, output_shape=None, type_model="regressor", target_labels=None):
        if type_model == "regressor":
            target_labels = []
        super(Wrapper, self).__init__(input_shape=input_shape, output_shape=output_shape, target_labels=target_labels,
                                      type_model=type_model, name=name)

        self.__model = model
        self.__clf = None

        # Reset score default
        self._score_function_list = {'list': [], 'changed': True, 'result': []}
        self.append_score(dummy_score, 'Accuracy')

        # Default cost
        self.append_cost(dummy_cost, 'Cost')
        self._labels_result_train = ['Error', 'Cost'] + self.get_labels_costs() + self.get_labels_scores()

        # Default update function
        self.set_update(dummy_update, name='dummy update')

    def get_new_metric(self):
        """ Get metrics for respective model.

        See Also
        --------
        FactoryMetrics
        """
        if self.is_classifier():
            return ClassifierMetrics(self)
        else:
            return RegressionMetrics(self)

    def fit(self, _input, _target, nfolds=20, max_epoch=300, batch_size=32, early_stop=True, valid_size=0.20,
            no_update_best_parameters=False, improvement_threshold=0.995, minibatch=True, update_sets=True):
        """ Training model.
        """
        folds = list(cross_validation.StratifiedKFold(_target, nfolds, shuffle=True))
        metric_model = FactoryMetrics().get_metric(self)

        best_score = 0
        cls_reset = clone(self.__model)
        for i, (train_index, test_index) in enumerate(Logger().progressbar_training2(folds, self)):
            fold_X_train = _input[train_index]
            fold_y_train = _target[train_index]
            fold_X_test = _input[test_index]
            fold_y_test = _target[test_index]

            self.__model = clone(cls_reset)
            cls = self.__model.fit(fold_X_train, fold_y_train)

            score_train = cls.score(fold_X_train, fold_y_train)
            score_test = cls.score(fold_X_test, fold_y_test)

            # TODO: it's necessary to add the error and cost values
            self._current_data_train = [0] * 4
            self._current_data_valid = [0] * 4

            self._current_data_train[3] = score_train
            self._current_data_valid[3] = score_test

            metric_model.append_data(self._current_data_train, i, type_set_data="train")
            metric_model.append_data(self._current_data_valid, i, type_set_data="test")

            if score_test >= best_score:
                self.__clf = deepcopy(cls)
                best_score = score_test

        return metric_model

    def output(self, _input, prob=True):
        """ Output model.
        """
        if _input != self._model_input:
            if self.__clf is None:
                return self.__model.predict_proba(_input)
            else:
                return self.__clf.predict_proba(_input)

        return None

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
        if self.__clf is None:
            return self.__model.predict(_input)
        else:
            return self.__clf.predict(_input)

    def _compile(self, fast=True, **kwargs):
        """ not does nothing.
        """
        pass

    def compile(self, fast=True, **kwargs):
        """ Update intern parameters.
        """
        Logger().start_measure_time("Start Compile %s" % self._name)

        # review possibles mistakes
        self.review_is_binary_classifier()
        self.review_shape_output()
        self._define_output()

        Logger().stop_measure_time()
        self._is_compiled = True
