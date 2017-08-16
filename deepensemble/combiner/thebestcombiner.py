from collections import OrderedDict

import theano.tensor as T
from theano import shared

from .modelcombiner import ModelCombiner
from ..utils import translate_output

__all__ = ['TheBestCombiner', 'TheBestVotingCombiner']


#
# For Regression
#
class TheBestCombiner(ModelCombiner):
    """ Class for selecting the best model in Ensemble.

    """

    def __init__(self, **kwargs):
        super(TheBestCombiner, self).__init__(**kwargs)
        self._param = {'name': 'index_the_best_model',
                       'value': shared(0, name='index_the_best_model', borrow=True),
                       'shape': (1, 1),
                       'init': False,
                       'include': False}

    # noinspection PyMethodMayBeStatic
    def output(self, ensemble_model, _input, prob):
        """ Average the output of the ensemble's models.

        Parameters
        ----------
        ensemble_model : EnsembleModel
            Ensemble Model it uses for get ensemble's models.

        _input : theano.tensor.matrix or numpy.array
            Input sample.

        prob : bool
            In the case of classifier if is True the output is probability, for False means the output is translated.
            Is recommended hold True for training because the translate function is non-differentiable.

        Returns
        -------
        theano.Op
            Returns output of the best model in ensemble.
        """
        outputs = [model.output(_input, prob) for model in ensemble_model.get_models()]
        if _input == ensemble_model.get_model_input():
            index = self.get_param(only_values=True).get_value()
        else:
            index = self.get_param(only_values=True).get_value()
        return outputs[index]

    def update_parameters(self, ensemble_model, _input, _target):
        """  Update internal parameters.

        Parameters
        ----------
        ensemble_model : EnsembleModel
            Ensemble Model it uses for get ensemble's models.

        _input : theano.tensor.matrix
            Input sample.

        _target : theano.tensor.matrix
            Target sample.

        Returns
        -------
        OrderedDict
            A dictionary mapping each parameter to its update expression.
        """
        errors = T.stack([T.sum(T.sqr(model.error(_input, _target))) for model in ensemble_model.get_models()])
        updates = OrderedDict()

        param = self.get_param(only_values=True)
        updates[param] = T.argmin(errors)

        return updates


#
# For Classification
#
class TheBestVotingCombiner(TheBestCombiner):
    """ Class for selecting the best model in Ensemble.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 74:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """

    def __init__(self):
        super(TheBestVotingCombiner, self).__init__(type_model="classifier")

    # noinspection PyMethodMayBeStatic
    def output(self, ensemble_model, _input, prob):
        """ Average the output of the ensemble's models.

        Parameters
        ----------
        ensemble_model : EnsembleModel
            Ensemble Model it uses for get ensemble's models.

        _input : theano.tensor.matrix or numpy.array
            Input sample.

        prob : bool
            In the case of classifier if is True the output is probability, for False means the output is translated.
            Is recommended hold True for training because the translate function is non-differentiable.

        Returns
        -------
        theano.Op
            Returns the average of the output models.
        """
        if prob:
            return super(TheBestVotingCombiner, self).output(ensemble_model, _input, prob)
        else:
            outputs = [translate_output(model.output(_input, prob),
                                        ensemble_model.get_fan_out(),
                                        ensemble_model.is_binary_classification()) for model in
                       ensemble_model.get_models()]
            if _input == ensemble_model.get_model_input():
                index = self.get_param(only_values=True).get_value()
            else:
                index = self.get_param(only_values=True).get_value()
            return outputs[index]

    def predict(self, ensemble_model, _input):
        """ Returns the class with more votes.

        Parameters
        ----------
        ensemble_model : EnsembleModel
            Ensemble model where gets the output.

        _input : theano.tensor.matrix or numpy.array
            Input sample.

        Returns
        -------
        numpy.array
            Return the diversity of model.
        """
        index = self.get_param(only_values=True).get_value()
        for i, model in enumerate(ensemble_model.get_models()):
            if i == index:
                return model.predict(_input)

        return ensemble_model.get_models()[0].predict(_input)  # default
