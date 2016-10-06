import theano.tensor as T
from .modelcombiner import ModelCombiner
from theano import shared, config
import numpy as np
from collections import OrderedDict
from ..utils.utils_classifiers import get_index_label_classes

__all__ = ['WeightAverageCombiner', 'WeightedVotingCombiner']


#
# For Regression
#
class WeightAverageCombiner(ModelCombiner):
    """ Class for compute the average the output models.

    Attributes
    ----------
    n_models : int
        Number of models in ensemble.

    params : theano.shared
        This parameter contain the weights of method.

    Parameters
    ----------
    n_models : int
        Number of models of ensemble.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 70:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.

    .. [2] M. P. Perrone and L. N. Cooper. When networks disagree: Ensemble method
           for neural networks. In R. J.Mammone, editor, Artificial Neural Networks
           for Spech and Vision, pages 126–142. Chapman & Hall, New York, NY,
           1993.
    """
    def __init__(self, n_models, **kwargs):
        super(WeightAverageCombiner, self).__init__(**kwargs)
        self.n_models = n_models
        self._params = shared(np.ones(shape=(n_models, 1), dtype=config.floatX), name='Wa_ens', borrow=True)

    # noinspection PyMethodMayBeStatic
    def output(self, ensemble_model, _input):
        """ Average the output of the ensemble's models.

        Parameters
        ----------
        ensemble_model : EnsembleModel
            Ensemble Model it uses for get ensemble's models.

        _input : theano.tensor.matrix or numpy.array
            Input sample.

        Returns
        -------
        numpy.array
            Returns the average of the output models.
        """
        output = 0.0
        if _input == ensemble_model.model_input:
            for i, model in enumerate(ensemble_model.get_models()):
                output += model.output(_input) * self._params[i, 0]  # index TensorVariable
        else:
            params = self._params.get_value()
            for i, model in enumerate(ensemble_model.get_models()):
                output += model.output(_input) * params[i]
        return output

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
        updates = OrderedDict()
        errors = []

        for model in ensemble_model.get_models():
            errors.append(model.error(_input, _target))

        sum_Cj = 0.0
        for j in range(self.n_models):
            sum_Cj += errors[j]

        inv_sum_Cij = []
        inv_sum_sum_inv_Ckj = 0.0
        for i in range(self.n_models):
            d = 1.0 / T.mean(sum_Cj * errors[i])
            inv_sum_Cij.append(d)
            inv_sum_sum_inv_Ckj += d

        update_param = (1.0 / inv_sum_sum_inv_Ckj) * inv_sum_Cij
        updates[self._params] = T.set_subtensor(self._params[:, 0], update_param[:])
        return updates


#
# For Classification
#
class WeightedVotingCombiner(WeightAverageCombiner):
    """ Class for compute the average the output models.

    Parameters
    ----------
    n_models : int
        Number of models of ensemble.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 74:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    def __init__(self, n_models):
        super(WeightedVotingCombiner, self).__init__(n_models=n_models, type_model="classifier")

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
            Return the prediction of model.
        """
        voting = [{} for i in range(_input.shape[0])]
        for i, model in enumerate(ensemble_model.get_models()):
            votes = model.predict(_input)
            WeightedVotingCombiner._vote(voting, votes, self._params[i].eval())

        return WeightedVotingCombiner._result(voting)

    @staticmethod
    def _vote(voting, votes, weight):
        for i, vote in enumerate(votes):
            if vote in voting[i]:
                (voting[i])[vote] += weight
            else:
                (voting[i])[vote] = weight

    @staticmethod
    def _result(voting):
        result = []
        for votes in voting:
            result.append(max(votes, key=lambda key: votes[key]))

        return result
