from .modelcombiner import ModelCombiner
from ..utils.utils_classifiers import *

__all__ = ['AverageCombiner', 'PluralityVotingCombiner']


#
# For Regression
#
class AverageCombiner(ModelCombiner):
    """ Class for compute the average the output models.
    """
    def __init__(self):
        super(AverageCombiner, self).__init__()

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
        for model in ensemble_model.list_models_ensemble:
            output += model.output(_input)
        n = len(ensemble_model.list_models_ensemble)
        return output / n


#
# For Classification
#
class PluralityVotingCombiner(AverageCombiner):
    """ Combiner classifier method where each model in ensemble votes by one class and the class with more votes win.

    Plurality voting takes the class label which receives the largest number of votes as the final winner.
    That is, the output class label of the ensemble.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 73:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    def __init__(self):
        super(PluralityVotingCombiner, self).__init__()
        self.type_model = "classifier"

    def predict(self, model, _input):
        """ Returns the class with more votes.

        Parameters
        ----------
        model : EnsembleModel
            Ensemble model where gets the output.

        _input : theano.tensor.matrix or numpy.array
            Input sample.

        Returns
        -------
        numpy.array
            Return the prediction of model.
        """
        return model.target_labels[get_index_label_classes(model.output(_input))]
