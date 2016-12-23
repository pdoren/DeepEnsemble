from .modelcombiner import ModelCombiner
from ..utils.utils_classifiers import translate_output
from ..utils.utils_classifiers import get_index_label_classes
import numpy as np

__all__ = ['AverageCombiner', 'PluralityVotingCombiner', 'SoftVotingCombiner']


#
# For Regression
#
class AverageCombiner(ModelCombiner):
    """ Class for compute the average the output models.
    """

    def __init__(self, **kwargs):
        super(AverageCombiner, self).__init__(**kwargs)

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
        output = [model.output(_input, prob) for model in ensemble_model.get_models()]
        return sum(output) / len(ensemble_model.get_models())


#
# For Classification
#
class PluralityVotingCombiner(ModelCombiner):
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
        super(PluralityVotingCombiner, self).__init__(type_model="classifier")

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
            output = [model.output(_input, prob) for model in ensemble_model.get_models()]
            return sum(output) / len(ensemble_model.get_models())
        else:
            outputs = [translate_output(model.output(_input, prob),
                                        ensemble_model.get_fan_out(),
                                        ensemble_model.is_binary_classification()) for model in
                       ensemble_model.get_models()]
            return translate_output(sum(outputs), ensemble_model.get_fan_out(),
                                    ensemble_model.is_binary_classification())

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
        voting = [{} for _ in range(_input.shape[0])]
        for model in ensemble_model.get_models():
            votes = model.predict(_input)
            PluralityVotingCombiner._vote(voting, votes)

        return PluralityVotingCombiner._result(voting)

    @staticmethod
    def _vote(voting, votes):
        """ Counting votes.

        Parameters
        ----------
        voting : list[dict]
            This dictionary keeps the votes.

        votes : list[]
            This list has votes.

        Returns
        -------
        None
        """
        for i, vote in enumerate(votes):
            if vote in voting[i]:
                (voting[i])[vote] += 1
            else:
                (voting[i])[vote] = 1

    @staticmethod
    def _result(voting):
        """ Gets the result of voting.

        Parameters
        ----------
        voting : list[dict]
            This dictionary has recount.

        Returns
        -------
        numpy.array
            Returns a list with results of voting.
        """
        result = []
        for votes in voting:
            result.append(max(votes, key=lambda key: votes[key]))

        return np.array(result)


class SoftVotingCombiner(ModelCombiner):
    """ Combiner classifier method where each model in ensemble votes by one class and the class with more votes win.

    Plurality voting takes the class label which receives the largest number of votes as the final winner.
    That is, the output class label of the ensemble.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 76:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """

    def __init__(self):
        super(SoftVotingCombiner, self).__init__(type_model="classifier")

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
            output = [model.output(_input, prob) for model in ensemble_model.get_models()]
            return sum(output) / len(ensemble_model.get_models())
        else:
            outputs = [translate_output(model.output(_input, prob),
                                        ensemble_model.get_fan_out(),
                                        ensemble_model.is_binary_classification()) for model in
                       ensemble_model.get_models()]
            return translate_output(sum(outputs), ensemble_model.get_fan_out(),
                                    ensemble_model.is_binary_classification())

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
        output = ensemble_model.output(_input, prob=False)
        labels = ensemble_model.get_target_labels()
        return np.squeeze(labels[get_index_label_classes(output, ensemble_model.is_binary_classification())])
