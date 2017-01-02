from .modelcombiner import ModelCombiner
from ..utils import translate_output
import numpy as np
import theano.tensor as T

__all__ = ['GeometricCombiner', 'GeometricVotingCombiner']


#
# For Regression
#
class GeometricCombiner(ModelCombiner):
    """ Class for compute the average the output models.
    """

    def __init__(self, **kwargs):
        super(GeometricCombiner, self).__init__(**kwargs)

    # noinspection PyMethodMayBeStatic
    def output(self, ensemble_model, _input, prob):
        """ Multiplied the output of the ensemble's models.

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
        L = ensemble_model.get_num_models()

        return T.power(np.prod([model.output(_input, prob) for model in ensemble_model.get_models()]), 1.0 / L)


#
# For Classification
#
class GeometricVotingCombiner(ModelCombiner):
    """ Combiner classifier method where each model in ensemble votes by one class and the class with more votes win.
    """

    def __init__(self):
        super(GeometricVotingCombiner, self).__init__(type_model="classifier")

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
            L = ensemble_model.get_num_models()

            return T.power(np.prod([model.output(_input, prob) for model in ensemble_model.get_models()]), 1.0 / L)
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
            GeometricVotingCombiner._vote(voting, votes)

        return GeometricVotingCombiner._result(voting)

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
