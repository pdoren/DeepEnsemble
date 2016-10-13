import theano.tensor as T
from ..utils.utils_functions import ITLFunctions

__all__ = [
    'dummy_score',
    'score_accuracy',
    'score_ensemble_ambiguity',
    'score_rms',
    'score_silverman'
]


# noinspection PyUnusedLocal
def dummy_score(_input, _output, _target, model):
    """ Dummy score function, this function only return zeros for each elements in _target.

    Parameters
    ----------
    _input : theano.tensor.matrix
        Input Sample.

    _output : theano.tensor.matrix
        Output model.

    _target : theano.tensor.matrix
        Target Sample.

    model : Model
        Model.

    Returns
    -------
    theano.tensor.matrix
        Returns only zeros for each elements in _target.
    """
    return T.zeros(_target.shape)


#
# Classification Functions
#

# noinspection PyUnusedLocal
def score_accuracy(_input, _output, _target, model):
    """ Accuracy score in a classifier models.

    Parameters
    ----------
    _input : theano.tensor.matrix
        Input sample.

    _output : theano.tensor.matrix
        Output sample.

    _target : theano.tensor.matrix
        Target sample.

    model : Model
        Model.

    Returns
    -------
    theano.tensor.matrix
        Returns accuracy in a classifier models.
    """
    return T.mean(T.eq(_output, _target))


# noinspection PyUnusedLocal
def score_ensemble_ambiguity(_input, _output, _target, model):
    """ Score ambiguity for Ensemble.

    Parameters
    ----------
    _input : theano.tensor.matrix
        Input sample.

    _output : theano.tensor.matrix
        Output sample.

    _target : theano.tensor.matrix
        Target sample.

    model : Model
        Model.

    Returns
    -------
    float
        Returns a score ambiguity.
    """
    ensemble = model
    err = [T.mean(T.sqr(model.output(_input, prob=False) - _output)) for model in ensemble.get_models()]
    return sum(err) / ensemble.get_num_models()


# noinspection PyUnusedLocal
def score_silverman(_input, _output, _target, model):
    """ Score Silverman.

    Parameters
    ----------
    _input : theano.tensor.matrix
        Input sample.

    _output : theano.tensor.matrix
        Output sample.

    _target : theano.tensor.matrix
        Target sample.

    model : Model
        Model.

    Returns
    -------
    float
        Returns size kernel with Silverman Rule.
    """
    return ITLFunctions.silverman(model.output(_input), _target.shape[0], model.get_dim_output())


#
# Regression Functions
#

# noinspection PyUnusedLocal
def score_rms(_input, _output, _target, model):
    """ Gets Root Mean Square like score in a regressor model.

    Parameters
    ----------
    _input : theano.tensor.matrix
        Input sample.

    _output : theano.tensor.matrix
        Output sample.

    _target : theano.tensor.matrix
        Target sample.

    model : Model
        Model.

    Returns
    -------
    theano.tensor.matrix
        Returns Root Mean Square.
    """
    e = _output - _target
    return T.mean(T.power(e, 2.0))