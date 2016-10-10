import theano.tensor as T

__all__ = [
    'dummy_score',
    'score_accuracy',
    'score_ensemble_ambiguity',
    'score_rms'
]


# noinspection PyUnusedLocal
def dummy_score(_input, _output, _target, model):
    pass


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
    ensemble = model
    err = [T.mean(T.sqr(model.output(_input, prob=False) - _output)) for model in ensemble.get_models()]
    return sum(err) / ensemble.get_num_models()


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
