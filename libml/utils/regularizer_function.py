import theano.tensor as T

__all__ = ['L2', 'L1']


def L2(model, lamb, batch_reg_ratio):
    """ Compute regularization square L2.

    Parameters
    ----------
    model: Model
        Model for generating output for compare with target sample.

    batch_reg_ratio: float
        Ratio batch.

    lamb: float or double
        Ratio regularization L2.

    Returns
    -------
    float or double
    Return regularization square L2.

    """

    sqrL2W = 0.0
    for layer in model.layers:
        sqrL2W += T.sum(T.power(layer.get_parameters(), 2.0))
    return sqrL2W * (lamb * batch_reg_ratio)


def L1(model, lamb, batch_reg_ratio):
    """ Compute regularization L1.

    Parameters
    ----------
    model: Model
        Model for generating output for compare with target sample.

    batch_reg_ratio: float
        Ratio batch.

    lamb: float or double
        Ratio regularization L1.

    Returns
    -------
    float or double
    Return regularization L1.

    """
    L1W = 0.0
    for layer in model.layers:
        L1W += T.sum(T.abs_(layer.get_parameters()))
    return L1W * (lamb * batch_reg_ratio)
