__all__ = ['Layer']


class Layer(object):
    """ Base class layers.

    Attributes
    ----------
    _n_input : int
        Dimension of input.

    _n_output : int
        Dimension of output.

    Parameters
    ----------
    n_input : int
        Dimension of input.

    n_output : int
        Dimension of output.
    """
    def __init__(self, n_input=None, n_output=None):
        self._n_input = n_input
        self._n_output = n_output
        self._params = []

    def get_n_inputs(self):
        return self._n_input

    def set_n_inputs(self, n_input):
        self._n_input = n_input

    def get_n_outputs(self):
        return self._n_output

    def set_n_outputs(self, n_output):
        self._n_output = n_output

    def get_parameters(self):
        return self._params
