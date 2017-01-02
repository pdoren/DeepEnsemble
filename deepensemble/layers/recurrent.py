from .layer import Layer
from ..utils.utils_functions import ActivationFunctions

import theano.tensor as T
from theano import config, shared, scan
import numpy as np

__all__ = ['RecurrentLayer', 'LSTMLayer']


class RecurrentLayer(Layer):
    def __init__(self, n_input=None, n_recurrent=None, activation=None):
        input_shape = n_input if isinstance(n_input, tuple) else (None, n_input)
        recurrent_shape = n_recurrent if isinstance(n_recurrent, tuple) else (None, n_recurrent)
        super(RecurrentLayer, self).__init__(input_shape=input_shape,
                                             output_shape=recurrent_shape,
                                             non_linearity=activation)

        self._Wr = {'name': 'Wr', 'value': None, 'shape': None, 'update_shape': True, 'include': True}
        self._W.append(self._Wr)
        self.update_W_shape()

    def get_shape_Wr(self):
        """ Gets shape weights of layer.
        """
        return self.get_fan_out(), self.get_fan_out()

    def get_shape_W(self):
        """ Gets shape weights of layer.
        """
        return self.get_fan_in(), self.get_fan_out()

    def get_shape_b(self):
        """ Gets shape bias of layer.
        """
        return self.get_fan_out(),

    def _step(self, input_t, previous):
        """ Step scan """
        lin_output = T.cast(T.dot(previous, self._Wr['value']) + input_t, dtype=config.floatX)

        # noinspection PyCallingNonCallable
        return (
            lin_output if self._non_linearity is None
            else self._non_linearity(lin_output)
        )

    def output(self, x, prob=True):
        """ Return output of layers

        Parameters
        ----------
        x : theano.tensor.matrix
            Input sample

        Returns
        -------
        theano.tensor.matrix
            Returns the output layers according to above equation:

            .. math::  Layer(x) = activation(Wx + b)
        """
        if x.ndim > 2:
            x = x.flatten(2)

        x_W_b = T.dot(x, self._W) + self._b.dimshuffle('x', 0)
        result, updates = scan(self._step,
                               sequences=[x_W_b],
                               outputs_info=[T.zeros_like(self._b)])
        return result


class LSTMLayer(Layer):
    def __init__(self, n_input=None, n_recurrent=None, mask=None,
                 activation=ActivationFunctions.sigmoid,
                 activation2=ActivationFunctions.tanh):
        input_shape = n_input if isinstance(n_input, tuple) else (None, n_input)
        recurrent_shape = n_recurrent if isinstance(n_recurrent, tuple) else (None, n_recurrent)
        super(LSTMLayer, self).__init__(input_shape=input_shape,
                                        output_shape=recurrent_shape,
                                        non_linearity=activation)

        self._W_f = {'name': 'Wf', 'value': None, 'shape': None, 'update_shape': True, 'include': True}
        self._W.append(self._W_f)
        self._W_c = {'name': 'Wc', 'value': None, 'shape': None, 'update_shape': True, 'include': True}
        self._W.append(self._W_c)
        self._W_o = {'name': 'Wo', 'value': None, 'shape': None, 'update_shape': True, 'include': True}
        self._W.append(self._W_o)

        self._U_i = {'name': 'Ui', 'value': None, 'shape': None, 'update_shape': True, 'include': True}
        self._W.append(self._U_i)
        self._U_f = {'name': 'Uf', 'value': None, 'shape': None, 'update_shape': True, 'include': True}
        self._W.append(self._U_f)
        self._U_c = {'name': 'Uc', 'value': None, 'shape': None, 'update_shape': True, 'include': True}
        self._W.append(self._U_c)
        self._U_o = {'name': 'Uo', 'value': None, 'shape': None, 'update_shape': True, 'include': True}
        self._W.append(self._U_o)

        self._b_f = {'name': 'bf', 'value': None, 'shape': None, 'update_shape': True, 'include': True}
        self._b.append(self._b_f)
        self._b_c = {'name': 'bc', 'value': None, 'shape': None, 'update_shape': True, 'include': True}
        self._b.append(self._b_c)
        self._b_o = {'name': 'bo', 'value': None, 'shape': None, 'update_shape': True, 'include': True}
        self._b.append(self._b_o)

        self._mask = mask

        self._non_linearity = activation if activation is not None \
            else ActivationFunctions.linear

        self._non_linearity2 = activation2 if activation2 is not None \
            else ActivationFunctions.linear

    @staticmethod
    def _index_dot(indices, w):
        return w[indices.flatten()]

    def _step_mask(self, _mask, _input, _h_prev, _c_prev):
        """ Step scan """
        i_preact = (T.dot(_input, self._W['value']) +
                    T.dot(_h_prev, self._U_i['value']) + self._b['value'])
        i = self._non_linearity(i_preact)

        f_preact = (T.dot(_input, self._W_f['value']) +
                    T.dot(_h_prev, self._U_f['value']) + self._b_f['value'])
        f = self._non_linearity(f_preact)

        o_preact = (T.dot(_input, self._W_o['value']) +
                    T.dot(_h_prev, self._U_o['value']) + self._b_o['value'])
        o = self._non_linearity(o_preact)

        c_preact = (T.dot(_input, self._W_c['value']) +
                    T.dot(_h_prev, self._U_c['value']) + self._b_c['value'])
        c = self._non_linearity2(c_preact)

        c = f * _c_prev + i * c
        c = _mask[:, None] * c + (1. - _mask)[:, None] * _c_prev

        h = o * self._non_linearity2(c)
        h = _mask[:, None] * h + (1. - _mask)[:, None] * _h_prev

        return T.cast(h, dtype=config.floatX), T.cast(c, dtype=config.floatX)

    def _step(self, _input, h_prev, c_prev):
        """ Step scan """
        i_preact = (T.dot(_input, self._W['value']) +
                    T.dot(h_prev, self._U_i['value']) + self._b['value'])
        i = self._non_linearity(i_preact)

        f_preact = (T.dot(_input, self._W_f['value']) +
                    T.dot(h_prev, self._U_f['value']) + self._b_f['value'])
        f = self._non_linearity(f_preact)

        o_preact = (T.dot(_input, self._W_o['value']) +
                    T.dot(h_prev, self._U_o['value']) + self._b_o['value'])
        o = self._non_linearity(o_preact)

        c_preact = (T.dot(_input, self._W_c['value']) +
                    T.dot(h_prev, self._U_c['value']) + self._b_c['value'])
        c = self._non_linearity2(c_preact)

        c = f * c_prev + i * c

        h = o * self._non_linearity2(c)

        return T.cast(h, dtype=config.floatX), T.cast(c, dtype=config.floatX)

    def output(self, x, prob=True):
        """ Return output of layers

        Parameters
        ----------
        x : theano.tensor.matrix
            Input sample.

        prob : bool
            Flag for changing behavior of some layers.

        Returns
        -------
        theano.tensor.matrix
            Returns the output layers according to above equation:

            .. math::  Layer(x) = activation(Wx + b)
        """

        outputs_info = [T.zeros((x.shape[1], self.get_fan_out())),
                        T.zeros((x.shape[1], self.get_fan_out()))]

        if x.ndim > 2:
            x = x.flatten(2)

        if self._mask is None:
            result, updates = scan(self._step,
                                   sequences=[x],
                                   outputs_info=outputs_info)
        else:
            result, updates = scan(self._step_mask,
                                   sequences=[self._mask, x],
                                   outputs_info=outputs_info)
        return result[0]
