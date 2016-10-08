import theano.tensor as T

__all__ = ['ActivationFunctions', 'ITLFunctions', 'DiversityFunctions']

sqrt2pi = T.constant(2.50662827)  # sqrt(2 * pi)


class ActivationFunctions:
    @staticmethod
    def linear(x):
        """ Linear function.

        .. math:: \\varphi(x) = x

        Parameters
        ----------
        x : float
            Input sample.

        Returns
        -------
        float
            The same value passed as input.
        """
        return x

    @staticmethod
    def softmax(x):
        """ Softmax function.

        .. math:: \\varphi(\\mathbf{x})_j = \\frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}

        Parameters
        ----------
        x : float
            Input sample.

        Returns
        -------
        float
            Returns the output of the softmax function.
        """
        return T.nnet.softmax(x)

    @staticmethod
    def sigmoid(x):
        """ Sigmoid function.

        .. math:: \\varphi(x) = \\frac{1}{1 + e^{-x}}

        Parameters
        ----------
        x : float
            Input sample.

        Returns
        -------
        float
            Returns the output of the sigmoid function.
        """
        return T.nnet.sigmoid(x)

    @staticmethod
    def tanh(x, alpha=1.0, beta=1.0):
        """ Tangent Hyperbolic function

        .. math:: \\varphi(x) = \\tanh(alpha * x) * beta

        Parameters
        ----------
        x : float
            Input sample.

        alpha : float
            The scale parameter.

        beta : float
            The scale parameter.

        Returns
        -------
        float
            Returns the output of the tanh function.
        """
        return T.tanh(x * alpha) * beta

    @staticmethod
    def relu(x, alpha=0):
        """ Relu function

        .. math:: \\varphi(x) = \\max(alpha * x, x)

        Parameters
        ----------
        x : float
            Input sample.

        alpha : float
            The scale parameter.

        Returns
        -------
        float
            Returns the output of the relu function.
        """
        return T.nnet.relu(x, alpha)

    @staticmethod
    def elu(x):
        """ Exponential Linear Unit function.

        .. math:: \\varphi(x) = (x > 0) ? x : e^x - 1

        Parameters
        ----------
        x : float
            Input sample.

        Returns
        -------
        float
            Returns the output of the elu function.
        """
        return T.switch(x > 0, x, T.exp(x) - 1)

    @staticmethod
    def softplus(x):
        """Softplus function

        .. math:: \\varphi(x) = \\log(1 + e^x)

        Parameters
        ----------
        x : float
            Input sample.

        Returns
        -------
        float
            Returns the output of the softplus function.
        """
        return T.nnet.softplus(x)


class ITLFunctions:
    @staticmethod
    def entropy(px):
        return -T.sum(px * T.log(px))

    @staticmethod
    def cross_entropy(px, py):
        return -T.sum(px * T.log(py))

    @staticmethod
    def mutual_information(px1, px2, px1x2):
        return T.sum(px1x2 * (T.log(px1x2) - T.log(px1 * px2)))

    @staticmethod
    def norm(x, s):
        return T.exp(- T.power(x - T.mean(x), 2.0) / (T.constant(2.0) * T.power(s, 2.0))) / (sqrt2pi * s)

    @staticmethod
    def silverman(x, N, d):
        K = T.power(4.0 / (N * (2.0 * d + 1.0)), 1.0 / (d + 4.0))
        return T.std(x) * K


class DiversityFunctions:
    @staticmethod
    def ambiguity(_input, model, ensemble):
        return T.power(model.output(_input) - ensemble.output(_input), 2.0)

    @staticmethod
    def mean_ambiguity(_input, model, ensemble):
        return T.mean(DiversityFunctions.ambiguity(_input, model, ensemble))

    @staticmethod
    def bias(_input, ensemble, _target):
        sum_e = 0.0
        for model_j in ensemble.get_models():
            sum_e += (model_j.output(_input) - _target)

        return T.power(sum_e, 2.0)

    @staticmethod
    def variance(_input, ensemble):
        sum_e = 0.0
        for model_j in ensemble.get_models():
            sum_e += (model_j.output(_input) - ensemble.output(_input))

        return T.power(sum_e, 2.0)
