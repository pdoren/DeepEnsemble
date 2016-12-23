import theano.tensor as T
import numpy as np

__all__ = ['ActivationFunctions', 'ITLFunctions', 'DiversityFunctions']

sqrt2pi = T.constant(2.50662827)  # sqrt(2 * pi)


class ActivationFunctions:
    """ Static class with common useful activation functions.
    """

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
    def softmax(x, alpha=1.):
        """ Softmax function.

        .. math:: \\varphi(\\mathbf{x'=\\alpha \\cdot x})_j = \\frac{e^{\mathbf{x'}_j}}{\sum_{k=1}^K e^{\mathbf{x'}_k}}

        Parameters
        ----------
        x : float
            Input sample.

        alpha : float
            The scale parameter.

        Returns
        -------
        function
            Returns the output of the softmax function.
        """

        return T.nnet.softmax(x * alpha)

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
    """ Static class with useful ITL (Information Theoretic Learning) functions.
    """

    @staticmethod
    def entropy(px):
        """ Entropy.

        Parameters
        ----------
        px : float
            Probability of random variable X.

        Returns
        -------
        float
            Returns entropy.
        """
        return -T.sum(px * T.log(px))

    @staticmethod
    def cross_entropy(px, py):
        """ Cross entropy.

        Parameters
        ----------
        px : float
            Probability of random variable X.

        py : float
            Probability of random variable Y.

        Returns
        -------
        float
            Returns cross entropy.
        """
        return -T.sum(px * T.log(py))

    @staticmethod
    def mutual_information(px1, px2, px1x2):
        """

        Parameters
        ----------
        px1
        px2
        px1x2

        Returns
        -------

        """
        return T.sum(px1x2 * (T.log(px1x2) - T.log(px1 * px2)))

    @staticmethod
    def norm(x, s):
        """ Normal.

        Parameters
        ----------
        x : theano.tensor.matrix
            Input data.

        s : float
            Deviation standard.

        Returns
        -------
        theano.tensor.matrix
            Returns normal.
        """
        return T.exp(- T.power(x - T.mean(x), 2.0) / (T.constant(2.0) * T.power(s, 2.0))) / (sqrt2pi * s)

    @staticmethod
    def kernel_gauss(x, s):
        """ Gaussian Kernel.

        Parameters
        ----------
        x : theano.tensor.matrix
            Input data.

        s : float
            Deviation standard.

        Returns
        -------
        theano.tensor.matrix
            Returns Gaussian Kernel.
        """
        norm_L2 = T.sum(T.power(x, 2), axis=-1)
        return T.exp(- norm_L2 / (2.0 * T.power(s, 2))) / (sqrt2pi * s)

    @staticmethod
    def silverman(x, N, d):
        """ Silverman

        Parameters
        ----------
        x : theano.tensor.matrix
            Input data.

        N : int
            Number of data.

        d : int
            Dimension of data.

        Returns
        -------
        theano.tensor.scalar
            Returns a size kernel computed with Silverman Rule.
        """
        K = T.power(4.0 / (N * (2.0 * d + 1.0)), 1.0 / (d + 4.0))
        return T.std(x) * K

    @staticmethod
    def information_potential(x, kernel, s):
        """ Information Potential.

        Parameters
        ----------
        x : theano.tensor.matrix
            Input data.

        kernel : callable
            Kernel function.

        s : float
            Size of kernel.

        Returns
        -------
        theano.tensor.matrix
            Returns Information Potential.
        """
        dx = T.tile(x, (x.shape[0], 1, 1))
        dx = dx - T.transpose(dx, axes=(1, 0, 2))
        return T.mean(kernel(dx, s))

    @staticmethod
    def error_rate(y, t):
        """ Error Rates for classification models.

        Assume that vector y and t have the same shape and length.

        Parameters
        ----------
        y : numpy.array or theano.matrix
            Array with Predictions.

        t : numpy.array or theano.matrix
            Array with Targets.

        Returns
        -------
        theano.tensor.scalar
            Returns a float with error rates.
        """
        M = T.prod(t.shape)  # Total elements
        N = t.shape[0]  # Total samples
        S = t.shape[1]  # Size vector sample
        return (M - T.sum(T.eq(y, t))) / (N * (S - 1))

    @staticmethod
    def _get_cip(Y, kernel, s):
        DY = []
        for y in Y:
            dy = T.tile(y, (y.shape[0], 1, 1))
            dy = dy - T.transpose(dy, axes=(1, 0, 2))
            DY.append(dy)

        DYK = [kernel(dy, s) for dy in DY]

        V_J = T.mean(np.prod(DYK))

        V_k = [T.mean(dyk) for dyk in DYK]

        V_nc = T.mean(np.prod([T.mean(dyk, axis=1) for dyk in DYK]))

        V_M = np.prod(V_k)

        return V_nc, V_J, V_M

    @staticmethod
    def cross_information_potential(Y, kernel, s, normalize=True):
        V_nc, V_J, V_M = ITLFunctions._get_cip(Y, kernel, s)

        if normalize:
            return T.power(V_nc, 2) / (V_J * V_M)
        else:
            return V_nc

    @staticmethod
    def mutual_information_ed(Y, kernel, s, normalize=True):
        V_nc, V_J, V_M = ITLFunctions._get_cip(Y, kernel, s)

        if normalize:
            return V_J - 2 * V_nc + V_M
        else:
            return V_nc


class DiversityFunctions:
    """ Static class with useful diversity functions (Ensembles diversity).
    """

    @staticmethod
    def ambiguity(_input, model, ensemble):
        """ Ambiguity of a model and its Ensemble.

        Parameters
        ----------
        _input : numpy.array or theano.matrix
            Input sample.

        model : Model
            Model.

        ensemble : EnsembleModel
            Ensemble.

        Returns
        -------
        theano.tensor
            Returns ambiguity.
        """
        return T.power(model.output(_input) - ensemble.output(_input), 2.0)

    @staticmethod
    def mean_ambiguity(_input, model, ensemble):
        """ Mean ambiguity of a model and its Ensemble.

        Parameters
        ----------
        _input : numpy.array or theano.matrix
            Input sample.

        model : Model
            Model.

        ensemble : EnsembleModel
            Ensemble.

        Returns
        -------
        theano.tensor
            Returns mean ambiguity.
        """
        return T.mean(DiversityFunctions.ambiguity(_input, model, ensemble))

    @staticmethod
    def bias(_input, ensemble, _target):
        """ Bias among outputs of models in Ensemble.

        Parameters
        ----------
        _input : numpy.array or theano.matrix
            Input sample.

        ensemble : EnsembleModel
            Ensemble.

        _target : numpy.array or theano.matrix
            Target sample.

        Returns
        -------
        theano.tensor
            Returns bias.
        """
        sum_e = 0.0
        for model_j in ensemble.get_models():
            sum_e += (model_j.output(_input) - _target)

        return T.power(sum_e, 2.0)

    @staticmethod
    def variance(_input, ensemble):
        """ Variance among outputs of models in Ensemble.

        Parameters
        ----------
        _input : numpy.array or theano.matrix
            Input sample.

        ensemble : EnsembleModel
            Ensemble.

        Returns
        -------
        theano.tensor
            Returns variance.
        """
        sum_e = 0.0
        for model_j in ensemble.get_models():
            sum_e += (model_j.output(_input) - ensemble.output(_input))

        return T.power(sum_e, 2.0)
