import theano.tensor as T
from theano import shared
import numpy as np

__all__ = ['ActivationFunctions', 'ITLFunctions', 'DiversityFunctions']

sqrt2pi = T.constant(2.50662827)  # sqrt(2 * pi)
sqrt2 = 1.41421356237  # sqrt(2)


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
    def mutual_information(px1, px2, px1x2, eps=1e-6):
        """ Mutual Information.

        Parameters
        ----------
        px1 : theano.tensor.matrix
            Probability of random variable X1.

        px2 : theano.tensor.matrix
            Probability of random variable X2.

        px1x2 : theano.tensor.matrix
            Joint Probability between X1 and X2.
        
        eps : float
            Constant for avoiding numerical errors.

        Returns
        -------
        float
            Returns Mutual Information.
        """
        return T.sum((px1x2 + eps) * (T.log(px1x2 + eps) - T.log(px1 * px2 + eps)))

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
    def kernel_gauss(x, y, s):
        """ Gaussian Kernel.

        Parameters
        ----------
        x : theano.tensor.matrix
            Input data.
        
        y : theano.tensor.matrix
            Input data.

        s : float
            Deviation standard.

        Returns
        -------
        theano.tensor.matrix
            Returns Gaussian Kernel.
        """
        divisor = T.cast(2.0 * T.sqr(s), T.config.floatX)

        normx = T.sqr(x - y)

        z = 1. / (T.power(sqrt2pi, normx.shape[-1]) * s)

        return T.squeeze(T.cast(T.exp(-normx.sum(axis=-1) / divisor) * z, T.config.floatX))

    @staticmethod
    def kernel_laplace(x, y, s):
        """ Gaussian Kernel.

        Parameters
        ----------
        x : theano.tensor.matrix
            Input data.
            
        y : theano.tensor.matrix
            Input data.

        s : float
            Deviation standard.

        Returns
        -------
        theano.tensor.matrix
            Returns Gaussian Kernel.
        """
        divisor = T.cast(2.0 * T.sqr(s), T.config.floatX)

        diffx = x - y

        z = 1. / (T.power(sqrt2pi, diffx.shape[-1]) * s)

        return T.squeeze(T.cast(T.exp(-diffx.sum(axis=-1) / divisor) * z, T.config.floatX))

    @staticmethod
    def kernel_gauss_numpy(x, y, s):
        """ Gaussian Kernel.

        Parameters
        ----------
        x : theano.tensor.matrix
            Input data.
        
        y : theano.tensor.matrix
            Input data.

        -------
        s : float
            Deviation standard.

        Returns
        -------
        theano.tensor.matrix
            Returns Gaussian Kernel.
        """
        # noinspection PyTypeChecker
        divisor = np.array(2.0 * (s ** 2), T.config.floatX)
        normx = (x - y) ** 2
        z = 1. / (np.power(sqrt2pi, normx.shape[-1], 2) * s)

        return np.exp(-normx.sum(axis=-1) / divisor) * z

    @staticmethod
    def kernel_gauss_diff(diff_x, s):
        divisor = T.cast(2.0 * T.sqr(s), T.config.floatX)

        normx = T.sqr(diff_x)

        z = 1. / (T.power(sqrt2pi, normx.shape[-1]) * s)

        return T.squeeze(T.cast(T.exp(-normx.sum(axis=-1) / divisor) * z, T.config.floatX))

    @staticmethod
    def silverman(x):
        """ Silverman

        Parameters
        ----------
        x : theano.tensor.matrix
            Input data.

        Returns
        -------
        theano.tensor.scalar
            Returns a size kernel computed with Silverman Rule.
        """
        if isinstance(x, np.ndarray):
            x = shared(x)
        N = x.shape[0]
        d = x.shape[-1]
        K = T.power(4.0 / (N * (2.0 * d + 1.0)), 1.0 / (d + 4.0))
        return T.cast(T.std(x) * K, T.config.floatX)

    @staticmethod
    def get_diff(X):
        """ Compute difference among each element in each set.

        Parameters
        ----------
        X : list
            List of sets.

        Returns
        -------
        list
            Returns a list with the differences of each set.

        """
        DT = []
        for t in X:
            dt = T.tile(t, (t.shape[0], 1, 1))
            dt = T.transpose(dt, axes=(1, 0, 2)) - dt
            DT.append(dt)
        return DT

    @staticmethod
    def get_prod(x, y):
        """ Compute difference among each element of x and y sets.

        Parameters
        ----------
        x
            Set 1.
        
        y
            Set 2.

        Returns
        -------
        list
            Returns a list with the differences of each set.

        """
        dx = T.tile(x, (x.shape[0], 1, 1))
        dy = T.tile(y, (y.shape[0], 1, 1))
        dt = dx * dy
        return dt

    # noinspection PyUnresolvedReferences
    @staticmethod
    def mutual_information_parzen(x, y, s):
        """ Mutual Information estimate with Parzen Windows.

        Parameters
        ----------
        x : theano.tensor.matrix
            The first input data.

        y : theano.tensor.matrix
            The second input data.

        s : float
            Deviation standard.

        Returns
        -------
        theano.tensor.scalar
            Returns mutual information
        """
        kernel = ITLFunctions.kernel_gauss_diff

        DT = ITLFunctions.get_diff([x, y])
        DTK = [kernel(dt, s) for dt in DT]

        px = T.mean(DTK[0], axis=-1)
        py = T.mean(DTK[1], axis=-1)

        dx = T.tile(DTK[0], (DTK[0].shape[0], 1, 1))
        dy = T.tile(DTK[1], (DTK[1].shape[0], 1, 1))

        dt = dx * T.transpose(dy, axes=(1, 0, 2))

        pxy = T.mean(dt, axis=-1)

        # Normalization
        px = T.tile(px, (px.shape[0], 1)) / T.sum(px)
        py = T.tile(py, (py.shape[0], 1)) / T.sum(py)
        pxy = pxy / T.sum(pxy)

        return ITLFunctions.mutual_information(px, py, pxy)

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
        return T.mean(kernel(dx, sqrt2 * s))

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
    def get_cip(X, y, s):

        kernel = ITLFunctions.kernel_gauss_diff

        Xy = X + [y]

        DY = ITLFunctions.get_diff(Xy)

        DYK = [kernel(dy, np.sqrt(2.0) * s) for dy in DY]

        V_J = T.mean(np.prod(DYK))

        V_k_i = [T.mean(dyk, axis=-1) for dyk in DYK]

        V_k = [T.mean(V_i) for V_i in V_k_i]

        V_M = np.prod(V_k)

        V_nc = T.mean(np.prod(V_k_i))

        return V_nc, V_J, V_M

    @staticmethod
    def get_cip_jenssen(X, y, s):

        kernel = ITLFunctions.kernel_gauss_diff

        Xy = X + [y]

        DX = ITLFunctions.get_diff(Xy)

        DY = [T.squeeze(T.sum(ITLFunctions.get_prod(x, y), axis=-1))  for x in Xy]

        DXK = [kernel(dx, np.sqrt(2.0) * s) * (1. - dy) for dx, dy in zip(DX, DY)]

        V_k_i = [T.mean(dxk, axis=-1) for dxk in DXK]

        V_nc = T.mean(np.prod(V_k_i))

        DX = ITLFunctions.get_diff(Xy)

        DYK = [kernel(dx, np.sqrt(2.0) * s) for dx in DX]

        V_J = T.mean(np.prod(DYK))

        V_k_i = [T.mean(dyk, axis=-1) for dyk in DYK]

        V_k = [T.mean(V_i) for V_i in V_k_i]

        V_M = np.prod(V_k)

        return V_nc, V_J, V_M

    @staticmethod
    def get_grad_cip(y, o, params, s):

        kernel = ITLFunctions.kernel_gauss_diff
        DY = ITLFunctions.get_diff([y])
        DX = ITLFunctions.get_diff(o)

        sigma = np.sqrt(2.0) * s

        DYK = kernel(DY, sigma)
        DDXK = T.grad(DX, params)

        # noinspection PyUnresolvedReferences
        DV_J = T.mean(DYK * DDXK)

        ind = -1

        DV_k_i = [T.mean(DYK, axis=ind), T.mean(DDXK, axis=ind)]

        DV_k = [T.mean(DV_i) for DV_i in DV_k_i]

        DV_M = np.prod(DV_k)

        DV_nc = T.mean(np.prod(DV_k_i))

        return DV_nc, DV_J, DV_M

    @staticmethod
    def get_cip_numpy(Y, s):

        kernel = ITLFunctions.kernel_gauss_diff

        DY = ITLFunctions.get_diff(Y)

        DYK = []
        for dy in DY:
            DYK.append(kernel(dy, sqrt2 * s))

        V_J = np.mean(np.prod(np.array([dyk for dyk in DYK]), axis=0))

        V_k_i = []

        for dyk in DYK:
            V_k_i.append(np.mean(dyk, axis=0))

        V_k = [np.mean(V_i) for V_i in V_k_i]

        p2 = np.prod(V_k_i, axis=0)
        V_nc = np.mean(p2)

        V_M = np.prod(V_k)

        return V_nc, V_J, V_M

    @staticmethod
    def cross_information_potential(X, y, s, dist='ED'):
        V_nc, V_J, V_M = ITLFunctions.get_cip(X, y, s)

        if dist == 'CS':
            return T.power(V_nc, 2) / (V_J * V_M)
        elif dist == 'ED':
            return V_J - 2 * V_nc + V_M
        else:
            raise ValueError('The dist must be CS or ED')

    @staticmethod
    def mutual_information_cs(X, y, s):
        V_nc, V_J, V_M = ITLFunctions.get_cip(X, y, s)

        return T.log(V_J) - 2 * T.log(V_nc) + T.log(V_M)

    @staticmethod
    def mutual_information_ed(X, y, s):
        V_nc, V_J, V_M = ITLFunctions.get_cip(X, y, s)

        return V_J - 2 * V_nc + V_M

    @staticmethod
    def annealing(sp, sm, i, num_iterations):
        return sp * T.power((sm / sp), i / num_iterations)


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
