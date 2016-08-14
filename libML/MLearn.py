import theano.tensor as T


class MLearn:

    def __init__(self):
        self.params = []

    def reset(self):
        """
        Reset params
        """
        raise NotImplementedError

    def get_target(self, _target):
        return _target

    def output(self, _input):
        raise NotImplementedError

    def predict(self, _input):
        raise NotImplementedError

    def fit(self, _input, _target):
        raise NotImplementedError

    def get_cost_function(self, cost, _input, _target, kernel_size):
        if cost == "MSE":
            return self.mse(_input, _target)
        elif cost == "MCC":
            return self.mcc(_input, _target, kernel_size)
        elif cost == "MEE":
            return self.mee(_input, _target, kernel_size)
        elif cost == "NLL":
            return self.neg_log_likelihood(_input, _target)
        else:
            raise ValueError("Incorrect cost function, options are MSE, MCC, MEE or NLL")

    def mse(self, _input, _target):
        """
        Compute MSE error between target and output prediction

        :type _input: theano.tensor.dmatrix
        :param _input: input
        :type _target: theano.tensor.dmatrix
        :param _target: target
        :return: Return MSE
        """
        e = self.output(_input) - _target
        return T.mean(T.power(e, 2.0))

    def mcc(self, _input, _target, s):
        """
        Compute the MCC

        :type _input: theano.tensor.dmatrix
        :param _input: input
        :type _target: theano.tensor.dmatrix
        :param _target: target
        :type s: float
        :param s: Standard deviation of Gaussian Kernel
        :return: Return MCC
        """
        e = self.output(_input) - _target
        return -T.mean(T.exp(-0.5 * T.power(e, 2.0) / s ** 2))

    def mee(self, _input, _target, s):
        """
        Compute the MEE

        :type _input: theano.tensor.dmatrix
        :param _input: input
        :type _target: theano.tensor.dmatrix
        :param _target: example
        :type s: float
        :param s: Standard deviation of Gaussian Kernel
        :return: Return MEE
        """
        e = self.output(_input) - _target
        de = T.tile(e, (e.shape[0], 1, 1))
        de = de - T.transpose(de, axes=(1, 0, 2))
        return -T.log(T.mean(T.exp(-0.5 * T.power(de, 2.0) / s ** 2)))

    def neg_log_likelihood(self, _input, _target):
        """
        Compute the negative means of errors between target and output prediction

        :type _input: theano.tensor.dmatrix
        :param _input: input
        :type _target: theano.tensor.dmatrix
        :param _target: example
        :return: Return the negative means of errors
        """
        labels = T.argmax(_target, axis=1)
        pred_y = self.output(_input)
        return -T.mean(T.log(T.power(pred_y, 2.0))[T.arange(_target.shape[0]), labels])
