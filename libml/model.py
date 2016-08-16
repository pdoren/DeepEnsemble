import theano.tensor as T


class Model:
    def __init__(self, n_input, n_output, type_model='classifier'):

        self.n_input = n_input
        self.n_output = n_output
        self.type_model = type_model
        self.params = []
        self.target_labels = []

    def __eq__(self, other):
        if isinstance(other, Model):
            return (self.n_input == other.n_input) and (self.n_output == other.n_output) and (
                self.type_model is other.type_model) and (list(self.target_labels) == list(other.target_labels))
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def reset(self):
        """
        Reset params
        """
        raise NotImplementedError

    def translate_target(self, _target):
        return _target

    def output(self, _input):
        raise NotImplementedError

    def translate_output(self, _output):
        raise NotImplementedError

    def predict(self, _input):
        output = self.output(_input)
        return self.translate_output(output)

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
        """ Compute MSE error between target and output prediction

        Parameters
        ----------
        _input
        _target

        Returns
        -------

        """
        e = self.output(_input) - _target
        return T.mean(T.power(e, 2.0))

    def mcc(self, _input, _target, s):
        """ Compute the MCC

        Parameters
        ----------
        _input
        _target
        s

        Returns
        -------

        """
        e = self.output(_input) - _target
        return -T.mean(T.exp(-0.5 * T.power(e, 2.0) / s ** 2))

    def mee(self, _input, _target, s):
        """ Compute the MEE

        Parameters
        ----------
        _input
        _target
        s

        Returns
        -------

        """
        e = self.output(_input) - _target
        de = T.tile(e, (e.shape[0], 1, 1))
        de = de - T.transpose(de, axes=(1, 0, 2))
        return -T.log(T.mean(T.exp(-0.5 * T.power(de, 2.0) / s ** 2)))

    def neg_log_likelihood(self, _input, _target):
        """ Compute the negative means of errors between target and output prediction

        Parameters
        ----------
        _input
        _target

        Returns
        -------

        """
        labels = T.argmax(_target, axis=1)
        return -T.mean(T.log(T.power(self.output(_input), 2.0))[T.arange(_target.shape[0]), labels])
