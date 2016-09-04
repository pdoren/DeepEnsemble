from libml.model import Model


class Trainer:
    def __init__(self, model):
        """ Base class for training models.

        Parameters
        ----------
        model: Model
            Model for training.

        """
        self.model = model

    def __eq__(self, other):
        """ Evaluate if the 'other' trainer model has the same form.

        Parameters
        ----------
        other: Trainer
            Trainer model.

        Notes
        -----
        Until now only is compared the models between trainers.

        Returns
        -------
        bool
        Return True if the 'other' trainer model has the same form oneself, False otherwise.
        """
        if isinstance(other, Trainer):
            return self.model == other.model
        else:
            return False

    def __ne__(self, other):
        """ Evaluate if the 'other' trainer model doesn't have the same form.

        Parameters
        ----------
        other: Trainer
        Trainer model.

        Notes
        -----
        Until now only is compared the models between trainers.

        Returns
        -------
        bool
        Return True if the 'other' trainer model doesn't have the same form oneself, False otherwise.
        """
        return not self.__eq__(other)

    def trainer(self, input_train, target_train, input_test, target_test, max_epoch, validation_jump):
        """ Training model

        Parameters
        ----------
        input_train: theano.tensor.array
            Input training set.

        target_train: theano.tensor.array
            Target training set.

        input_test: theano.tensor.array
            Input test set.

        target_test: theano.tensor.array
            Target test set.

        max_epoch: int
            Number of epoch for training.

        validation_jump: int
            Number of times until doing validation jump.

        Notes
        -----
        Must be implemented.

        Returns
        -------
        tuple
        Returns the training cost, testing cost and the best testing prediction.

        """
        pass
