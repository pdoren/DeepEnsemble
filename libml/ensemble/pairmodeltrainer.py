from libml.model import Model
from libml.trainers.trainer import Trainer


class PairModelTrainer:
    def __init__(self, model, class_trainer, **kwargs):
        """ Pair Model and Trainer.

        Parameters
        ----------
        model: Model
            Model.

        class_trainer: Trainer
            Trainer.

        kwargs
            Another parameters.

        Raises
        ------
        If the 'class_trainer' doesn't of type Trainer and the 'model' doesn't of type Model.

        """
        if not issubclass(class_trainer, Trainer):
            raise ValueError("Incorrect Class Trainer")

        if not isinstance(model, Model):
            raise ValueError("Incorrect Class Learner")

        self.model = model
        self.trainer = class_trainer(self.model, **kwargs)

    def reset(self):
        """ Reset parameters of the models
        """
        self.model.reset()

    def __eq__(self, other):
        """ Compare if the 'other' has the same type model and trainer.

        Parameters
        ----------
        other:
            Pair model and trainer for comparison.

        Returns
        -------
        Returns True if the 'other' pair has the same type of model and trainer, False otherwise.
        """
        if isinstance(other, PairModelTrainer):
            return (self.model == other.model) and (self.trainer == other.trainer)
        else:
            return False

    def __ne__(self, other):
        """ Compare if the 'other' doesn't have the same type model and trainer.

        Parameters
        ----------
        other:
            Pair model and trainer for comparison.

        Returns
        -------
        Returns True if the 'other' pair doesn't have the same type of model and trainer, False otherwise.
        """
        return not self.__eq__(other)
