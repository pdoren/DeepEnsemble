from libML.Model import Model
from libML.trainers.Trainer import Trainer


class PairModelTrainer:
    def __init__(self, model, class_trainer, **kwargs):

        if not issubclass(class_trainer, Trainer):
            raise ValueError("Incorrect Class Trainer")

        if not isinstance(model, Model):
            raise ValueError("Incorrect Class Learner")

        self.model = model
        self.trainer = class_trainer(self.model, **kwargs)

    def reset(self):
        self.model.reset()

    def __eq__(self, other):
        if isinstance(other, PairModelTrainer):
            return (self.model == other.model) and (self.trainer == other.trainer)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
