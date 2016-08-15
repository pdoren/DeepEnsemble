from libML.MLearn import MLearn
from libML.trainers.Trainer import Trainer


class Learner:
    def __init__(self, learner, class_trainer, **kwargs):

        if not issubclass(class_trainer, Trainer):
            raise ValueError("Incorrect Class Trainer")

        if not isinstance(learner, MLearn):
            raise ValueError("Incorrect Class Learner")

        self.learner = learner
        self.trainer = class_trainer(self.learner, **kwargs)

    def __eq__(self, other):
        if isinstance(other, Learner):
            return (self.learner == other.learner) and (self.trainer == other.trainer)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
