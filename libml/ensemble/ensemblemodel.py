from libml.ensemble.combiner.averagecombiner import AverageCombiner
from libml.ensemble.combiner.modelcombiner import ModelCombiner
from libml.models.model import Model
import numpy as np


class EnsembleModel(Model):
    def __init__(self):
        """ Base class Ensemble Model.
        """
        super(EnsembleModel, self).__init__(n_input=0, n_output=0)
        self.combiner = AverageCombiner()
        self.list_models_ensemble = []

    def set_combiner(self, combiner):
        """ Setter combiner.

        Parameters
        ----------
        combiner: ModelCombiner
            Object ModelCombiner for combining model outputs in ensemble.
        """
        self.combiner = combiner

    def append_model(self, new_model):
        """ Add model to ensemble.

        Parameters
        ----------
        new_model: Model
            Model.

        Raises
        ------
        If the model is the different type of the current list the models, it is generated an error.

        """
        if len(self.list_models_ensemble) == 0:
            # copy data model
            self.n_input = new_model.n_input
            self.n_output = new_model.n_output
            self.type_model = new_model.type_model
            self.target_labels = new_model.target_labels

            self.list_models_ensemble.append(new_model)
        elif self.list_models_ensemble[0] == new_model:
            self.list_models_ensemble.append(new_model)
        else:
            str_error = ''
            if self.list_models_ensemble[0].model.n_input != new_model.n_input:
                str_error += 'different input, '
            if self.list_models_ensemble[0].model.n_output != new_model.n_output:
                str_error += 'different output, '
            if self.list_models_ensemble[0].model.type_model is not new_model.type_model:
                str_error += 'different type learner, '
            if self.list_models_ensemble[0].model.target_labels is not new_model.target_labels:
                str_error += 'different target labels, '

            raise ValueError('Incorrect Learner: ' + str_error[0:-2] + '.')

    def get_num_models(self):
        """ Get number of the Ensemble's models

        Returns
        -------
        int
            Returns current number of models in the Ensemble.
        """
        return len(self.list_models_ensemble)

    def reset(self):
        """ Reset parameters of the ensemble's models.
        """
        for model in self.list_models_ensemble:
            model.reset()

    def output(self, _input):
        """ Output of ensemble model.

        Parameters
        ----------
        _input: theano.tensor.matrix
            Input sample.

        Returns
        -------
        theano.tensor.matrix
        Returns of combiner the outputs of the different the ensemble's models.

        """
        return self.combiner.output(self.list_models_ensemble, _input)

    def compile(self, **kwargs):
        """ Compile ensemble's models.

        Parameters
        ----------
        kwargs
            Compilers parameters of models.
        """
        for model in self.list_models_ensemble:
            model.compile(**kwargs)

    def fit(self, _input, _target, max_epoch, validation_jump, **kwargs):
        """ Training ensemble.

        Parameters
        ----------
        _input : theano.tensor.matrix
            Training Input sample.

        _target : theano.tensor.matrix
            Training Target sample.

        max_epoch: int
            Number of epoch for training.

        validation_jump: int
            Number of times until doing validation jump.

        kwargs
            Other parameters.

        Returns
        -------
        numpy.array[float]
            Returns training cost for each batch.
        """
        n_models = len(self.list_models_ensemble)
        train_cost = np.zeros(shape=(max_epoch, n_models))

        for i, model in enumerate(self.list_models_ensemble):
            train_cost[:, i] = model.fit(_input=_input, _target=_target, max_epoch=max_epoch,
                                         validation_jump=validation_jump, **kwargs)

        self.combiner.update_parameters(self, _input=_input, _target=_target)
        return train_cost

    def add_cost_ensemble(self, fun_cost, **kwargs):
        """ Adds cost function for each models in Ensemble.

        Parameters
        ----------
        fun_cost: str
            Name of cost function.

        kwargs
            Other parameters.
        """
        for i, model in enumerate(self.list_models_ensemble):
            model.append_cost(fun_cost=fun_cost, index_current_model=i, ensemble=self, **kwargs)
