import os
import matplotlib.pylab as plt

from ..metrics import *
from ..models import *
from .logger import *

__all__ = ['test_classifier', 'test_models']


def test_classifier(_dir, cls, input_train, target_train, input_test, target_test, min_score_valid=0.7, folds=25,
                    max_epoch=300, **kwargs):
    """ Test on classifier.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)

    metrics = FactoryMetrics.get_metric(cls)

    best_params = None
    best_score = 0
    i = 0
    invalid_training = 0
    while i < folds:
        metric = cls.fit(input_train, target_train, max_epoch=max_epoch, **kwargs)

        # Compute metrics
        score = metrics.append_prediction(input_test, target_test)

        if score < min_score_valid:
            Logger().print('Invalid training (fold: %d), score %0.4f < %.4f' % (i, score, min_score_valid))
            if invalid_training > 0.25 * folds:
                min_score_valid = 0.0
            else:
                invalid_training += 1
                # Reset parameters
                cls.reset()
                continue
        else:
            i += 1

        metrics.append_metric(metric)

        # Save the best params
        if score > best_score:
            best_params = cls.save_params()
            best_score = score
        elif score == best_params:
            score_curr = metrics.get_score_prediction(target_train, cls.predict(input_train))
            params_curr = cls.save_params()
            cls.load_params(best_params)
            score_best = metrics.get_score_prediction(target_train, cls.predict(input_train))

            if score_curr > score_best:
                best_params = params_curr

        # Reset parameters
        cls.reset()

    # Load the best params
    if best_params is not None:
        cls.load_params(best_params)

    # Save classifier
    cls.save(_dir + '%s_classifier.pkl' % cls.get_name())

    # Compute and Show metrics
    plt.style.use('ggplot')
    metrics.classification_report()
    if isinstance(cls, EnsembleModel):
        metrics.diversity_report()

    Logger().print('The best score: %.4f' % best_score)
    Logger().print('wait ... ', end='', flush=True)

    fig_ = [(metrics.plot_confusion_matrix(), 'confusion_matrix'),
            (metrics.plot_confusion_matrix_prediction(target_train, cls.predict(input_train)),
             'confusion_matrix_best_train'),
            (metrics.plot_confusion_matrix_prediction(target_test, cls.predict(input_test)),
             'confusion_matrix_best_test'), (metrics.plot_cost(max_epoch), 'Cost'),
            (metrics.plot_costs(max_epoch), 'Costs'), (metrics.plot_scores(max_epoch), 'Scores')]

    if isinstance(cls, EnsembleModel):
        for key in metrics.get_models_metric():
            _model = metrics.get_models_metric()[key].get_model()
            fig_.append((metrics.get_models_metric()[key].plot_costs(max_epoch), 'Cost_' + _model.get_name()))
            fig_.append((metrics.get_models_metric()[key].plot_scores(max_epoch), 'Cost_' + _model.get_name()))

    for fig, name in fig_:
        fig.savefig(_dir + name + '.pdf', format='pdf', dpi=1200)
        fig.clf()

    Logger().save_buffer(_dir + 'info.txt')

    print(':) OK')

    return best_score, metrics, cls


def test_models(models, input_train, target_train, input_valid, target_valid,
                classes_labels, name_db, desc, col_names,
                folds, **kwargs):
    """ Training Ensemble with American Credit Card data base.
    """

    for model in models:
        n_input = input_train.shape[1]
        n_output = len(classes_labels)

        dir_db = name_db + '/'

        # Print Info Data and Training
        Logger().reset()
        Logger().print('Model:\n %s | in: %d, out: %d\n info:\n %s' %
                       (model.get_name(), n_input, n_output, model.get_info()))
        Logger().print('Data (%s):\n DESC: %s.\n Features(%d): %s\n Classes(%d): %s' %
                       (name_db, desc, n_input, col_names, n_output, classes_labels))
        Logger().print('Training:\n total data: %d | train: %d, validation: %d ' %
                       (input_train.shape[0] + input_valid.shape[0], input_train.shape[0], input_valid.shape[0]))
        Logger().print(' folds: %d | Epoch: %d, Batch Size: %d ' %
                       (folds, kwargs['max_epoch'], kwargs['batch_size']))
        dir = dir_db + model.get_name() + '/'
        test_classifier(dir, model, input_train, target_train, input_valid, target_valid,
                        folds=folds, **kwargs)

        
