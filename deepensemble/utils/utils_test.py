import os
import matplotlib.pylab as plt

from ..metrics import *
from ..models import *
from .logger import *

__all__ = ['test_classifier']


def test_classifier(_dir, cls, input_train, target_train, input_test, target_test, folds=25, max_epoch=300, **kwargs):
    """ Test on classifier.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)

    metrics = FactoryMetrics.get_metric(cls)

    best_params = None
    best_score = 0
    for i in range(folds):
        metric = cls.fit(input_train, target_train, max_epoch=max_epoch, **kwargs)

        # Compute metrics
        score = metrics.append_prediction(target_test, cls.predict(input_test))
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

    Logger().print('wait ... ', end='')
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

    return best_score, cls
