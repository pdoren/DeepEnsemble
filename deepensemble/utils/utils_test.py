import os
import numpy as np
from sklearn.metrics import *
import matplotlib.pylab as plt

from ..metrics import *
from ..models import *
from .logger import *

__all__ = ['test_classifier', 'test_models',
           'plot_hist_train_test',
           'plot_scores_classifications']


def make_dirs(_dir):
    """ Makes directories.

    Parameters
    ----------
    _dir : str
        Path of directory.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def test_classifier(_dir, cls, input_train, target_train, input_test, target_test, min_score_test=0.7, folds=25,
                    max_epoch=300, **kwargs):
    """ Test on classifier.
    """
    make_dirs(_dir)

    metrics = FactoryMetrics.get_metric(cls)

    best_params = None
    best_score = 0
    i = 0
    invalid_training = 0
    while i < folds:
        metric = cls.fit(input_train, target_train, max_epoch=max_epoch, **kwargs)

        # Compute metrics
        score = metrics.append_prediction(input_test, target_test)

        if score < min_score_test:
            Logger().print('Invalid training (fold: %d), score %0.4f < %.4f' % (i, score, min_score_test))
            if invalid_training > 0.25 * folds:
                min_score_test = 0.0
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
    metrics.classification_report()
    if isinstance(cls, EnsembleModel):
        metrics.diversity_report()

    # Save Metrics
    metrics.save(_dir + '%s_metrics.pkl' % cls.get_name())

    Logger().print('The best score: %.4f' % best_score)
    Logger().print('wait .', end='', flush=True)

    fig_ = [(metrics.plot_confusion_matrix(), 'confusion_matrix'),
            (metrics.plot_confusion_matrix_prediction(target_train, cls.predict(input_train)),
             'confusion_matrix_best_train'),
            (metrics.plot_confusion_matrix_prediction(target_test, cls.predict(input_test)),
             'confusion_matrix_best_test'), (metrics.plot_cost(max_epoch), 'Cost'),
            (metrics.plot_costs(max_epoch), 'Costs'), (metrics.plot_scores(max_epoch), 'Scores')]

    if isinstance(cls, EnsembleModel):
        for key in metrics.get_models_metric():
            _model = metrics.get_models_metric()[key].get_model()

            dir_model_costs = 'costs_models/'
            dir_model_scores = 'scores_models/'
            make_dirs(_dir + dir_model_costs)
            make_dirs(_dir + dir_model_scores)

            fig_.append((metrics.get_models_metric()[key].plot_costs(max_epoch),
                         dir_model_costs + 'cost_' + _model.get_name()))
            fig_.append((metrics.get_models_metric()[key].plot_scores(max_epoch),
                         dir_model_scores + 'score_' + _model.get_name()))

        # Diversity current model in metrics, so best model
        dir_diversity = 'diversity/'
        make_dirs(_dir + dir_diversity)
        fig_ += metrics.plot_diversity(input_test, target_test, prefix=dir_diversity + 'diversity')

    for fig, name in fig_:
        if fig is not None:
            fig.savefig(_dir + name + '.pdf', format='pdf', dpi=1200)
            plt.close(fig)
            Logger().print('.', end='', flush=True)

    Logger().save_buffer(_dir + 'info.txt')

    print(':) OK')

    return best_score, metrics, cls


def test_models(models, input_train, target_train, input_valid, target_valid,
                classes_labels, name_db, desc, col_names,
                folds, **kwargs):
    """ Training Ensemble with American Credit Card data base.
    """

    data_models = []

    for _model in models:
        n_input = input_train.shape[1]
        n_output = len(classes_labels)

        dir_db = name_db + '/'

        # Print Info Data and Training
        Logger().reset()
        Logger().print('Model:\n %s | in: %d, out: %d\n info:\n %s' %
                       (_model.get_name(), n_input, n_output, _model.get_info()))
        Logger().print('Data (%s):\n DESC: %s.\n Features(%d): %s\n Classes(%d): %s' %
                       (name_db, desc, n_input, col_names, n_output, classes_labels))
        Logger().print('Training:\n total data: %d | train: %d, validation: %d ' %
                       (input_train.shape[0] + input_valid.shape[0], input_train.shape[0], input_valid.shape[0]))
        Logger().print(' folds: %d | Epoch: %d, Batch Size: %d ' %
                       (folds, kwargs['max_epoch'], kwargs['batch_size']))
        _dir = dir_db + _model.get_name() + '/'
        data_models.append(test_classifier(_dir, _model, input_train, target_train, input_valid, target_valid,
                                           folds=folds, **kwargs))

    return data_models


def plot_hist_train_test(train_means, test_means, ylabel, title, labels):
    """

    Parameters
    ----------
    train_means
    test_means
    ylabel
    title
    labels
    """
    fig, ax = plt.subplots()

    ind = np.arange(len(labels))
    width = 0.25

    # the bars
    rects_1 = ax.bar(ind, train_means, width,
                     color='blue', alpha=0.5)

    rects_2 = ax.bar(ind + width, test_means, width,
                     color='red', alpha=0.5)

    # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    xTickMarks = labels
    ax.set_xticks(ind + width)
    xTickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xTickNames, rotation=45, fontsize=10)

    # add a legend
    ax.legend((rects_1[0], rects_2[0]), ('Train', 'Test'))

    def auto_label(rects):
        """ Generate labels.

        Parameters
        ----------
        rects
        """
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%.2f' % height,
                    ha='center', va='bottom')

    auto_label(rects_1)
    auto_label(rects_2)

    plt.tight_layout()


def plot_hist_train_test2(train_means, train_std, test_means, test_std, ylabel, title, labels):
    """

    Parameters
    ----------
    train_means
    train_std
    test_means
    test_std
    ylabel
    title
    labels
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ind = np.arange(len(train_means))
    width = 0.25

    # the bars
    rects_1 = ax.bar(ind, train_means, width,
                     color='blue',
                     yerr=train_std,
                     error_kw=dict(elinewidth=2, ecolor='red'))

    rects_2 = ax.bar(ind + width, test_means, width,
                     color='red',
                     yerr=test_std,
                     error_kw=dict(elinewidth=2, ecolor='blue'))

    # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    ax.set_ylim(0, 45)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    xTickMarks = labels
    ax.set_xticks(ind + width)
    xTickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xTickNames, rotation=45, fontsize=10)

    # add a legend
    ax.legend((rects_1[0], rects_2[0]), ('Train', 'Test'))
    plt.tight_layout()


def plot_scores_classifications(models, input_train, target_train, input_test, target_test, classes_labels):
    """

    Parameters
    ----------
    models
    input_train
    target_train
    input_test
    target_test
    classes_labels
    """
    metrics = {'Accuracy': accuracy_score,
               'F1-Score': f1_score,
               'Precision': precision_score,
               'Recall': recall_score}

    classes_labels = list(classes_labels)
    target_train = [classes_labels.index(i) for i in target_train]
    target_test = [classes_labels.index(i) for i in target_test]

    predictions = []
    labels = []
    for _model in models:
        pred_train = _model.predict(input_train)
        pred_test = _model.predict(input_test)
        pred_train = [classes_labels.index(i) for i in pred_train]
        pred_test = [classes_labels.index(i) for i in pred_test]
        predictions.append({'train': pred_train, 'test': pred_test})
        labels.append(_model.get_name())

    for key in metrics:
        train_means = []
        test_means = []

        metric = metrics[key]

        for data in predictions:
            # noinspection PyTypeChecker
            train_means.append(metric(target_train, data['train']))
            # noinspection PyTypeChecker
            test_means.append(metric(target_test, data['test']))

        plot_hist_train_test(train_means, test_means, 'score', key, labels)
