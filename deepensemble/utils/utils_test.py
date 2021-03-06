import os

import matplotlib.pylab as plt
import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.neighbors.kde import KernelDensity
from matplotlib import cm
from matplotlib.mlab import griddata
from scipy.interpolate import griddata as griddata2

from ..metrics import FactoryMetrics
from ..models import Model, EnsembleModel
from ..utils import Logger, Serializable, score_ensemble_ambiguity
from ..utils.utils_translation import TextTranslation
from ..utils.utils_plot import ConfigPlot

__all__ = ['cross_validation_score', 'test_model', 'load_model',
           'plot_hist_train_test',
           'plot_scores_classifications',
           'plot_pdf', 'make_dirs', 'get_scores',
           'get_best_score', 'get_mean_score', 'get_min_score',
           'get_mean_diversity']


def make_dirs(_dir):
    """ Makes directories.

    Parameters
    ----------
    _dir : str
        Path of directory.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def test_model(cls, input_train, target_train, input_test, target_test, folds=25,
               max_epoch=300, **kwargs):
    metrics = FactoryMetrics.get_metric(cls)

    best_params = None
    best_score = 0
    list_score = []
    for _ in range(folds):
        metric = cls.fit(input_train, target_train, max_epoch=max_epoch, **kwargs)

        # Compute metrics
        score_train = metrics.append_prediction(input_train, target_train)
        score_test = metrics.append_prediction(input_test, target_test, append_last_pred=True)

        metrics.append_metric(metric)

        list_score.append((score_train, score_test))
        # Save the best params
        if score_test > best_score:
            best_params = cls.save_params()
            best_score = score_test
        elif score_test == best_score:
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

    return metrics, best_score, list_score


def testing_model(_dir, cls, input_train, target_train, input_test, target_test, folds=25,
                  max_epoch=300, save_file=True, **kwargs):
    if save_file:
        make_dirs(_dir)

    metrics, best_score, list_score = test_model(cls, input_train, target_train, input_test, target_test, folds,
                                                 max_epoch, **kwargs)

    Logger().log('The best score (test): %.4f' % best_score)
    Logger().log('wait .', end='', flush=True)

    if save_file:

        # Save classifier
        cls.save(_dir + '%s_classifier.pkl' % cls.get_name())

        # Save Metrics
        metrics.save(_dir + '%s_metrics.pkl' % cls.get_name())

        fig_ = [(metrics.plot_confusion_matrix(), 'confusion_matrix'),
                (metrics.plot_confusion_matrix_prediction(target_train, cls.predict(input_train)),
                 'confusion_matrix_best_train'),
                (metrics.plot_confusion_matrix_prediction(target_test, cls.predict(input_test)),
                 'confusion_matrix_best_test'), (metrics.plot_cost(max_epoch), TextTranslation().get_str('Cost')),
                (metrics.plot_costs(max_epoch), TextTranslation().get_str('Costs')),
                (metrics.plot_scores(max_epoch), TextTranslation().get_str('Scores'))]

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

            if cls.is_classifier():
                # Diversity current model in metrics, so best model
                dir_diversity = 'diversity/'
                make_dirs(_dir + dir_diversity)
                fig_ += metrics.plot_diversity(input_test, target_test, prefix=dir_diversity + 'diversity')

        for fig, name in fig_:
            if fig is not None:
                fig.savefig(_dir + name + '.pdf', format='pdf', dpi=1200)
                plt.close(fig)
                Logger().log('.', end='', flush=True)

        Logger().save_buffer(_dir + 'info.txt')

    print(':) OK')

    return {'best_score': best_score, 'metrics': metrics, 'model': cls, 'list_score': list_score}


def get_scores(_model, file_model_fold, input_train, target_train, input_test, target_test, save=True, **kwargs):
    if save or not os.path.exists(file_model_fold):
        metric = _model.fit(input_train, target_train, **kwargs)
        # Compute metrics
        score_train = metric.append_prediction(input_train, target_train)
        score_test = metric.append_prediction(input_test, target_test, append_last_pred=True)
        if save:
            s_data = Serializable((score_train, score_test, metric, _model.save_params()))
            s_data.save(file_model_fold)
    else:
        # Load sets
        Logger().log('Load file: %s' % file_model_fold)
        s_data = Serializable()
        s_data.load(file_model_fold)
        score_train, score_test, metric, _ = s_data.get_data()

    return score_train, score_test, metric


def cross_validation_score(models, data_input, data_target, _compile=True, test_size=0.1,
                           folds=10, path_db='', **kwargs):
    Logger().reset()
    # Generate data models
    models_data = []
    list_data_training_models = {}
    for _model in models:
        _dir = path_db + _model.get_name() + '/'
        make_dirs(_dir)
        file_model = _dir + 'model.pkl'
        if not os.path.exists(file_model):
            # Extra Scores
            if isinstance(_model, EnsembleModel):
                _model.append_score(score_ensemble_ambiguity, TextTranslation().get_str('Ambiguity'))
            # Compile
            if _compile:
                _model.compile(fast=False)
            # Save model
            _model.save(file_model)
        else:
            # Load model
            Logger().log(TextTranslation().get_str('Load_Model') + ': %s' % file_model)
            _model.load(file_model)

        models_data.append((_model, _dir))
        list_data_training_models[_model.get_name()] = []

    Logger().reset()
    Logger().log_enable()

    for fold in range(folds):
        Logger().log('Fold: %d' % (fold + 1))
        file_sets_fold = path_db + 'sets_fold_%d.pkl' % fold
        if not os.path.exists(file_sets_fold):
            # Generate testing and training sets
            input_train, input_test, target_train, target_test = \
                model_selection.train_test_split(data_input, data_target, test_size=test_size)
            sets_data = Serializable((input_train, input_test, target_train, target_test))
            sets_data.save(file_sets_fold)
        else:
            # Load sets
            Logger().log(TextTranslation().get_str('Load_Sets') + ': %s' % file_sets_fold)
            sets_data = Serializable()
            sets_data.load(file_sets_fold)
            input_train, input_test, target_train, target_test = sets_data.get_data()

        for (_model, _dir) in models_data:
            file_model_fold = _dir + 'data_fold_%d.pkl' % fold
            _model.reset()
            score_train, score_test, metric = get_scores(_model, file_model_fold,
                                                         input_train, target_train, input_test, target_test, **kwargs)

            list_data_training_models[_model.get_name()].append((score_train, score_test, metric))

    return list_data_training_models


def load_model(name_db, name_model):
    file_model = name_db + '/' + name_model + '/model.pkl'
    s_model = Serializable()
    s_model.load(file_model)
    return s_model.get_data()


def show_info_model(_model):
    """ Print Info model
    Parameters
    ----------
    _model : Model
        Model.

    Returns
    -------
    None
    """
    Logger().log('info:\n%s' % _model.get_info())


def show_info_data_base(classes_labels, name_db, desc, col_names):
    Logger().log('Data (%s):\n DESC: %s.\n Features(%d): %s\n Classes(%d): %s' %
                 (name_db, desc, len(col_names), col_names, len(classes_labels), classes_labels))


def show_info_training(input_train, input_test, folds, max_epoch, batch_size):
    Logger().log('Training:\n total data: %d | train: %d, validation: %d ' %
                 (input_train.shape[0] + input_test.shape[0], input_train.shape[0], input_test.shape[0]))
    Logger().log(' folds: %d | Epoch: %d, Batch Size: %d ' %
                 (folds, max_epoch, batch_size))


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
    fig, ax = plt.subplots(figsize=ConfigPlot().get_fig_size(), dpi=ConfigPlot().get_dpi())

    ind = np.arange(len(labels))
    width = 0.25

    # the bars
    rects_1 = plt.bar(ind, train_means, width,
                     color='blue', alpha=0.5)

    rects_2 = plt.bar(ind + width, test_means, width,
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
    plt.legend((rects_1[0], rects_2[0]), ('Train', 'Test'))

    def auto_label(rects):
        """ Generate labels.

        Parameters
        ----------
        rects
        """
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%.2f' % (height * 100),
                    ha='center', va='bottom')

    auto_label(rects_1)
    auto_label(rects_2)

    plt.tight_layout()


# noinspection PyTypeChecker,PyTypeChecker
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
    fig = plt.figure(figsize=ConfigPlot().get_fig_size(), dpi=ConfigPlot().get_dpi())
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
    ax.legend((rects_1[0], rects_2[0]), (TextTranslation().get_str('Train'),
                                         TextTranslation().get_str('Test')))
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


def plot_pdf(ax, x, label, x_min=-1, x_max=1, n_points=1000):
    N = len(x)
    # noinspection PyTypeChecker
    s = float(1.06 * np.std(x) / np.power(N, 0.2))  # Silverman
    kde = KernelDensity(kernel='gaussian', bandwidth=s)
    kde.fit(x[:, np.newaxis])

    x_plot = np.linspace(x_min, x_max, n_points)[:, np.newaxis]
    y = np.exp(kde.score_samples(x_plot))
    ax.plot(x_plot, y / np.sum(y), label=label)


def get_min_score(_scores, _s, train=False):
    best_scores = []
    beta = []
    lamb = []
    i = 0 if train else 1
    for key, value in _scores.items():
        if abs(key[2] - _s) < 0.0001:
            best_scores.append(np.min(value, axis=0)[i])
            beta.append(key[0])
            lamb.append(key[1])
    return beta, lamb, best_scores


def get_best_score(_scores, _s, train=False):
    best_scores = []
    beta = []
    lamb = []
    i = 0 if train else 1
    for key, value in _scores.items():
        if abs(key[2] - _s) < 0.0001:
            best_scores.append(np.max(value, axis=0)[i])
            beta.append(key[0])
            lamb.append(key[1])
    return beta, lamb, best_scores


def get_mean_score(_scores, _s, train=False):
    best_scores = []
    beta = []
    lamb = []
    i = 0 if train else 1
    for key, value in _scores.items():
        if abs(key[2] - _s) < 0.0001:
            d = np.mean(value, axis=0)
            best_scores.append(d[i])
            beta.append(key[0])
            lamb.append(key[1])
    return beta, lamb, best_scores


def get_mean_diversity(_diversity, _s):
    best_diversity = []
    beta = []
    lamb = []
    for key, value in _diversity.items():
        if abs(key[2] - _s) < 0.0001:
            d = np.mean(value)
            best_diversity.append(d)
            beta.append(key[0])
            lamb.append(key[1])
    return beta, lamb, best_diversity


def plot_graph(fig, ax, X, Y, Z, xlabel, ylabel, zlabel):
    x = np.linspace(min(X), max(X), 25)
    y = np.linspace(min(Y), max(Y), 25)

    # noinspection PyTypeChecker,PyTypeChecker
    zz = griddata(X, Y, Z, x, y, interp='linear')
    xx, yy = np.meshgrid(x, y)

    x = xx.flatten()
    y = yy.flatten()
    z = zz.flatten()
    z_max = np.max(z)
    z_min = np.min(z)

    surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
    surf.set_clim([z_min, z_max])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_zlim(z_min, z_max)

    fig.colorbar(surf, ticks=np.linspace(z_min, z_max, 5))


# noinspection PyUnusedLocal
def plot_graph2(ax, X, Y, Z, xlabel, ylabel, zlabel, s_title, vmin=0, vmax=1):
    x = np.linspace(min(X), max(X), 200)
    y = np.linspace(min(Y), max(Y), 200)

    xx, yy = np.meshgrid(x, y)
    # noinspection PyTypeChecker,PyTypeChecker
    zz = griddata2((X, Y), Z, (xx, yy), method='cubic', rescale=True)

    p = ax.pcolor(xx, yy, zz, cmap=cm.jet, vmin=vmin, vmax=vmax)
    ax.title.set_text(s_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return p
