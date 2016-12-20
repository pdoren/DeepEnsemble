import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from theano import shared, config

from deepensemble.utils import Serializable, jacobs
from deepensemble.utils.utils_test import load_model
from deepensemble.metrics import EnsembleRegressionMetrics
from deepensemble.utils.utils_functions import ActivationFunctions, ITLFunctions
from test_models.test_regression.test_regression import test_regression

#############################################################################################################
# Load Data
#############################################################################################################
X, y = jacobs(sample_len=1000, seed=42)
y = (y + 1.0) / 2.0
y = np.array(y[:, np.newaxis], dtype=config.floatX)
X = np.array(X, dtype=config.floatX)

s = ITLFunctions.silverman(shared(y), y.shape[0], y.shape[1]).eval()

name_db = 'Jacobs'

#############################################################################################################
# Testing
#############################################################################################################

file_scores = name_db + '/score.pkl'

if not os.path.exists(file_scores):
    # 10-Cross Validation (sets: 90% train 10% test)
    scores = test_regression(name_db, X, y,
                             factor_number_neurons=1.0,
                             early_stop=False,
                             n_ensemble_models=4,
                             lamb_ncl=0.8,
                             beta_cip=0.3, lamb_cip=0.1, s=s, dist='CS',
                             is_relevancy=True, pre_training=True, bias_layer=False,
                             kernel=ITLFunctions.kernel_gauss,
                             fn_activation1=ActivationFunctions.sigmoid,
                             fn_activation2=ActivationFunctions.sigmoid,
                             folds=10, lr_mse=0.01, lr_klg=0.01, max_epoch=300, batch_size=40)
    scores_data = Serializable(scores)
    scores_data.save(file_scores)
else:
    scores_data = Serializable()
    scores_data.load(file_scores)
    scores = scores_data.get_data()

r_score = {}
d_diversity = {}
for s in scores:
    d_score = scores[s]
    d = [(t1, t2) for t1, t2, _ in d_score]
    if "Ensamble" in s:
        _model = load_model(name_db, s)
        metrics = EnsembleRegressionMetrics(_model)
        for _, _, metric in d_score:
            metrics.append_metric(metric)
        metrics.plot_cost(title='Costo %s' % s ,max_epoch=300)
        metrics.plot_scores(title='Desempeño %s' % s, max_epoch=300)
    _mean = np.mean(d, axis=0)
    _std = np.std(d, axis=0)
    max_score = np.max(d, axis=0)
    min_score = np.min(d, axis=0)
    s1 = ['%.4f +-%.4f' % (_mean[0], _std[0])]
    s2 = ['%.4f +-%.4f' % (_mean[1], _std[1])]
    s3 = ['%.4f / %.4f' % (max_score[1], min_score[1])]
    r_score[s] = [s1, s2, s3]

plt.show()
df = pd.DataFrame(r_score)

print(df)
