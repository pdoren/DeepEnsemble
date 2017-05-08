import os
import matplotlib
matplotlib.use('Qt4Agg')  # debug

import numpy as np
from theano import config

from deepensemble.utils import Serializable, jacobs
from deepensemble.utils.utils_functions import ActivationFunctions, ITLFunctions
from test_models.test_regression.test_regression import test_regression, show_data_regression

#############################################################################################################
# Load Data
#############################################################################################################
X, y = jacobs(sample_len=1000, seed=42)
y = np.array(y[:, np.newaxis], dtype=config.floatX)
X = np.array(X, dtype=config.floatX)

s = ITLFunctions.silverman(y).eval()

name_db = 'Jacobs'

max_epoch = 300

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
                             beta_cip=1.0, lamb_cip=1.0, s=s, dist='CS',
                             is_cip_full=False, bias_layer=False, mse_first_epoch=True,
                             annealing_enable=True,
                             fn_activation1=ActivationFunctions.sigmoid,
                             fn_activation2=ActivationFunctions.sigmoid,
                             folds=10, lr=0.1, max_epoch=max_epoch, batch_size=32)
    scores_data = Serializable(scores)
    scores_data.save(file_scores)
else:
    scores_data = Serializable()
    scores_data.load(file_scores)
    scores = scores_data.get_data()

show_data_regression(name_db, scores=scores, max_epoch=max_epoch)
