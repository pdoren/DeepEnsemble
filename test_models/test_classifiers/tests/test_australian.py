from deepensemble.utils import load_data, Serializable
from deepensemble.utils.utils_classifiers import get_index_label_classes, translate_target
from deepensemble.utils.utils_functions import ActivationFunctions, ITLFunctions
from deepensemble.utils.cost_functions import mse
from test_models.test_classifiers.test_classifiers import test_classifiers

import os
import pandas as pd
import numpy as np
from theano import shared

#############################################################################################################
# Load Data
#############################################################################################################
data_input, data_target, classes_labels, name_db, desc, col_names = load_data('australian_scale',
                                                                              data_home='../../data', normalize=False)
y = get_index_label_classes(translate_target(data_target, classes_labels))
s = ITLFunctions.silverman(shared(np.array(y)), len(y), len(classes_labels)).eval()

#############################################################################################################
# Testing
#############################################################################################################

file_scores = name_db + '/score.pkl'

if not os.path.exists(file_scores):
    # 10-Cross Validation (sets: 90% train 10% test)
    scores = test_classifiers(name_db, data_input, data_target, classes_labels,
                              factor_number_neurons=1.0,
                              is_binary=False, early_stop=False,
                              only_cip=False, n_ensemble_models=3,
                              lamb_ncl=1.0,
                              beta_cip=0.8, lamb_cip=0.01, s=s, dist='CS',
                              kernel=ITLFunctions.kernel_gauss,
                              fn_activation1=ActivationFunctions.sigmoid,
                              fn_activation2=ActivationFunctions.sigmoid,
                              folds=10, lr=0.5, max_epoch=500, batch_size=40)
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
        metrics = [t1.get_fails() for _, _, t1 in d_score]
        print(s)
        print(metrics)
    _mean = np.mean(d, axis=0)
    _std = np.std(d, axis=0)
    max_score = np.max(d, axis=0)
    min_score = np.min(d, axis=0)
    s1 = ['%.2f +-%.2f' % (100 * _mean[0], 100 * _std[0])]
    s2 = ['%.2f +-%.2f' % (100 * _mean[1], 100 * _std[1])]
    s3 = ['%.2f / %.2f' % (100 * max_score[1], 100 * min_score[1])]
    r_score[s] = [s1, s2, s3]


df = pd.DataFrame(r_score)

print(df)
