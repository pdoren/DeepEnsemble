from deepensemble.utils import load_data
from deepensemble.utils.utils_classifiers import get_index_label_classes, translate_target
from deepensemble.utils.utils_functions import ActivationFunctions, ITLFunctions
from test_models.test_classifiers.test_classifiers import test_classifiers

import pandas as pd
import numpy as np
from theano import shared

#############################################################################################################
# Load Data
#############################################################################################################
data_input, data_target, classes_labels, name_db, desc, col_names = load_data('diabetes_scale',
                                                                              data_home='../../data')
y = get_index_label_classes(translate_target(data_target, classes_labels))
s = ITLFunctions.silverman(shared(np.array(y)), len(y), len(classes_labels)).eval()

#############################################################################################################
# Testing
#############################################################################################################

# 10-Cross Validation (sets: 90% train 10% test)
scores = test_classifiers(name_db, data_input, data_target, classes_labels,
                         only_cip=False, n_ensemble_models=3,
                         lamb_ncl=0.6, beta_cip=0.2, lamb_cip=0.05, s=None, bias_layer=False, dist='CS',
                          kernel=ITLFunctions.kernel_gauss,
                         fn_activation1=ActivationFunctions.tanh, fn_activation2=ActivationFunctions.sigmoid,
                         folds=10, test_size=0.1, lr=0.01, max_epoch=500, batch_size=40)
r_score = {}
for s in scores:
    d = scores[s]
    _mean = np.mean(d, axis=0)
    _std = np.std(d, axis=0)
    max_score = np.max(d, axis=0)
    s1 = ['%.2f +-%.2f' % (100 * _mean[0], 100 * _std[0])]
    s2 = ['%.2f +-%.2f' % (100 * _mean[1], 100 * _std[1])]
    s3 = ['%.2f' % (100 * max_score[1])]
    r_score[s] = [s1, s2, s3]
df = pd.DataFrame(r_score)

print(df)