:mod:`deepensemble.utils`

Utils Functions
===============

Cost Functions
--------------

.. automodule:: deepensemble.utils.cost_functions

.. autofunction:: mse
.. autofunction:: mcc
.. autofunction:: mee
.. autofunction:: neg_log_likelihood

Cost Function for Ensemble Models
---------------------------------

.. autofunction:: neg_corr

Regularizer Functions
---------------------

.. automodule:: deepensemble.utils.regularizer_functions

.. autofunction:: L1
.. autofunction:: L2

Update Functions
----------------

.. automodule:: deepensemble.utils.update_functions

.. autofunction:: sgd
.. autofunction:: sgd_momentum
.. autofunction:: adadelta
.. autofunction:: adagrad

Metrics
=======

.. currentmodule:: deepensemble.utils.metrics

Base Class Metrics
------------------

.. automodule:: deepensemble.utils.metrics.basemetrics

.. autoclass:: BaseMetrics
   :members:

.. autoclass:: EnsembleMetrics
   :members:

Classifier Metrics
------------------

.. automodule:: deepensemble.utils.metrics.classifiermetrics

.. autoclass:: ClassifierMetrics
   :members:

.. autoclass:: EnsembleClassifierMetrics
   :members:

Regression Metrics
------------------

.. automodule:: deepensemble.utils.metrics.regressionmetrics

.. autoclass:: RegressionMetrics
   :members:

.. autoclass:: EnsembleMetrics
   :members:

Diversity
=========

.. automodule:: deepensemble.utils.metrics.diversitymetrics

Utils
-----

.. autofunction:: oracle
.. autofunction:: contingency_table

Pairwise metrics
----------------

.. autofunction:: correlation_coefficient
.. autofunction:: disagreement_measure
.. autofunction:: double_fault_measure
.. autofunction:: kappa_statistic
.. autofunction:: q_statistic