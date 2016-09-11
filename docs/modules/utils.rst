:mod:`libml.utils`

Utils Functions
===============

Cost Functions
--------------

.. automodule:: libml.utils.cost_functions

.. autofunction:: mse
.. autofunction:: mcc
.. autofunction:: mee
.. autofunction:: neg_log_likelihood

Cost Function for Ensemble Models
---------------------------------

.. autofunction:: neg_corr

Regularizer Functions
---------------------

.. automodule:: libml.utils.regularizer_functions

.. autofunction:: L1
.. autofunction:: L2

Update Functions
----------------

.. automodule:: libml.utils.update_functions

.. autofunction:: sgd
.. autofunction:: sgd_momentum
.. autofunction:: adadelta
.. autofunction:: adagrad

Metrics
=======

.. currentmodule:: libml.utils.metrics

Base Class Metrics
------------------

.. automodule:: libml.utils.metrics.basemetrics

.. autoclass:: BaseMetrics
   :members:

.. autoclass:: EnsembleMetrics
   :members:

Classifier Metrics
------------------

.. automodule:: libml.utils.metrics.classifiermetrics

.. autoclass:: ClassifierMetrics
   :members:

.. autoclass:: EnsembleClassifierMetrics
   :members:

Regression Metrics
------------------

.. automodule:: libml.utils.metrics.regressionmetrics

.. autoclass:: RegressionMetrics
   :members:

.. autoclass:: EnsembleMetrics
   :members:
