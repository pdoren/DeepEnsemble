import numpy as np

__all__ = ['oracle', 'contingency_table',
           # pairwise
           'disagreement_measure', 'double_fault_measure', 'q_statistic',
           'correlation_coefficient', 'kappa_statistic',
           # non pairwise
           'kohavi_wolpert_variance', 'interrater_agreement',
           'entropy_cc', 'entropy_sk',
           'coincident_failure', 'difficulty', 'generalized_diversity']

"""
Different metrics for compute diversity
"""

eps = 0.0000001

#
# Utils
#

def oracle(y, c):
    """ Compare between two arrays that represent the target and the output of a classifier.

    .. note:: The input arrays must be same shape and type (numpy.array).

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c : numpy.array
        Output model for comparing with target.

    Returns
    -------
    numpy.array
    return an array with 1 and 0 by each elements of the input arrays.

    References
    ----------
    .. [1] Ludmila I. Kuncheva (2004), pp 298:
           Combining Pattern Classifiers Methods and Algorithms
           A Wiley-Interscience publication, ISBN 0-471-21078-1 (cloth).
    """
    if y.shape != c.shape:
        raise ValueError("Incorrect arrays size")
    return (y == c).astype(int)


def contingency_table(y, c1, c2):
    """ Compute de contingency table.

    .. note:: The input arrays must be same shape and type (numpy.array).

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c1 : numpy.array
        Output of the first classifier.

    c2 : numpy.array
        Output of the second classifier.

    Returns
    -------
    tuple
        Returns a tuple with the elements of contingency table between two classifiers.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 105:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    o1 = oracle(y, c1)
    o2 = oracle(y, c2)
    a, b, c, d = 0, 0, 0, 0
    for i in range(len(o1)):
        if o1[i] == 1 and o2[i] == 1:
            a += 1
        elif o1[i] == 1 and o2[i] != 1:
            b += 1
        elif o1[i] != 1 and o2[i] == 1:
            c += 1
        else:
            d += 1
    return a, b, c, d


#
# Pairwise measures
#

def disagreement_measure(y, c1, c2):
    """ Measure is equal to the probability that the two classifiers will disagree on their decisions.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c1 : numpy.array
        Output of the first classifier.

    c2 : numpy.array
        Output of the second classifier.

    Returns
    -------
    float
        Return the disagreement measure between the classifiers 'c1' and 'c2', takes value in the range of [0, 1].

    References
    ----------
    .. [1] Skalak, D. (1996):
           The sources of increased accuracy for two proposed boosting algorithms. In Proc. American
           Association for Artificial Intelligence, AAAI-96, Integrating Multiple Learned Models Workshop.
    .. [2] Ludmila I. Kuncheva (2004), pp 300:
           Combining Pattern Classifiers Methods and Algorithms
           A Wiley-Interscience publication, ISBN 0-471-21078-1 (cloth).
    """
    a, b, c, d = contingency_table(y, c1, c2)
    return (b + d) / (a + b + c + d)


def double_fault_measure(y, c1, c2):
    """ Measure is equal to the probability that both classifiers being wrong.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c1 : numpy.array
        Output of the first classifier.

    c2 : numpy.array
        Output of the second classifier.

    Returns
    -------
    float
    Return the double fault measure between the classifiers 'c1' and 'c2', takes value in the range of [0, 1].

    References
    ----------
    .. [1] Ludmila I. Kuncheva (2004), pp 301:
           Combining Pattern Classifiers Methods and Algorithms
           A Wiley-Interscience publication, ISBN 0-471-21078-1 (cloth).
    """
    a, b, c, d = contingency_table(y, c1, c2)
    return d / (a + b + c + d)


def q_statistic(y, c1, c2):
    """ Q-Statistic.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c1 : numpy.array
        Output of the first classifier.

    c2 : numpy.array
        Output of the second classifier.

    Returns
    -------
    float
        Return the Q-Statistic measure between the classifiers 'c1' and 'c2'.
        Q-Statistic takes value in the range of [-1, 1]:

         - is zero if 'c1' and 'c2' are independent.
         - is positive if 'c1' and 'c2' make similar predictions.
         - is negative if 'c1' and 'c2' make different predictions.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 105:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    a, b, c, d = contingency_table(y, c1, c2)
    return (a * d - b * c) / (a * d + b * c)


def correlation_coefficient(y, c1, c2):
    """ Correlation between two binary vectors in this case between the output of the classifiers.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c1 : numpy.array
        Output of the first classifier.

    c2 : numpy.array
        Output of the second classifier.

    Returns
    -------
    float
        Return the correlation coefficient between the output of the classifiers 'c1' and 'c2'.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 105:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    a, b, c, d = contingency_table(y, c1, c2)
    return (a * d - b * c) / np.sqrt((a + b) * (a + c) * (c + d) * (b + d))


def kappa_statistic(y, c1, c2):
    """ Kappa-Statistic.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    c1 : numpy.array
        Output of the first classifier.

    c2 : numpy.array
        Output of the second classifier.

    Returns
    -------
    float
        Return the Kappa-Statistic (kp) measure between the classifiers 'c1' and 'c2'.

         - kp = 1 if the two classifiers totally agree.
         - kp = 0 if the two classifiers agree by chance.
         - kp < 0 is a rare case where the agreement is even less than what is expected by chance.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 105:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    a, b, c, d = contingency_table(y, c1, c2)
    m = (a * d + b * c)
    theta1 = (a + d) / m
    theta2 = ((a + b) * (a + c) + (c + d) * (b + d)) / (m * m)
    return (theta1 - theta2) / (1 - theta2)

#
# Non-PairwiseMeasures
#

def kohavi_wolpert_variance(y, classifiers):
    """ Kohavi-Wolpert Variance.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    classifiers : numpy.array
        Output of the classifiers.

    Returns
    -------
    float
        Return the Kohavi-Wolpert Variance.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 107:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    T = classifiers.shape[0]
    I = [oracle(y, c) for c in classifiers]  # indicator function
    d = np.sum(I, axis=0)  # is the number of individual classifiers that classify correctly.
    return np.mean(d * (T - d)) / (T**2)

def interrater_agreement(y, classifiers):
    """ Interrater agreement.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    classifiers : numpy.array
        Output of the classifiers.

    Returns
    -------
    float
        Return the Interrater agreement.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 107:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    T = classifiers.shape[0]
    I = [oracle(y, c) for c in classifiers]  # indicator function
    d = np.sum(I, axis=0)  # is the number of individual classifiers that classify correctly.
    p = np.mean(I)
    return 1 - ((np.mean(d * (T - d))) / (T * (T - 1) * p * (1 - p) + eps))

def entropy_cc(y, classifiers):
    """ Entropy Cuningham and Carney [2000].

    Parameters
    ----------
    y : numpy.array
        Target sample.

    classifiers : numpy.array
        Output of the classifiers.

    Returns
    -------
    float
        Return the Entropy Cuningham and Carney.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 107:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    I = [oracle(y, c) for c in classifiers]  # indicator function
    p = np.mean(I, axis=0)
    return np.mean(-p * np.log(p))


def entropy_sk(y, classifiers):
    """ Entropy Shipp and Kuncheva  [2002].

    Parameters
    ----------
    y : numpy.array
        Target sample.

    classifiers : numpy.array
        Output of the classifiers.

    Returns
    -------
    float
        Return the Entropy Shipp and Kuncheva.

    References
    ----------
    .. [1] Kuncheva, Ludmila I. (2004), pp 301:
           Combining pattern classifiers: methods and algorithms.
           John Wiley & Sons, 2004.
    """
    T = classifiers.shape[0]
    I = [oracle(y, c) for c in classifiers]  # indicator function
    d = np.sum(I, axis=0)  # is the number of individual classifiers that classify correctly.
    _min = np.min((d, T - d), axis=0)
    return 2 * np.mean(_min) / (T - 1)


def prY(y, classifiers):
    T = classifiers.shape[0]
    I = [oracle(y, c) for c in classifiers]  # indicator function
    pi = []
    fail = T - np.sum(I, axis=0)
    for i in range(T + 1):
        ff = np.sum(fail == i)
        pi.append(np.mean(ff))
    return pi / np.sum(pi)


def difficulty(y, classifiers):
    """ Difficulty.

    Parameters
    ----------
    y : numpy.array
        Target sample.

    classifiers : numpy.array
        Output of the classifiers.

    Returns
    -------
    float
        Return the Generalized Diversity.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 108:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    T = classifiers.shape[0]
    I = [oracle(y, c) for c in classifiers]  # indicator function
    X = np.sum(I, axis=0) / T
    return np.var(X)


def generalized_diversity(y, classifiers):
    """ Generalized Diversity .

    Parameters
    ----------
    y : numpy.array
        Target sample.

    classifiers : numpy.array
        Output of the classifiers.

    Returns
    -------
    float
        Return the Generalized Diversity.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 108:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    T = classifiers.shape[0]
    pi = prY(y, classifiers)
    p1 = 0.0
    p2 = 0.0
    for i in range(1, T + 1):
        p1 += i * pi[i] / T
        p2 += i * (i - 1) * pi[i] / (T * (T - 1))

    return 1 - p2 / (p1 + eps)


def coincident_failure(y, classifiers):
    """ Coincident Failure (Partridge and Krzanowski, 1997).

    Parameters
    ----------
    y : numpy.array
        Target sample.

    classifiers : numpy.array
        Output of the classifiers.

    Returns
    -------
    float
        Return the Coincident Failure.

    References
    ----------
    .. [1] Zhi-Hua Zhou. (2012), pp 109:
           Ensemble Methods Foundations and Algorithms
           Chapman & Hall/CRC Machine Learning & Pattern Recognition Series.
    """
    T = classifiers.shape[0]
    pi = prY(y, classifiers)
    p0 = pi[0]

    if p0 == 1:
        return 0.0
    else:
        p = 0.0
        for i in range(1, T + 1):
            p += (T - i) * pi[i] / (T - 1)

        return p / (1 - p0)


def test_non_pairwise():
    """ Test diversity metrics

    Returns
    -------
    None

    References
    ----------
    .. [1] Kuncheva, Ludmila I. (2004), pp 306--307:
           Combining pattern classifiers: methods and algorithms.
           John Wiley & Sons, 2004.
    """

    ys = np.array( [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    Cs = np.array([[1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    metrics = [kohavi_wolpert_variance(ys, Cs),
               interrater_agreement(ys, Cs),
               entropy_sk(ys, Cs),
               difficulty(ys, Cs),
               generalized_diversity(ys, Cs),
               coincident_failure(ys, Cs)]
    metrics = np.around(metrics, decimals=2)
    assert (metrics == [0.16, 0.03, 0.70, 0.08, 0.58, 0.64]).all()
    print(metrics)
