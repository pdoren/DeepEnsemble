from unittest import TestCase

__author__ = 'pdoren'
__project__ = 'DeepEnsemble'


class TestITLFunctions(TestCase):
    def test_cross_information_potential(self):
        from deepensemble.utils.utils_functions import ITLFunctions
        import numpy as np
        import theano.tensor as T
        from sklearn.metrics import mutual_info_score

        y1 = [1, 0, 0]
        y2 = [1, 0, 0]

        s = 1.0

        Y = [y1[:, np.newaxis], y2[:, np.newaxis]]
        V_c1 = ITLFunctions.cross_information_potential(Y, kernel=ITLFunctions.kernel_gauss, s=s)

        DY = []
        for y in Y:
            dy = np.tile(y, (len(y), 1, 1))
            dy = dy - np.transpose(dy, axes=(1, 0, 2))
            DY.append(dy)

        DYK = [ITLFunctions.kernel_gauss(dy, s) for dy in DY]

        V_J = np.mean(np.prod(DYK))

        V_k_i = [np.mean(dyk, axis=-1) for dyk in DYK]

        V_k = [np.mean(V_i) for V_i in V_k_i]

        V_nc = np.mean(np.prod(V_k_i))

        V_M = np.prod(V_k)

        V_nc, V_J, V_M
        V_c2 = V_nc**2/(V_J*V_M)
        self.assertEquals(V_c1, V_c2, 'Problem')


    def test_mutual_information_cs(self):
        from deepensemble.utils.utils_functions import ITLFunctions
        import numpy as np
        import theano.tensor as T
        from sklearn.metrics import mutual_info_score

        N = 4
        y1 = np.random.binomial(1, 0.5, N)
        y2 = y1.copy()
        m = int(0.5 * N)
        y2[:m] = 1 - y2[:m]

        Y = [y1[:, np.newaxis], y2[:, np.newaxis]]

        s = 1.06 * np.std(y1) * (len(y1)) ** (-0.2)

        Ics = ITLFunctions.mutual_information_cs(Y, kernel=ITLFunctions.kernel_gauss, s=max(s, 0.00001))
        I = mutual_info_score(y1, y2)

        self.assertFalse(abs(Ics.eval() - I) < 0.01, 'Problem Ics and I')

    def test_mutual_information_ed(self):
        from deepensemble.utils.utils_functions import ITLFunctions
        import numpy as np
        from sklearn.metrics import mutual_info_score

        y1 = np.array([1, 1, 0, 0, 1, 1, 0])
        y2 = np.array([0, 0, 1, 1, 0, 0, 1])

        Y = [y1[:, np.newaxis], y2[:, np.newaxis]]

        s = 1.06 * np.std(y1) * (len(y1)) ** (-0.2)

        Ied = ITLFunctions.mutual_information_ed(Y, kernel=ITLFunctions.kernel_gauss, s=max(s, 0.00001))
        I = mutual_info_score(y1, y2)

        self.assertFalse(abs(Ied.eval() - I) < 0.01, 'Problem Ied and I')
