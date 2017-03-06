from unittest import TestCase

__author__ = 'pdoren'
__project__ = 'DeepEnsemble'


class TestITLFunctions(TestCase):
    # noinspection PyStringFormat
    def test_cross_information_potential(self):
        from deepensemble.utils.utils_functions import ITLFunctions
        import numpy as np

        N = 4
        n_classes = 2
        y1 = np.squeeze(np.random.binomial(1, 0.5, (N, n_classes)))
        y2 = y1.copy()
        m = int(0.8 * N)
        y2[:m] = 1 - y2[:m]

        s = 0.2

        if n_classes > 1:
            Y = [y1, y2]
        else:
            Y = [y1[:, np.newaxis], y2[:, np.newaxis]]

        DY = []
        for y in Y:
            dy = np.tile(y, (len(y), 1, 1))
            dy = dy - np.transpose(dy, axes=(1, 0, 2))
            DY.append(dy)

        DYK = []
        for dy in DY:
            DYK.append(ITLFunctions.kernel_gauss(dy, s).eval())

        p1 = np.prod(np.array([dyk for dyk in DYK]), axis=0)
        self.assertTrue(p1.size == N ** 2, 'Problem V_J2 (%g != %g)' % (p1.size, N ** 2))
        V_J2 = np.mean(p1)

        V_k_i = []

        for dyk in DYK:
            V_k_i.append(np.mean(dyk, axis=0))

        V_k = [np.mean(V_i) for V_i in V_k_i]

        p2 = np.prod(V_k_i, axis=0)
        self.assertTrue(p2.size == N, 'Problem V_nc2 (%g != %g)' % (p2.size, N))
        V_nc2 = np.mean(p2)

        V_M2 = np.prod(V_k)

        V_nc1, V_J1, V_M1 = ITLFunctions.get_cip(Y, s=s)

        self.assertTrue(abs(V_nc1.eval() - V_nc2) < 0.00001, 'Problem V_nc (%g != %g)' % (V_nc1.eval(), V_nc2))
        self.assertTrue(abs(V_J1.eval() - V_J2) < 0.00001, 'Problem V_J (%g != %g)' % (V_J1.eval(), V_J2))
        self.assertTrue(abs(V_M1.eval() - V_M2) < 0.00001, 'Problem V_M (%g != %g)' % (V_M1.eval(), V_M2))

        V_c2 = V_nc2 ** 2 / (V_J2 * V_M2)
        V_c1 = ITLFunctions.cross_information_potential(Y, s=s)
        self.assertTrue(abs(V_c1.eval() - V_c2) < 0.00001, 'Problem V_c (%g != %g)' % (V_c1.eval(), V_c2))

    def test_mutual_information_cs(self):
        from deepensemble.utils.utils_functions import ITLFunctions
        import numpy as np
        from sklearn.metrics import mutual_info_score

        N = 4
        y1 = np.random.binomial(1, 0.5, N)
        y2 = y1.copy()
        m = int(0.5 * N)
        y2[:m] = 1 - y2[:m]

        Y = [y1[:, np.newaxis], y2[:, np.newaxis]]

        s = 1.06 * np.std(y1) * (len(y1)) ** (-0.2)

        Ics = ITLFunctions.mutual_information_cs(Y, s=max(s, 0.00001))
        I = mutual_info_score(y1, y2)

        self.assertTrue(abs(Ics.eval() - I) < 0.001, 'Problem Ics and I')

    def test_mutual_information_ed(self):
        from deepensemble.utils.utils_functions import ITLFunctions
        import numpy as np
        from sklearn.metrics import mutual_info_score

        y1 = np.array([1, 1, 0, 0, 1, 1, 0])
        y2 = np.array([0, 0, 1, 1, 0, 0, 1])

        Y = [y1[:, np.newaxis], y2[:, np.newaxis]]

        s = 1.06 * np.std(y1) * (len(y1)) ** (-0.2)

        Ied = ITLFunctions.mutual_information_ed(Y, s=max(s, 0.00001))
        I = mutual_info_score(y1, y2)

        self.assertFalse(abs(Ied.eval() - I) < 0.01, 'Problem Ied and I')
