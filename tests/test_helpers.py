import unittest
import numpy as np


class _EnsureInitializeInputsAre:
    def __init__(self, X, y, testobj):
        self.X = X
        self.y = y
        self.testobj = testobj

    def initializer(self, X, y):

        test = self.testobj
        test.assertTrue(np.allclose(self.X, X))

        if y is None:
            test.assertTrue(self.y is None)
        else:
            test.assertTrue(np.allclose(self.y, y))


class TestHelpers(unittest.TestCase):
    def test_force_matrix(self):
        from pyonline.helpers import force_matrix

        self.assertEquals(
            (2, 3),
            force_matrix([
                [1, 2, 3],
                [4, 5, 6],
            ]).shape
        )

        x = np.array([1, 2])

        self.assertEquals(
            (1, 2),
            force_matrix(x, col=False).shape
        )

        self.assertEquals(
            (2, 1),
            force_matrix(x, col=True).shape
        )

        self.assertEquals(
            (1, 1),
            force_matrix(1, col=True).shape
        )

    def test_initializer(self):
        from pyonline.helpers import Initializer
        X = np.ones((10, 3))
        y1 = 2. * np.ones(10)
        y2 = 2. * np.ones((10, 2))

        Xp, yp = Initializer(
            _EnsureInitializeInputsAre(
                X=X,
                y=y1.reshape((10, 1)),
                testobj=self
            ),
            'initializer',
            min_samples=10,
        ).fit(X, y1).predict()

        self.assertEqual((Xp, yp), (None, None))

        class FailTestIfCalled:
            def fail_test(self, X, y):
                self.assertTrue(False)

        Xp, yp = Initializer(
            FailTestIfCalled(),
            'fail_test',
            min_samples=11,
        ).fit(X, y1).predict()
        self.assertEqual((Xp, yp), (None, None))

        Xp, yp = Initializer(
            _EnsureInitializeInputsAre(
                X=X[:-1],
                y=y2[:-1],
                testobj=self
            ),
            'initializer',
            min_samples=9
        ).fit(X, y2).predict()

        self.assertTrue(np.allclose(X[-1:], Xp))
        self.assertTrue(np.allclose(yp[-1:], y2[-1:]))

        Xp, yp = Initializer(
            _EnsureInitializeInputsAre(
                X=X[:-1],
                y=y2[:-1],
                testobj=self
            ),
            'initializer',
            min_samples=9
        ).fit(X[:5], y2[:5]).fit(X[5:], y2[5:]).predict()

        self.assertTrue(np.allclose(X[-1:], Xp))
        self.assertTrue(np.allclose(y2[-1:], yp))

        Xp, yp = Initializer(
            _EnsureInitializeInputsAre(
                X=X,
                y=None,
                testobj=self
            ),
            'initializer',
            min_samples=10,
        ).fit(X, None).predict()

        self.assertEqual((Xp, yp), (None, None))


if __name__ == '__main__':
    unittest.main()
