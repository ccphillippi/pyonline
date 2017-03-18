import unittest


class TestLearn(unittest.TestCase):
    def test_ewm_covariance_vs_pandas(self):
        import numpy as np
        import pandas as pd
        from pyonline.covariance import Covariance

        span = 3
        random_state = np.random.RandomState(3)
        X = random_state.normal(.1, .324, (100, 10))

        candidate = Covariance(span=span).fit(X).covariance_
        true = pd.DataFrame(X).ewm(span=span).cov(adjust=True).iloc[-1].values

        passed_test = np.allclose(true, candidate)

        if not passed_test:
            print('Failed test:\nCandidate:\n%s\nTrue:\n%s\nDiffs\n%s\n' % (
                candidate, true, true - candidate
            ))

        self.assertTrue(passed_test)


if __name__ == '__main__':
    unittest.main()
