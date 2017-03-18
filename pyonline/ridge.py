import numpy as np
from sklearn.base import BaseEstimator


class RidgeRegression(BaseEstimator):
    def __init__(self, span=None, precision=0.):
        self.span = span
        self.precision = float(precision)

        if span is None:
            self.retain = 1.
        else:
            self.retain = 1. - 2. / (1. + float(self.span))

    def _update_state(self, X, y):
        if not hasattr(self, 'XtX'):
            self.XtX = 0.
            self.XtY = 0.
            self.n = 0.

        for x, y in zip(X, y):
            self.XtX = (
                self.retain * self.XtX +
                np.outer(x, x)
            )
            self.XtY = (
                self.retain * self.XtY +
                np.outer(x, y)
            )
            self.n = 1. + self.retain * self.n

        return self

    def partial_fit(self, X, y):
        self._update_state(X, y)

        nI = self.n * np.eye(len(self.XtX))
        self.coef_ = np.linalg.pinv(
            self.XtX + self.precision * nI
        ).dot(self.XtY)

        return self

    def fit(self, X, y):
        return self.partial_fit(X, y)

    def predict(self, X):
        return X.dot(self.coef_)

    def score(self, X, y):
        predicted = self.predict(X)
        residual = y - predicted
        tss = ((y - y.mean()) ** 2).sum()
        rss = (residual ** 2).sum()

        return 1. - rss / tss
