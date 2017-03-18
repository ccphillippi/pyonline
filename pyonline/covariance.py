from sklearn.base import BaseEstimator
import numpy as np


class Covariance(BaseEstimator):
    def __init__(self, span=None):
        self.span = span

        if self.span is None:
            self.retain = 1.
        else:
            self.retain = 1. - 2. / (1. + self.span)

        self.initialized = False

    def partial_fit(self, X, y=None):

        if not self.initialized:
            mean = X[0]
            new_wt = 1.
            nobs = int((mean == mean).all())
            cov = 0.
            sum_wt = 1.
            sum_wt2 = 1.
            old_wt = 1.
            self.covariance_ = np.nan * np.eye(len(mean))
        else:
            mean = self.mean
            new_wt = self.new_wt
            nobs = self.nobs
            cov = self._cov
            sum_wt = self.sum_wt
            sum_wt2 = self.sum_wt2
            old_wt = self.old_wt

        retain = self.retain

        for x in X[1:]:
            is_obs = (x == x).all()
            nobs += is_obs
            if (mean == mean).all():
                if is_obs:
                    sum_wt *= retain
                    sum_wt2 *= retain * retain
                    old_wt *= retain

                    old_mean = mean

                    # avoid numerical errors on constant series
                    if (mean != x).all():
                        mean = (
                            (old_wt * old_mean) +
                            (new_wt * x)
                        ) / (old_wt + new_wt)

                    dmean = old_mean - mean
                    x_m_mu = x - mean
                    cov = (
                        (old_wt * (cov + np.outer(dmean, dmean))) +
                        (new_wt * np.outer(x_m_mu, x_m_mu))
                    ) / (old_wt + new_wt)

                    sum_wt += new_wt
                    sum_wt2 += new_wt * new_wt
                    old_wt += new_wt
            elif is_obs:
                mean = x

            numerator = sum_wt * sum_wt
            denominator = numerator - sum_wt2
            if denominator > 0.:
                self.covariance_ = (numerator / denominator) * cov
            else:
                self.covariance_ = np.nan * cov

        self.mean = mean
        self.new_wt = new_wt
        self.nobs = nobs
        self._cov = cov
        self.sum_wt = sum_wt
        self.sum_wt2 = sum_wt2
        self.old_wt = old_wt
        self.initialized = True

        return self

    def fit(self, X, y=None):
        return self.partial_fit(X, y)
