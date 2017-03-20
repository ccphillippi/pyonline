import numpy as np

from scipy.linalg import block_diag
from sklearn.base import BaseEstimator

from .helpers import Initializer, force_matrix
from .covariance import DiagonalCovariance


class DynamicRidge(BaseEstimator):
    def __init__(self, span, coef_sd, min_samples=30,
                 covariance_model=None):
        self.span = float(span)
        self.coef_sd = coef_sd
        self.coef_mean = 0.
        self.min_samples = min_samples
        if covariance_model is None:
            self.covariance_model = DiagonalCovariance()
        else:
            self.covariance_model = covariance_model

        self.decay = 1. - 2. / (self.span + 1.)

        self.initializer = Initializer(
            base_obj=self,
            initializer='_initialize',
            min_samples=min_samples
        )

    def _initialize(self, X, y):
        _, n_x = X.shape
        _, n_y = y.shape

        n_coefs = n_x * n_y

        if self.coef_mean is None:
            coef_mean = np.zeros(n_coefs)
        else:
            coef_mean = np.ones(n_coefs) * self.coef_mean

        coef_sd = np.ones(n_coefs) * self.coef_sd

        # Initialize State
        self.coef_ = coef_mean
        self.coef_covariance = np.diag(coef_sd ** 2)

        self.coef_transition = self.decay * np.eye(n_coefs)
        self.Q = (
            self.coef_covariance -
            self.coef_transition.dot(
                self.coef_covariance
            ).dot(self.coef_transition.T)
        )

        self.covariance_model.fit(X)

    def _predict_mean_cov(self, transition, state,
                          state_covariance, residual_covariance):
        return (
            np.dot(transition, state),
            np.dot(transition,
                   np.dot(state_covariance,
                          transition.T)) +
            residual_covariance
        )

    def partial_fit(self, X, y):
        X, y = self.initializer.fit(X, y).predict()

        if X is None:
            n_x = self.initializer.n_x
            n_y = self.initializer.n_y
            self.coef_ = np.ones(n_x * n_y) * self.coef_mean
            return self

        for x, y in zip(X, y):

            # Convert x to a duplicated diagonal block matrix:
            # [[x, 0, ..., 0
            #  [0, x, ..., 0]
            #  [0, 0, ..., 0]
            #  [0, 0, ..., x]]
            # so that all coefficients can be on the single 1d vector
            obs = block_diag(*([x] * len(y)))

            # See pykalman/standard.py:_filter_correct for reference
            predicted_obs_mean, predicted_obs_cov = self._predict_mean_cov(
                obs, self.coef_, self.coef_covariance,
                self.covariance_model.predict()
            )
            residual = y - predicted_obs_mean

            kalman_gain = (
                np.dot(self.coef_covariance,
                       np.dot(obs.T,
                              np.linalg.pinv(predicted_obs_cov)))
            )

            corrected_coef = (
                self.coef_ +
                np.dot(kalman_gain, residual)
            )

            corrected_coef_cov = (
                self.coef_covariance -
                np.dot(kalman_gain,
                       np.dot(obs,
                              self.coef_covariance))
            )

            # Now update residual covariance
            self.covariance_model.partial_fit(residual)

            # See pykalman/standard.py:_filter_predict for reference
            self.coef_, self.coef_covariance = self._predict_mean_cov(
                self.coef_transition, corrected_coef, corrected_coef_cov,
                self.Q
            )

    @property
    def block_coef(self):
        return self.coef_.reshape((
            self.initializer.n_y,
            self.initializer.n_x,
        )).T

    def predict(self, X):
        X = force_matrix(X, col=False)

        return np.dot(X, self.block_coef)


class FilteredMean(DynamicRidge):
    def __init__(self, span, sd, min_samples=30, covariance_model=None):
        self.span = span
        self.sd = sd
        self.min_samples = min_samples
        self.covariance_model = covariance_model

        super(FilteredMean, self).__init__(
            span=span, coef_sd=sd,
            min_samples=min_samples,
            covariance_model=covariance_model,
        )

    def partial_fit(self, X, y=None):
        return super(FilteredMean, self).partial_fit(
            np.ones(len(X)), X
        )
