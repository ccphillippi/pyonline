from sklearn.base import BaseEstimator
import numpy as np


def force_matrix(X, col=False):
    row_vector = np.atleast_2d(X)
    if col and (row_vector.shape[0] == 1):
        return row_vector.T

    return row_vector


class Initializer(BaseEstimator):
    def __init__(self, base_obj, initializer, min_samples=None):
        self.min_samples = min_samples
        self.base_obj = base_obj
        self.initializer = initializer

        assert hasattr(self.base_obj, self.initializer)

        self.min_samples = min_samples

        self.n_samples = 0

        self.Xs = []
        self.Ys = []
        self.n_x = None
        self.n_y = None

    def fit(self, X, y=None):
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        X = np.asarray(X)
        y = None if y is None else np.asarray(y)

        x_is_2d = len(X.shape) == 2
        y_is_2d = False if y is None else len(y.shape) == 2

        X = force_matrix(X, col=y_is_2d)
        if self.n_x is None:
            self.n_x = X.shape[1]
        elif self.n_x != X.shape[1]:
            raise RuntimeError(
                'Currently X has %d columns, yet previously had %d.' % (
                    X.shape[1], self.n_x
                )
            )

        # Convert y to 2d if not None, else ensure it's always None
        if y is None:
            if self.Ys:
                raise RuntimeError(
                    'Current y is None but previous Ys were:\n%s' % self.Ys)
        else:
            y = force_matrix(y, col=x_is_2d)

            if self.n_y is None:
                self.n_y = y.shape[1]
            elif self.n_y != y.shape[1]:
                raise RuntimeError(
                    'Currently y has %d columns, yet previously had %d.' % (
                        y.shape[1], self.n_y
                    )
                )

            if len(X) != len(y):
                raise RuntimeError(
                    'X should be the same length as y:\nX:\n%s\ny:\n%s' % (
                        X, y
                    )
                )

        # If already initialized (self.Xs is None) return
        if self.Xs is None:
            self.X, self.y = X, y
            return self

        # Stash X and y for later
        self.Xs.append(X)
        self.Ys.append(y)
        self.n_samples += len(X)

        # If we don't have enough yet, quit
        if self.n_samples < self.min_samples:
            self.X, self.y = None, None
            return self

        # We have at least enough, initialize
        initializer = getattr(self.base_obj, self.initializer)
        X_init = np.vstack(self.Xs)
        if y is not None:
            Y_init = np.vstack(self.Ys)
        else:
            Y_init = None

        # Initialize model
        initializer(X_init[:self.min_samples],
                    None if y is None else Y_init[:self.min_samples])
        # Release memory
        self.Xs = None
        self.Ys = None

        # Edge case where none left over
        if self.n_samples == self.min_samples:
            self.X, self.y = None, None
            return self

        # Return leftover in predict
        self.X = X_init[self.min_samples:]
        self.y = None if y is None else Y_init[self.min_samples:]
        return self

    def predict(self, X=None, y=None):
        return self.X, self.y
