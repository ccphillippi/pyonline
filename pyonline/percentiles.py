import numpy as np
from sklearn.base import BaseEstimator


def numba_jit(func):
    try:
        from numba import jit
    except ImportError:
        return func

    return jit(func)


@numba_jit
def _closest_index(x, array, n_array):
    left = 0
    right = n_array - 1

    while left <= right:
        m = (left + right) // 2

        if array[m] < x:
            left = m + 1
        elif array[m] > x:
            right = m - 1
        else:
            return m

    if left == n_array:
        return right
    elif right == -1:
        return left
    elif (x - array[right]) < (array[left] - x):
        return right

    return left


@numba_jit
def _interpolate_p2_quadratic(d, xm1, x, xp1, ym1, y, yp1):
    dx_plus = xp1 - x
    dx_minus = x - xm1
    dy_plus = yp1 - y
    dy_minus = y - ym1

    return (
        y +
        (d / (xp1 - xm1)) * (
            (dx_minus + d) * (dy_plus / dx_plus) +
            (dx_plus - d) * (dy_minus / dx_minus)
        )
    )


@numba_jit
def _cdf(xp, xs, ys, n_xp, n_markers, out):
    for i in range(n_xp):
        # Find k
        k = _closest_index(xp[i], xs, n_markers)
        if k == 0:
            k += 1
        elif k == (n_markers - 1):
            k -= 1

        xm1 = xs[k - 1]
        x = xs[k]
        xp1 = xs[k + 1]
        ym1 = ys[k - 1]
        y = ys[k]
        yp1 = ys[k + 1]

        yp = _interpolate_p2_quadratic(
            xp[i] - x, xm1, x, xp1, ym1, y, yp1
        )

        if yp < ym1:
            out[i] = ym1
        elif yp > yp1:
            out[i] = yp1
        else:
            out[i] = yp


@numba_jit
def _interpolate_p2(d, xm1, x, xp1, ym1, y, yp1):
    yp = _interpolate_p2_quadratic(
        d, xm1, x, xp1, ym1, y, yp1
    )

    if (
        (yp <= ym1) or
        (yp >= yp1)
    ):
        if d > 0.:
            dy = yp1 - y
            dx = xp1 - x
        else:
            dy = y - ym1
            dx = x - xm1

        yp = y + d * (dy / dx)

    return yp


@numba_jit
def _update_state(xs, markers, quantiles, desired_markers, dn,
                  n_x, n_markers):
    for i in range(n_x):
        x = xs[i]

        # Find the quantile bucket (k)
        # and handle edge cases, k=0 or k=N
        if x >= quantiles[n_markers - 1]:
            quantiles[n_markers - 1] = x
            k = n_markers
            markers[n_markers - 1] += 1
        elif x <= quantiles[0]:
            quantiles[0] = x
            k = 1
        else:
            for j in range(1, n_markers):
                if x < quantiles[j]:
                    k = j
                    break

        # Update markers/desired markers
        for j in range(k, n_markers):
            markers[j] += 1

        bins_m1 = markers[n_markers - 1] - 1
        for j in range(n_markers - 1):
            desired_markers[j] = 1. + (float(j) / (n_markers - 1)) * bins_m1
        desired_markers[n_markers - 1] = float(markers[j])

        # Update inner quantiles
        for j in range(1, n_markers - 1):

            dj = desired_markers[j] - markers[j]
            nm1 = markers[j - 1]
            n = markers[j]
            np1 = markers[j + 1]
            dn_plus = (np1 - n)
            dn_minus = (n - nm1)

            if (
                ((dj >= 1.) and (dn_plus > 1)) or
                ((dj <= -1) and (dn_minus > 1))
            ):
                if dj > 0.:
                    dj = 1.
                else:
                    dj = -1.
                quantiles[j] = _interpolate_p2(
                    dj, float(nm1), float(n), float(np1),
                    quantiles[j - 1],
                    quantiles[j],
                    quantiles[j + 1],
                )
                markers[j] += dj


class CDFEstimator(BaseEstimator):
    def __init__(self, n_markers=10):
        self.n_markers = n_markers
        self.markers = np.arange(self.n_markers, dtype=np.int64) + 1
        self.desired_markers = np.arange(self.n_markers, dtype=np.float64) + 1.
        self.quantiles = np.nan * np.ones(self.n_markers)
        self.dn = np.linspace(0., 1., n_markers)
        self.initial_samples = []
        self._use_partial_fit = self._initial_partial_fit

    def _initial_partial_fit(self, x):

        x = np.asarray(x)
        assert len(x.shape) == 1

        n_samples = self.n_markers - len(self.initial_samples)
        self.initial_samples = (
            self.initial_samples +
            list(x[:n_samples])
        )

        if len(self.initial_samples) == self.n_markers:
            self.quantiles = np.array(
                sorted(self.initial_samples)
            ).astype(np.float32)
            self._use_partial_fit = self._partial_fit

        if len(x) > n_samples:
            return self._partial_fit(x[n_samples:])

        return self

    def _partial_fit(self, x):
        _update_state(
            np.asarray(x), self.markers, self.quantiles,
            self.desired_markers, self.dn,
            len(x), self.n_markers
        )

        return self

    def partial_fit(self, x, y=None):
        return self._use_partial_fit(x)

    def transform(self, x):
        x = np.asarray(x)

        rank = np.empty(x.shape, dtype=np.float32)
        _cdf(
            x, self.quantiles, self.markers.astype(np.float32),
            len(x), self.n_markers, rank)

        return (rank - 1.) / (self.markers[-1] - 1.)
