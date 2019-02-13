import numpy as np
from sklearn.base import BaseEstimator

class Config:
    DISABLE_JIT = False

def numba_jit(nopython=True, nogil=True, **kwds):
    def wrapped(func):

        if Config.DISABLE_JIT:
            return func

        try:
            from numba import jit
        except ImportError:
            return func

        return jit(nopython=nopython, nogil=nogil, **kwds)(func)

    return wrapped


@numba_jit()
def _fit_monotone_spline(sorted_xs, sorted_ys, n, coefs):
    dys = np.empty(n - 1, dtype=np.float64)
    dxs = np.empty(n - 1, dtype=np.float64)
    ms = np.empty(n - 1, dtype=np.float64)

    for i in range(n - 1):
        dxs[i] = dx = float(sorted_xs[i + 1] - sorted_xs[i])
        dys[i] = dy = float(sorted_ys[i + 1] - sorted_ys[i])
        ms[i] = dy / dx

    # Get linear coefficients
    coefs[0, 0] = ms[0]
    coefs[n - 1, 0] = ms[n - 2]
    for i in range(n - 2):
        m = ms[i]
        m_next = ms[i + 1]
        if ((m * m_next) <= 0):
            coefs[i + 1, 0] = 0.
        else:
            dx = dxs[i]
            dx_next = dxs[i + 1]
            dx_p_dx = dx + dx_next
            coefs[i + 1, 0] = (
                3. * dx_p_dx /
                (
                    (dx_p_dx + dx_next) / m +
                    (dx_p_dx + dx) / m_next
                )
            )

    # Get quadratic/cubic coefficients
    for i in range(n - 1):
        c1 = coefs[i, 0]
        m = ms[i]
        inv_dx = 1. / dxs[i]
        excess_c1 = c1 + coefs[i + 1, 0] - m - m
        coefs[i, 1] = (m - c1 - excess_c1) * inv_dx
        coefs[i, 2] = excess_c1 * inv_dx * inv_dx


@numba_jit()
def _interpolate_monotone_spline_one(x, xs, ys, coefs, n):
    if (x >= xs[n - 1]):
        return ys[n - 1]

    # Binary search
    low = 0
    high = n - 2
    while low <= high:
        m = (low + high) // 2

        here = xs[m]
        if here < x:
            low = m + 1
        elif here > x:
            high = m - 1
        else:
            return ys[m]

    i = max(0, high)
    d = x - xs[i]

    return ys[i] + d * (coefs[i, 0] + d * (coefs[i, 1] + d * coefs[i, 2]))


@numba_jit()
def _interpolate_monotone_spline(x, xs, ys, coefs, n, n_x):
    out = np.empty(n_x, dtype=np.float64)
    for i in range(n_x):
        out[i] = _interpolate_monotone_spline_one(
            x[i], xs, ys, coefs, n
        )

    return out


@numba_jit()
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


@numba_jit()
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


@numba_jit()
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


@numba_jit()
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


@numba_jit()
def _update_markers(x, markers, quantiles, n_markers):
    # Find the quantile bucket (k)
    # and handle edge cases, k=0 or k=N
    if x >= quantiles[n_markers - 1]:
        quantiles[n_markers - 1] = x
        k = n_markers

        # won't be incremented later since k == n_markers
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

    return k


@numba_jit()
def _update_state(xs, markers, quantiles, desired_markers,
                  n_x, n_markers, dn):
    for i in range(n_x):
        x = xs[i]

        _update_markers(x, markers, quantiles, n_markers)

        n_m1 = float(markers[n_markers - 1] - 1.)
        for j in range(n_markers):
            # N-1 because we're adding 1
            desired_markers[j] = 1. + dn[j] * n_m1

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


@numba_jit()
def _update_state_pcubed(xs, markers, quantiles, desired_markers,
                         n_x, n_markers, dn, coefs):

    for i in range(n_x):
        x = xs[i]

        _update_markers(x, markers, quantiles, n_markers)

        _fit_monotone_spline(markers, quantiles, n_markers, coefs)

        n_m1 = float(markers[n_markers - 1] - 1.)
        for j in range(n_markers):
            # N-1 because we're adding 1
            desired_markers[j] = 1. + dn[j] * n_m1

        # Update inner quantiles
        for j in range(1, n_markers - 1):

            ni = markers[j]
            n = float(ni)
            d = desired_markers[j] - n

            if (
                ((d >= 1.) and ((markers[j + 1] - ni) > 1)) or
                ((d <= -1) and ((ni - markers[j - 1]) > 1))
            ):
                if d > 0.:
                    markers[j] += 1
                    quantiles[j] += coefs[j, 0] + coefs[j, 1] + coefs[j, 2]
                else:
                    markers[j] -= 1
                    d = n - 1. - markers[j - 1]
                    quantiles[j] = (
                        quantiles[j-1] + d * (coefs[j-1, 0] + d * (coefs[j-1, 1] + d * coefs[j-1, 2]))
                    )


@numba_jit()
def _update_adaptive_state(xs, markers, quantiles, desired_markers,
                           n_x, n_markers, dn):

    nm1 = n_markers - 1
    # store derivatives
    dx = np.empty(nm1, dtype=np.float64)
    dy = np.empty(nm1, dtype=np.float64)
    dydx = np.empty(nm1, dtype=np.float64)
    f_1 = np.empty(n_markers, dtype=np.float64)
    curvature = np.empty(n_markers - 1, dtype=np.float64)

    for i in range(n_x):
        x = xs[i]

        k = _update_markers(x, markers, quantiles, n_markers)

        for j in range(n_markers - 1):
            dx[j] = float(markers[j + 1] - markers[j])
            dy[j] = quantiles[j + 1] - quantiles[j]
            dydx[j] = dy[j] / dx[j]

        dydx_prev = dydx[0]
        dx_prev = dx[0]
        f_1[0] = dydx_prev - dx_prev / (dx_prev + dx[1]) * (dydx[1] - dydx_prev)
        for j in range(1, n_markers - 1):
            dx_curr = dx[j]
            dx_prev = dx[j - 1]
            f_1[j] = (
                dx_prev * dydx[j] +
                dx_curr * dydx[j - 1]
            ) / (dx_prev + dx_curr)

        dx_prev = dx[nm1 - 1]
        dydx_prev = dydx[nm1 - 1]
        f_1[nm1] = dydx_prev + (
            dx_prev / (dx_prev + dx[nm1 - 2]) * (dydx_prev - dydx[nm1 - 2])
        )

        for j in range(n_markers - 1):
            _dx = dx[j] / float(markers[nm1])

            # Notice we're using abs(f'' * dx) here in order to approximate the integral over x
            curvature[j] = np.abs((f_1[j + 1] - f_1[j]) / dydx[j]) * _dx

        n_m1 = float(markers[n_markers - 1] - 1.)
        desired_markers[0] = 1.
        for j in range(1, n_markers - 1):
            left_curve = curvature[j - 1]
            right_curve = curvature[j]

            forward = right_curve - left_curve
            dn_plus = desired_markers[j + 1] - desired_markers[j]
            dn_minus = desired_markers[j] - desired_markers[j - 1]
            threshold = 30.

            if (j >= k) and (forward > 0.) and (dn_plus > threshold):
                dn[j] += 1. / float(n_m1)
            elif (j < k) and (forward < 0.) and (dn_minus > threshold):
                dn[j] -= 1. / float(n_m1)

            # N-1 because we're adding 1
            desired_markers[j] = 1. + dn[j] * n_m1
        desired_markers[n_markers - 1] = n_m1 + 1

        # Update inner quantiles
        for j in range(1, n_markers - 1):

            dj = desired_markers[j] - markers[j]
            n_m1 = markers[j - 1]
            n = markers[j]
            n_p1 = markers[j + 1]
            dn_plus = (n_p1 - n)
            dn_minus = (n - n_m1)

            if (
                ((dj >= 1.) and (dn_plus > 1)) or
                ((dj <= -1) and (dn_minus > 1))
            ):
                if dj > 0.:
                    dj = 1.
                else:
                    dj = -1.
                quantiles[j] = _interpolate_p2(
                    dj, float(n_m1), float(n), float(n_p1),
                    quantiles[j - 1],
                    quantiles[j],
                    quantiles[j + 1],
                )
                markers[j] += dj

    return curvature

class Percentiles(BaseEstimator):
    FLOAT = np.float64
    INT = np.int64

    def __init__(self, n_markers=10):
        self.n_markers = n_markers

        self.desired_markers = np.arange(
            self.n_markers, dtype=self.FLOAT
        ) + 1.
        self.markers = np.asarray(self.desired_markers[:], dtype=self.INT)

        self.dn = np.linspace(0., 1., self.n_markers)
        self.quantiles = np.nan * np.ones(len(self.desired_markers))
        self.initial_samples = []
        self._use_partial_fit = self._initial_partial_fit

        self.monotonic_interpolation = None
        self.monotonic_inverse_interpolation = None

    def _initial_partial_fit(self, x):

        x = np.asarray(x)
        assert len(x.shape) == 1

        n_samples = self.n_markers - len(self.initial_samples)
        self.initial_samples = (
            self.initial_samples +
            list(x[:n_samples])
        )

        if len(self.initial_samples) == len(self.markers):
            self.quantiles = np.array(
                sorted(self.initial_samples)
            ).astype(np.float64)
            self._use_partial_fit = self._partial_fit

        if len(x) > n_samples:
            return self._partial_fit(x[n_samples:])

        return self

    def _partial_fit(self, x):
        _update_state(
            np.asarray(x),
            self.markers,
            self.quantiles,
            self.desired_markers, len(x),
            self.n_markers,
            self.dn
        )

        self.monotonic_interpolation = None
        self.monotonic_inverse_interpolation = None

        return self

    def partial_fit(self, x, y=None):
        return self._use_partial_fit(x)

    def transform(self, x, cubic_interp=False):
        x = np.asarray(x)

        if cubic_interp:
            coefs = self.get_monotonic_interpolation()
            rank = _interpolate_monotone_spline(
                x, self.quantiles, self.markers.astype(self.FLOAT),
                coefs, len(coefs), len(x)
            )
        else:
            rank = np.empty(x.shape, dtype=self.FLOAT)
            _cdf(
                x, self.quantiles, self.markers.astype(self.FLOAT),
                len(x), self.n_markers, rank)

        return (rank - 1.) / (self.markers[-1] - 1.)

    def inverse_transform(self, y, cubic_interp=False):
        y = np.asarray(y)
        rank = y * (self.markers[-1] - 1.) + 1.

        if cubic_interp:
            coefs = self.get_monotonic_inverse_interpolation()
            x = _interpolate_monotone_spline(
                rank, self.markers.astype(self.FLOAT), self.quantiles,
                coefs, len(coefs), len(y)
            )
        else:
            x = np.empty(y.shape, self.FLOAT)
            _cdf(
                rank, self.markers.astype(self.FLOAT), self.quantiles,
                len(y), self.n_markers, x
            )

        return x

    def get_monotonic_inverse_interpolation(self):

        out = self.monotonic_inverse_interpolation
        if out is None:
            out = np.empty((self.n_markers, 3), dtype=self.FLOAT)
            _fit_monotone_spline(
                self.markers,
                self.quantiles,
                len(self.quantiles),
                out
            )
            self.monotonic_inverse_interpolation = out

        return out

    def get_monotonic_interpolation(self):

        out = self.monotonic_interpolation
        if out is None:
            out = np.empty((self.n_markers, 3), dtype=self.FLOAT)
            _fit_monotone_spline(
                self.quantiles,
                self.markers,
                len(self.quantiles),
                out
            )
            self.monotonic_interpolation = out

        return out


class PCubed(Percentiles):
    def _partial_fit(self, x):
        self.curvature = _update_adaptive_state(
            np.asarray(x),
            self.markers,
            self.quantiles,
            self.desired_markers,
            len(x),
            self.n_markers,
            self.dn
        )

        self.monotonic_interpolation = None
        self.monotonic_inverse_interpolation = None

        return self
