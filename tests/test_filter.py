import unittest
import numpy as np
from scipy.linalg import block_diag


def r2(actual, prediction):
    sst = ((actual - actual.mean()) ** 2).sum()
    ssr = ((actual - prediction) ** 2).sum()

    return 1. - ssr / sst


def build_dynamic_ridge_data(
    random_state, n, n_x, n_y, coef_span, coef_mean,
    residual_sd, coef_sd,
):
    coef_decay = 1. - 2. / (float(coef_span) + 1.)

    # Generate coefficients:
    coefs = []
    coef = np.ones(n_x * n_y) * coef_mean
    coef_var = coef_sd ** 2
    coef_resid_sd = np.sqrt((1. - (coef_decay ** 2)) * coef_var)
    transition_matrix = coef_decay * np.eye(n_x * n_y)
    for i in range(n):
        coef = (
            transition_matrix.dot(coef) +
            random_state.normal(0., coef_resid_sd, (n_x * n_y))
        )
        coefs.append(coef)
    coefs = np.asarray(coefs)

    # Generate X
    X = random_state.normal(0., 1., (n, n_x))

    # Generate Y
    Y = np.asarray([
        (
            block_diag(*([x] * n_y)).dot(c) +
            random_state.normal(0., residual_sd, n_y)
        )
        for x, c in zip(X, coefs)
    ])

    return Y, X, coefs


class TestFilters(unittest.TestCase):
    def test_dynamic_ridge(self):
        import pandas as pd
        from pyonline.covariance import DiagonalCovariance
        from pyonline.filter import DynamicRidge

        plot = True

        dynamic_ridge = DynamicRidge(
            span=200,
            coef_sd=.3,
            min_samples=30,
            covariance_model=DiagonalCovariance()
        )

        # Get train data
        Y, X, coefs = build_dynamic_ridge_data(
            random_state=np.random.RandomState(1),
            n=1000,
            n_x=2,
            n_y=2,
            coef_span=500,
            coef_mean=0.2,
            residual_sd=1.5,
            coef_sd=.3,
        )

        predictions = []
        filtered_coefs = []
        last_prediction = np.nan * np.ones(2)
        for x, y, next_x in zip(X, Y, X[1:]):
            dynamic_ridge.partial_fit(x, y)
            corrected_coef = dynamic_ridge.coef_ / dynamic_ridge.decay

            filtered_coefs.append(corrected_coef)
            predictions.append(last_prediction.ravel())

            last_prediction = dynamic_ridge.predict(next_x)

        coef_comparison = pd.concat(
            dict(
                Actual=pd.DataFrame(coefs),
                Estimated=pd.DataFrame(filtered_coefs)
            ),
            axis=1
        ).swaplevel(0, 1, axis=1).sort_index(axis=1).iloc[30:]

        predictions = pd.concat(
            dict(
                Actual=pd.DataFrame(Y),
                Predicted=pd.DataFrame(np.asarray(predictions)),
            ), axis=1,
        ).iloc[30:]

        all_predictions = predictions.stack(1)
        oos_r2 = r2(all_predictions.Actual, all_predictions.Predicted)

        if plot:
            from pylab import show
            import seaborn

            coef_comparison.groupby(axis=1, level=0).plot(
                colormap='Paired',
                title='Estimated Coefficients',
            )

            all_predictions.plot(
                kind='scatter', x='Actual', y='Predicted',
                title='Actual vs Predicted Observations: OOS R2(%.4f)' % (
                    oos_r2
                ),
            )

            show()

        self.assertTrue(oos_r2 > 0.04)


if __name__ == '__main__':
    unittest.main()
