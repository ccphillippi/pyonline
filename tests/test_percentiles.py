import unittest
import numpy as np
import pandas as pd


def to_percentile(markers):
    return (markers - 1.) / (markers[-1] - 1.)

class TestPercentiles(unittest.TestCase):
    def test_pcubed(self):
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from pyonline.percentiles import Percentiles, PCubed
        from time import time as seconds

        n = 10000000
        random_state = np.random.RandomState(10)
        #X = random_state.exponential(0.00000000000001, size=(n,))
        X = random_state.normal(0, 1, size=(n,))

        n_markers = 20
        def get_psquared():
            return Percentiles(n_markers=n_markers)

        def get_pcubed():
            return PCubed(n_markers=n_markers)

        # JIT compile the code
        get_psquared().partial_fit(X[:n_markers]).transform(X[:n_markers], cubic_interp=True)
        get_pcubed().partial_fit(X[:n_markers]).transform(X[:n_markers], cubic_interp=True)

        psquared = get_psquared()
        pcubed = get_pcubed()

        start_seconds = seconds()
        pcubed_percentiles = pcubed.partial_fit(X).transform(X, cubic_interp=True)
        stop_seconds = seconds()

        start_seconds = seconds()
        percentiles = pcubed.partial_fit(X).transform(X, cubic_interp=True)
        stop_seconds = seconds()
        pcubed_seconds = stop_seconds - start_seconds

        print('P-Cubed in %.2f seconds!' % pcubed_seconds)

        true_percentiles = pd.Series(X).rank() / X.shape[0]
        marker_percentiles = to_percentile(percentile_model.markers)

        gs = gridspec.GridSpec(2, 2)
        top_left = plt.subplot(gs[0, :1])
        bottom_left = plt.subplot(gs[1, :1])
        right = plt.subplot(gs[:, 1:])

        plot_n = 10000
        plot_X = X[:plot_n]

        pd.DataFrame(
            dict(
                Actual=plot_X,
                PSquare=percentile_model.inverse_transform(true_percentiles.values[:plot_n]),
                PCubed=percentile_model.inverse_transform(true_percentiles.values[:plot_n], cubic_interp=True)
            ),
            index=true_percentiles.values[:plot_n],
        ).sort_index().plot(ax=top_left, title="n = %d" % n, colormap='Paired')

        for marker in marker_percentiles:
            top_left.axvline(marker, alpha=0.5, color='grey')
            bottom_left.axvline(marker, alpha=0.5, color='grey')

        top_left.scatter(marker_percentiles, percentile_model.quantiles)

        if hasattr(percentile_model, 'curvature'):

            print(percentile_model.curvature)
            pd.DataFrame(
                dict(
                    StressLeft=percentile_model.curvature[:-1],
                    StressRight=percentile_model.curvature[1:]
                ),
                index=marker_percentiles[1:-1]
            ).plot(ax=bottom_left)

        pd.DataFrame(
            dict(
                Actual=true_percentiles[:10000],
                Estimated=percentiles[:10000]
            )
        ).plot.scatter(
            ax=right,
            x='Actual',
            y='Estimated',
            figsize=(9, 7)
        )

        fig = plt.gcf()
        fig.set_size_inches(24, 12)
        plt.tight_layout()
        plt.show()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()