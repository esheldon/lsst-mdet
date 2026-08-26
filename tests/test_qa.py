"""
the stacked star-residual QA: pure noise around fake census
stars must stack to zero within the errors
"""
import numpy as np

from lsst_mdet.qa import (
    QA_GBINS, measure_stacked_star_residuals,
)

DIM = 900


class Plane:
    def __init__(self, arr):
        self.array = arr


class FakeCoadd:
    def __init__(self, image, var):
        self.image = Plane(image)
        self.variance = Plane(var)
        self.mask = Plane(
            np.zeros(image.shape + (1,), dtype='i4'),
        )
        self.band = 'r'


def test_stacked_residuals_zero_on_noise():
    rng = np.random.RandomState(3)
    sigma = 2.0
    coadd = FakeCoadd(
        rng.normal(scale=sigma, size=(DIM, DIM)),
        np.full((DIM, DIM), sigma ** 2, dtype='f4'),
    )
    # fake faint census stars at interior positions
    n = 6
    star_table = np.zeros(n, dtype=[
        ('x', 'f8'), ('y', 'f8'), ('G', 'f4'),
        ('on_image', 'i2'),
    ])
    star_table['x'] = rng.uniform(300, DIM - 300, size=n)
    star_table['y'] = rng.uniform(300, DIM - 300, size=n)
    star_table['G'] = 17.0
    star_table['on_image'] = 1
    starmask = np.zeros((DIM, DIM), dtype=bool)

    results = measure_stacked_star_residuals(
        coadd, star_table, starmask,
    )
    assert len(results) == len(QA_GBINS)
    # only the faint bin is populated
    assert results[0]['nstars'] == 0
    assert results[2]['nstars'] == n

    res = results[2]
    w = np.isfinite(res['stacked'])
    assert w.any()
    # pure noise: consistent with zero
    assert np.all(
        np.abs(res['stacked'][w]) < 5 * res['err'][w],
    )
    assert np.nanmax(np.abs(res['stacked'])) < 0.05
