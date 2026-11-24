import numpy as np
import ngmix

from lsst_mdet.deblend import DEBLEND_SETTINGS, cell_noise_autocovs


def _boxcar_noise(rng, shape, sigma):
    ny, nx = shape
    w = rng.normal(size=(ny + 2, nx + 2))
    n = sum(w[i:i + ny, j:j + nx] for i in range(3) for j in range(3))
    return n * (sigma / 3.0)


def test_cell_noise_autocovs():
    """
    the per-band autocovariance of the cell noise field: the known
    correlation of each band, with the masked pixels left out
    """
    rng = np.random.RandomState(5)
    shape = (250, 250)
    sigmas = [2.0, 5.0]
    max_lag = DEBLEND_SETTINGS['noise_max_lag']
    assert max_lag > 0

    mbobs = ngmix.MultiBandObsList()
    for sigma in sigmas:
        noise = _boxcar_noise(rng, shape, sigma)
        weight = np.full(shape, 1.0 / sigma ** 2)
        # a masked region, zeroed as the star taper leaves it
        weight[40:90, 100:160] = 0.0
        noise[weight == 0] = 0.0
        obslist = ngmix.ObsList()
        obslist.append(ngmix.Observation(
            rng.normal(size=shape), weight=weight, noise=noise,
        ))
        mbobs.append(obslist)

    acovs = cell_noise_autocovs(mbobs, max_lag)
    assert len(acovs) == len(sigmas)
    tri = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 3.0
    for acov, sigma in zip(acovs, sigmas):
        n = 2 * max_lag + 1
        assert acov.shape == (n, n)
        truth = np.zeros((n, n))
        truth[max_lag - 2:max_lag + 3, max_lag - 2:max_lag + 3] = (
            np.outer(tri, tri) * sigma ** 2
        )
        assert np.all(np.abs(acov - truth) < 0.04 * sigma ** 2)
        assert np.allclose(acov, acov[::-1, ::-1])
