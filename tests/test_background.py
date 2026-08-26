"""
redo_background returns the sky-variance map: measured from
the image, tracking depth structure, free of object poisson,
and consumed by the pixel weights
"""
import numpy as np
import pytest

from lsst_mdet.background import redo_background
from lsst_mdet.cells import _make_cell_obs

DIM = 512


class Plane:
    def __init__(self, arr):
        self.array = arr


class FakeCoadd:
    def __init__(self, image, var, noise):
        self.image = Plane(image)
        self.variance = Plane(var)
        self.mask = Plane(
            np.zeros(image.shape + (1,), dtype='i4'),
        )
        self.noise_realizations = [Plane(noise)]
        self.band = 'r'


def test_skyvar_uniform():
    rng = np.random.RandomState(5)
    sigma = 2.0
    coadd = FakeCoadd(
        rng.normal(scale=sigma, size=(DIM, DIM)),
        np.full((DIM, DIM), sigma ** 2, dtype='f4'),
        rng.normal(scale=sigma, size=(DIM, DIM)),
    )
    skyvar = redo_background(coadd)
    assert skyvar.shape == (DIM, DIM)
    assert np.all(skyvar > 0)
    assert np.median(skyvar) == pytest.approx(
        sigma ** 2, rel=0.1,
    )
    # background subtracted in place
    assert np.abs(np.mean(coadd.image.array)) < 0.05


def test_skyvar_tracks_depth_step():
    rng = np.random.RandomState(7)
    s1, s2 = 2.0, 3.0
    image = rng.normal(scale=s1, size=(DIM, DIM))
    image[:, DIM // 2:] = rng.normal(
        scale=s2, size=(DIM, DIM // 2),
    )
    var = np.full((DIM, DIM), s1 ** 2, dtype='f4')
    var[:, DIM // 2:] = s2 ** 2
    coadd = FakeCoadd(
        image, var, rng.normal(scale=s1, size=(DIM, DIM)),
    )
    skyvar = redo_background(coadd)
    # medians per side, away from the boundary
    left = np.median(skyvar[:, :DIM // 2 - 96])
    right = np.median(skyvar[:, DIM // 2 + 96:])
    assert left == pytest.approx(s1 ** 2, rel=0.15)
    assert right == pytest.approx(s2 ** 2, rel=0.15)


def test_weight_from_weight_var():
    import ngmix

    rng = np.random.RandomState(11)
    dim = 50
    image = rng.normal(size=(dim, dim))
    # raw variance with a fake poisson bump; the weight must
    # come from the sky-variance argument instead
    weight_var = np.full((dim, dim), 2.0)
    weight_var[10:20, 10:20] = 3.0
    good = np.ones((dim, dim), dtype=bool)
    good[0, 0] = False

    pdim = 21
    pc = (pdim - 1) / 2
    gy, gx = np.mgrid[0:pdim, 0:pdim]
    psf_im = np.exp(
        -((gy - pc) ** 2 + (gx - pc) ** 2) / (2 * 2.0 ** 2)
    )
    jac = ngmix.DiagonalJacobian(
        scale=0.2, row=(dim - 1) / 2, col=(dim - 1) / 2,
    )
    obs = _make_cell_obs(
        image=image,
        weight_var=weight_var,
        good=good,
        noise=rng.normal(size=(dim, dim)),
        mfrac=np.zeros((dim, dim)),
        psf_image=psf_im,
        jacobian=jac,
    )
    assert obs.weight[0, 0] == 0
    assert obs.weight[5, 5] == pytest.approx(1 / 2.0)
    assert obs.weight[15, 15] == pytest.approx(1 / 3.0)
