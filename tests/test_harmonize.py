"""
mask harmonization: the bands share one footprint.  The bad
pixels are unioned across bands in pull_mbobs, and the star
taper uses the union-mask distance field (the minimum of the
per-band distance transforms)
"""
import numpy as np

from lsst_mdet.cells import pull_mbobs
from lsst_mdet.defaults import (
    CELL_OVERLAP,
    CELL_SIZE,
    DM_NO_DATA,
    MIN_GOOD_FRAC,
)

# one-cell "patch": the cell window with its overlap fills the
# whole fake patch
DIM = CELL_SIZE + 2 * CELL_OVERLAP
PSF_DIM = 25


class Bounds:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop


class Box:
    def __init__(self, y0, y1, x0, x1):
        self.y = Bounds(y0, y1)
        self.x = Bounds(x0, x1)


class Plane:
    def __init__(self, arr):
        self.array = arr

    def __getitem__(self, box):
        return Plane(self.array[
            box.y.start: box.y.stop, box.x.start: box.x.stop,
        ])


class FakeWcs:
    def linearize_matrix(self, x, y):
        # d(sky)/d(pixel) in arcsec/pixel with mirror parity,
        # as for a real sky image
        return np.array([[0.0, 0.2], [0.2, 0.0]])


class FakeCellCoadd:
    """the interface pull_mbobs uses, for a one-cell patch"""

    def __init__(self, rng, band, sigma=2.0):
        self.band = band
        self.bbox = Box(0, DIM, 0, DIM)
        self.image = Plane(
            rng.normal(scale=sigma, size=(DIM, DIM)),
        )
        self.variance = Plane(
            np.full((DIM, DIM), sigma ** 2, dtype='f4'),
        )
        self.mask = Plane(np.zeros((DIM, DIM, 1), dtype='i4'))
        self.noise_realizations = [Plane(
            rng.normal(scale=sigma, size=(DIM, DIM)),
        )]
        self.mask_fractions = {
            'rejected': Plane(np.zeros((DIM, DIM))),
        }

    def set_bad(self, sl):
        """a hole with no data: the DM mask bit and the nan
        variance the stack writes there"""
        self.mask.array[sl + (0,)] |= DM_NO_DATA
        self.variance.array[sl] = np.nan

    def cell_window(self, cell_i, cell_j, overlap):
        return Box(0, DIM, 0, DIM)

    def psf_image(self, x, y):
        cen = (PSF_DIM - 1) / 2
        yy, xx = np.mgrid[0:PSF_DIM, 0:PSF_DIM]
        r2 = (yy - cen) ** 2 + (xx - cen) ** 2
        psf = np.exp(-0.5 * r2 / 2.0 ** 2)
        return psf / psf.sum()


def test_pull_mbobs_union_footprint():
    """a hole in one band zeroes the weight in every band, with
    mfrac = 1 there; the per-band good fractions record which
    band drove the loss"""
    rng = np.random.RandomState(3)
    coadds = [
        FakeCellCoadd(rng, 'r'),
        FakeCellCoadd(rng, 'i'),
    ]
    hole = np.s_[40:80, 60:120]
    coadds[1].set_bad(hole)

    mbobs, cell_meta = pull_mbobs(
        coadds, 1, 1, FakeWcs(),
    )
    assert mbobs is not None and cell_meta['kept'][0]

    hole_frac = (40 * 60) / DIM ** 2
    assert cell_meta['good_frac'][0, 0] == 1.0
    assert np.isclose(
        cell_meta['good_frac'][0, 1], 1.0 - hole_frac,
    )

    zero0 = mbobs[0][0].weight == 0
    zero1 = mbobs[1][0].weight == 0
    # the union footprint: identical zero sets, exactly the hole
    assert np.array_equal(zero0, zero1)
    assert np.all(zero0[hole])
    assert zero0.mean() == hole_frac

    for obs in (mbobs[0][0], mbobs[1][0]):
        assert np.all(obs.mfrac[hole] == 1.0)
        assert np.all(obs.bmask[hole] != 0)
        assert np.isclose(
            obs.meta['good_frac'], 1.0 - hole_frac,
        )


def test_pull_mbobs_union_gate():
    """disjoint holes that pass the good-fraction cut per band
    but fail it in union drop the cell"""
    rng = np.random.RandomState(5)
    coadds = [
        FakeCellCoadd(rng, 'r'),
        FakeCellCoadd(rng, 'i'),
    ]
    # each hole leaves half the area: fine per band, but the
    # disjoint union leaves nothing
    split = DIM // 2
    coadds[0].set_bad(np.s_[:split, :])
    coadds[1].set_bad(np.s_[split:, :])

    mbobs, cell_meta = pull_mbobs(
        coadds, 1, 1, FakeWcs(),
    )
    assert mbobs is None
    assert not cell_meta['kept'][0]
    # each band alone would have passed
    assert np.all(cell_meta['good_frac'][0] > MIN_GOOD_FRAC)


def test_union_distance_is_min():
    """the identity the shared star taper relies on: the
    distance to the union of the masks is the minimum of the
    per-mask distances, so the union taper zone is exactly the
    union of the per-band taper zones"""
    from scipy import ndimage

    rng = np.random.RandomState(7)
    dim = 200
    masks = []
    for k in range(3):
        m = np.zeros((dim, dim), dtype=bool)
        for _ in range(10):
            y, x = rng.randint(0, dim, size=2)
            r = rng.randint(2, 15)
            yy, xx = np.mgrid[0:dim, 0:dim]
            m |= (yy - y) ** 2 + (xx - x) ** 2 < r ** 2
        masks.append(m)

    dstars = [
        ndimage.distance_transform_edt(~m) for m in masks
    ]
    dstar_min = np.minimum.reduce(dstars)
    union = np.logical_or.reduce(masks)
    dstar_union = ndimage.distance_transform_edt(~union)

    assert np.array_equal(dstar_min, dstar_union)

    # and dstar_min == 0 recovers the union mask itself
    assert np.array_equal(dstar_min == 0, union)

    apod = 20.0
    zones = [d < apod for d in dstars]
    assert np.array_equal(
        dstar_min < apod, np.logical_or.reduce(zones),
    )
