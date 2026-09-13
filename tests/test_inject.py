"""
the object injection: flux and position of the drawn objects on a
fake coadd with a gaussian psf and a mirror-parity 0.2 arcsec wcs,
and clipping at the image edge
"""
from types import SimpleNamespace

import numpy as np

from lsst_mdet.inject import TRUTH_COLUMNS, inject_objects


class FakePsf:
    def compute_kernel_image(self, x, y):
        yy, xx = np.mgrid[-12:13, -12:13]
        k = np.exp(-0.5 * (xx ** 2 + yy ** 2) / 2.0 ** 2)
        return SimpleNamespace(array=k / k.sum())


class FakeWcs:
    def linearize_matrix(self, x, y):
        # d(ra cos dec, dec) / d(x, y) in arcsec/pixel, east left
        return np.array([[-0.2, 0.0], [0.0, 0.2]])


def make_coadd(shape=(200, 240)):
    return SimpleNamespace(
        band='r',
        bbox=SimpleNamespace(x=SimpleNamespace(start=1000),
                             y=SimpleNamespace(start=2000)),
        psf=FakePsf(),
        image=SimpleNamespace(array=np.zeros(shape)),
    )


def make_truth(rows):
    dtype = [(c, '?' if c == 'is_star' else 'f8') for c in TRUTH_COLUMNS]
    truth = np.zeros(len(rows), dtype=dtype)
    for k, row in enumerate(rows):
        for name, val in row.items():
            truth[name][k] = val
    return truth


def test_inject_flux_and_position():
    coadd = make_coadd()
    truth = make_truth([
        dict(x=100.3, y=80.7, flux_r=1000.0, hlr=0.5, g1=0.1, g2=0.0),
        dict(x=180.0, y=150.0, flux_r=500.0, is_star=True),
    ])
    n = inject_objects(coadd, FakeWcs(), truth)
    assert n == 2
    im = coadd.image.array
    assert abs(im.sum() / 1500.0 - 1) < 0.005

    # the galaxy's centroid, in the patch frame
    yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    gal = (np.abs(xx - 100) < 40) & (np.abs(yy - 80) < 40)
    w = im * gal
    assert abs((w * xx).sum() / w.sum() - 100.3) < 0.02
    assert abs((w * yy).sum() / w.sum() - 80.7) < 0.02


def test_inject_clipped_at_edge():
    coadd = make_coadd()
    truth = make_truth([dict(x=2.0, y=198.0, flux_r=1000.0, is_star=True)])
    assert inject_objects(coadd, FakeWcs(), truth) == 1
    # the image ends 2.5 px left of the star and 1.5 px above it
    # (psf sigma 2 px): Phi(1.25) Phi(0.75) of the flux lands on it
    assert abs(coadd.image.array.sum() / 1000.0 - 0.894 * 0.773) < 0.02
