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


def test_inject_sersic():
    """
    a sersic_n column selects a truncated Sersic with the truth flux
    in the image; n = 1 is the exponential; a cuspy n = 4 profile is
    drawn through the psf without losing flux
    """
    import numpy.lib.recfunctions as rf

    coadd = make_coadd()
    truth = make_truth([
        dict(x=60.0, y=60.0, flux_r=1000.0, hlr=0.4, g1=0.0, g2=0.1),
        dict(x=180.0, y=140.0, flux_r=1000.0, hlr=0.4, g1=0.0, g2=0.1),
    ])
    truth = rf.append_fields(truth, 'sersic_n', [1.0, 4.0], usemask=False)
    assert inject_objects(coadd, FakeWcs(), truth) == 2
    im = coadd.image.array
    assert abs(im.sum() / 2000.0 - 1) < 0.005
    # the two profiles differ (the truncated n = 4 carries more of
    # its flux in the wings than the exponential)
    exp_cut = im[40:81, 40:81]
    ser_cut = im[120:161, 160:201]
    assert np.abs(ser_cut - exp_cut).max() > 0.05 * exp_cut.max()
    assert ser_cut[10:31, 10:31].sum() < exp_cut[10:31, 10:31].sum()
    # the same objects without the column are both exponentials
    coadd2 = make_coadd()
    inject_objects(coadd2, FakeWcs(), truth[list(TRUTH_COLUMNS)])
    im2 = coadd2.image.array
    assert abs(im2.sum() / 2000.0 - 1) < 0.005
    assert np.allclose(im2[120:161, 160:201], im2[40:81, 40:81])


def test_inject_clipped_at_edge():
    coadd = make_coadd()
    truth = make_truth([dict(x=2.0, y=198.0, flux_r=1000.0, is_star=True)])
    assert inject_objects(coadd, FakeWcs(), truth) == 1
    # the image ends 2.5 px left of the star and 1.5 px above it
    # (psf sigma 2 px): Phi(1.25) Phi(0.75) of the flux lands on it
    assert abs(coadd.image.array.sum() / 1000.0 - 0.894 * 0.773) < 0.02
