"""
the psf fits of a cell: a good psf is fit and its gmix set; a psf
the adaptive moments cannot fit (run-dp2-phot-v01, 05873-00012: the
original coadd psf of one cell on the photometry-only route) is
reported through the flags instead of an assertion, so the cell's
detections get PSF_FAILURE rather than the patch dying
"""
import numpy as np
import ngmix

from lsst_mdet.psf import fit_and_set_mcal_psfs

SCALE = 0.2


def make_obs(psf_image, band):
    n = psf_image.shape[0]
    cen = (n - 1) / 2
    jac = ngmix.DiagonalJacobian(scale=SCALE, row=cen, col=cen)
    psf_obs = ngmix.Observation(
        psf_image, weight=np.ones_like(psf_image), jacobian=jac,
    )
    obs = ngmix.Observation(
        np.zeros((n, n)), weight=np.ones((n, n)), jacobian=jac, psf=psf_obs,
    )
    obs.meta['band'] = band
    obslist = ngmix.ObsList()
    obslist.append(obs)
    return obslist


def gauss_psf(n=25, sigma=1.5):
    yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2
    im = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    return im / im.sum()


def test_psf_fit_good():
    rng = np.random.RandomState(1)
    mbobs = ngmix.MultiBandObsList()
    for band in ('r', 'i'):
        mbobs.append(make_obs(gauss_psf(), band))
    res = fit_and_set_mcal_psfs(mbobs, weights=[1.0, 1.0], rng=rng)
    assert res['psf_flags'] == 0
    assert all(obslist[0].psf.has_gmix() for obslist in mbobs)
    # T = 2 sigma^2 in arcsec^2
    assert abs(res['psf_T'] / (2 * (1.5 * SCALE) ** 2) - 1) < 0.02


def test_psf_fit_failure_flags():
    rng = np.random.RandomState(1)
    mbobs = ngmix.MultiBandObsList()
    mbobs.append(make_obs(gauss_psf(), 'r'))
    # an image the moments cannot converge on: all zero
    mbobs.append(make_obs(np.zeros((25, 25)), 'i'))
    res = fit_and_set_mcal_psfs(mbobs, weights=[1.0, 1.0], rng=rng)
    assert res['psf_flags_r'] == 0
    assert res['psf_flags_i'] != 0
    assert res['psf_flags'] != 0
    assert np.isnan(res['psf_T'])
    assert np.isnan(res['psf_T_i'])
    assert not mbobs[1][0].psf.has_gmix()
