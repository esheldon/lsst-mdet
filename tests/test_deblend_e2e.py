"""
fit_deblend end to end on a synthetic two-band scene with the exp
and the ladder models: detection, grouping, the kdeblend fit with
the footprint size passed through, and the packing of the result
into the model's catalog
"""
import numpy as np
import ngmix
import pytest

DIM = 96
SCALE = 0.2
CEN = (DIM - 1) / 2
PSF_FWHM = 0.8
NOISE_SIGMA = 0.05
BANDS = ['r', 'i']

# offsets from the image center in pixels, T in arcsec^2, flux
BLOBS = [
    (0.0, 0.0, 0.5, 80.0),
    (12.0, 8.0, 0.3, 50.0),
    (-20.0, 15.0, 0.4, 60.0),
]


def make_band_obs(rng, band):
    from ngmix.moments import fwhm_to_T

    jacobian = ngmix.DiagonalJacobian(scale=SCALE, row=CEN, col=CEN)
    psf_T = fwhm_to_T(PSF_FWHM)
    psf_gm = ngmix.GMixModel(
        pars=[0.0, 0.0, 0.0, 0.0, psf_T, 1.0], model='gauss',
    )
    psf_im = psf_gm.make_image((DIM, DIM), jacobian=jacobian)
    psf_obs = ngmix.Observation(
        psf_im, weight=np.ones((DIM, DIM)) * 1.0e12, jacobian=jacobian,
    )

    im = np.zeros((DIM, DIM))
    for drow, dcol, T, flux in BLOBS:
        gm0 = ngmix.GMixModel(
            pars=[drow * SCALE, dcol * SCALE, 0.1, -0.05, T, flux],
            model='exp',
        )
        im += gm0.convolve(psf_gm).make_image((DIM, DIM), jacobian=jacobian)

    im += rng.normal(scale=NOISE_SIGMA, size=im.shape)
    noise = rng.normal(scale=NOISE_SIGMA, size=im.shape)
    obs = ngmix.Observation(
        im,
        weight=np.ones((DIM, DIM)) / NOISE_SIGMA ** 2,
        bmask=np.zeros((DIM, DIM), dtype='i4'),
        noise=noise,
        jacobian=jacobian,
        psf=psf_obs,
    )
    obs.meta['band'] = band
    obs.meta['good_frac'] = 1.0
    return obs


def make_mbobs(rng):
    mbobs = ngmix.MultiBandObsList()
    for band in BANDS:
        obslist = ngmix.ObsList()
        obslist.append(make_band_obs(rng, band))
        mbobs.append(obslist)
    return mbobs


@pytest.mark.parametrize('model', ['exp', 'ladder'])
def test_fit_deblend_end_to_end(model):
    pytest.importorskip('kdeblend')
    from lsst_mdet.coadd import coadd_mbobs
    from lsst_mdet.detect import run_sep
    from lsst_mdet.deblend import fit_deblend
    from lsst_mdet.structs import get_struct

    rng = np.random.RandomState(7)
    mbobs = make_mbobs(rng)
    detect_obs, _ = coadd_mbobs(mbobs)
    sxcat, seg = run_sep(detect_obs)
    assert sxcat.size == len(BLOBS)

    cat = get_struct(bands=BANDS, n=sxcat.size, model=model)
    cat['xcell'] = sxcat['x']
    cat['ycell'] = sxcat['y']
    fit_deblend(
        mbobs=mbobs, sxcat=sxcat, cat=cat, seg=seg, model=model, rng=rng,
    )

    assert np.all(cat['flags'] == 0)
    assert np.all(cat['deblend_flags'] == 0)
    assert np.all(np.isfinite(cat['fwhm_smooth']))
    assert np.all(cat['fwhm_smooth'] == cat['fwhm_smooth'][0])
    assert np.all(cat['g_flags'] == 0)

    for band in BANDS:
        assert np.all(np.isfinite(cat[f'flux_{band}']))
        assert np.all(cat[f'flux_err_{band}'] > 0)
    assert np.all(np.isfinite(cat['rmi']))
    assert np.all(cat['rmi_err'] > 0)
    assert np.all(cat['s2n'] > 10)

    # the truth is a flat color, recovered to a few percent
    flux_true = np.array([b[3] for b in BLOBS])
    order = np.argsort(sxcat['x'])
    true_order = np.argsort([CEN + b[1] for b in BLOBS])
    ratio = cat['flux_i'][order] / flux_true[true_order]
    assert np.all(np.abs(ratio - 1) < 0.15)
    assert np.all(np.abs(cat['rmi']) < 0.1)

    if model == 'ladder':
        for band in BANDS:
            assert np.all(np.isfinite(cat[f'gauss_flux_{band}']))
            assert np.all(cat[f'gauss_flux_err_{band}'] > 0)
        assert np.all(np.isfinite(cat['gradient_rmi']))
        assert np.all(cat['gradient_rmi_err'] > 0)
        # the total completes the aperture flux
        assert np.all(cat['flux_i'] > cat['gauss_flux_i'])
    else:
        assert 'gauss_flux_i' not in cat.dtype.names
