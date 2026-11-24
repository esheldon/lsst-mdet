import numpy as np
import pytest


def _make_obs(rng, dim=64, scale=0.2, sigma=1.0):
    import galsim
    import ngmix

    psf = galsim.Gaussian(fwhm=0.8)
    psf_dim = 33
    psf_im = psf.drawImage(nx=psf_dim, ny=psf_dim, scale=scale).array
    pcen = (psf_dim - 1) / 2
    psf_obs = ngmix.Observation(
        psf_im, weight=np.full(psf_im.shape, 1.0e12),
        jacobian=ngmix.DiagonalJacobian(scale=scale, row=pcen, col=pcen),
    )
    gal = galsim.Convolve(
        galsim.Exponential(half_light_radius=0.5, flux=500.0), psf,
    )
    im = gal.drawImage(nx=dim, ny=dim, scale=scale).array
    cen = (dim - 1) / 2
    return ngmix.Observation(
        im + rng.normal(scale=sigma, size=im.shape),
        weight=np.full(im.shape, 1.0 / sigma ** 2),
        noise=rng.normal(scale=sigma, size=im.shape),
        jacobian=ngmix.DiagonalJacobian(scale=scale, row=cen, col=cen),
        psf=psf_obs,
    )


def test_metacal_one_band():
    """
    metacal runs with the configured noise kernel: every type comes
    back with the metacal'd noise field attached and a finite,
    calibrated weight map
    """
    from lsst_mdet.metacal import METACAL_SETTINGS, do_metacal_one_band

    rng = np.random.RandomState(8)
    obs = _make_obs(rng)
    odict = do_metacal_one_band(obs=obs, rng=rng)

    assert list(odict) == METACAL_SETTINGS['types'].split(',')
    noise = odict['noshear'].noise
    for mobs in odict.values():
        assert mobs.image.shape == obs.image.shape
        assert np.all(np.isfinite(mobs.image))
        assert np.array_equal(mobs.noise, noise)
        assert np.all(np.isfinite(mobs.weight)) and np.all(mobs.weight > 0)
    # the reconvolution correlates and smooths the noise
    assert 0 < noise.std() < obs.noise.std()


def test_noise_kernel_setting(monkeypatch):
    """
    the default kernel is the white noise one, and a kernel that
    cannot be made from its name is refused
    """
    import metacal
    from lsst_mdet.metacal import METACAL_SETTINGS, get_noise_kernel

    assert METACAL_SETTINGS['noise_kernel'] == 'DeltaImpulse'
    assert isinstance(get_noise_kernel(), metacal.DeltaImpulse)

    monkeypatch.setitem(METACAL_SETTINGS, 'noise_kernel', 'ImpulseResponse')
    with pytest.raises(ValueError):
        get_noise_kernel()
