"""
metacal image and noise processing
"""
from .detect import get_detect_noise, make_kernel

# the metacal settings, recorded in the output meta table (see
# provenance.py): the sheared types run on the image, the type the
# noise field is run with (the fusion filter takes the max over
# the requested types; 1p reproduces the sheared-pair envelope),
# and the names of the metacal target psf and noise filter classes.
# target_psf can also be a fixed round gaussian 'Gauss:<fwhm>',
# see get_target_psf
METACAL_SETTINGS = dict(
    types='noshear,1p,1m',
    noise_types='1p',
    target_psf='AZGauss',
    noise_filter='FusionFilter',
)


def get_target_psf():
    """
    the target reconvolution psf from the settings: an adaptive
    class from the metacal package (e.g. AZGauss), or a fixed
    round gaussian 'Gauss:<fwhm>' with the fwhm in arcsec, e.g.
    'Gauss:1.3'.  The fixed gaussian is a galsim.GSObject, which
    metacal uses as-is apart from the standard dilation, so every
    cell and band is reconvolved to the same psf; it must be
    chosen larger than any input psf
    """
    import metacal

    spec = METACAL_SETTINGS['target_psf']
    if spec.startswith('Gauss:'):
        import galsim
        return galsim.Gaussian(fwhm=float(spec.split(':', 1)[1]))
    return getattr(metacal, spec)()


def do_all_metacal(mbobs, rng):
    import ngmix

    band_dicts = []
    for iband, obslist in enumerate(mbobs):
        band_dict = do_metacal_one_band(
            obs=obslist[0],
            rng=rng,
        )
        band_dicts.append(band_dict)

    odict = {}
    for key in band_dicts[0]:
        out = ngmix.MultiBandObsList()
        for bdict in band_dicts:
            obslist = ngmix.ObsList()
            obslist.append(bdict[key])
            out.append(obslist)
        odict[key] = out
    return odict


def do_metacal_one_band(obs, rng):
    """
    metacal one band: run the image and its noise field through the
    same operations, attach the metacal'd noise and calibrate the
    weight map; see do_metacal

    Parameters
    ----------
    obs: ngmix.Observation
        The observation holding image, psf image, noise field etc.
    rng: np.random.RandomState
        The random number generator

    Returns
    -------
    dict keyed by metacal type, each holding an Observation
    """
    import numpy as np

    noise_types = METACAL_SETTINGS['noise_types'].split(',')
    odict = run_metacal(
        obs=obs,
        rng=rng,
        types=METACAL_SETTINGS['types'].split(','),
    )
    ndict = run_metacal(
        obs=make_noise_obs(obs),
        rng=rng,
        types=noise_types,
    )

    mcal_noise = ndict[noise_types[0]].image

    sigma_band = get_detect_noise(
        noise=mcal_noise,
        kernel=make_kernel(),
        weight=obs.weight,
    )

    for key, mobs in odict.items():
        mobs.noise = mcal_noise

        # rescale so the median weight is 1/sigma_band^2, preserving
        # zero-weight pixels and any relative depth structure
        medwt = np.median(mobs.weight[mobs.weight > 0])
        mobs.weight = mobs.weight / (sigma_band**2 * medwt)

    return odict


def make_noise_obs(obs):
    """
    An observation whose image is the noise field, used to propagate a
    noise realization through the same metacal operations as the image.

    Its own correction field is the 90 degree rotation of the same
    realization: for stationary noise the rotation maps fourier modes to
    distinct modes, so the two are uncorrelated and the output has the
    same marginal covariance as the noise in the metacal'd image

    Parameters
    ----------
    obs: ngmix.Observation
        The observation, with the .noise field set

    Returns
    -------
    ngmix.Observation
    """
    import numpy as np

    noise_obs = obs.copy()
    noise_obs.image = obs.noise.copy()
    noise_obs.noise = np.ascontiguousarray(np.rot90(obs.noise))
    return noise_obs


def run_metacal(obs, rng, types):
    """
    Run metacal with the configured noise correction method

    Parameters
    ----------
    obs: ngmix.Observation
        The observation holding image, psf image, noise field etc.
    rng: np.random.RandomState
        The random number generator
    types: sequence of str
        The metacal types to process

    Returns
    -------
    dict keyed by metacal type
    """
    import metacal

    target_psf = get_target_psf()
    noise_filter = getattr(metacal, METACAL_SETTINGS['noise_filter'])()
    odict = metacal.metacal_obs(
        obs=obs,
        target_psf=target_psf,
        noise_filter=noise_filter,
        rng=rng,
        types=types,
    )

    return odict
