"""
metacal image and noise processing
"""
from .detect import get_detect_noise, make_kernel


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

    odict = run_metacal(
        obs=obs,
        rng=rng,
        types=['noshear', '1p', '1m'],
    )
    ndict = run_metacal(
        obs=make_noise_obs(obs),
        rng=rng,
        types=['1p'],
    )

    mcal_noise = ndict['1p'].image

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

    odict = metacal.metacal_obs(
        obs=obs,
        target_psf=metacal.AZGauss(),
        noise_filter=metacal.FusionFilter(),
        rng=rng,
        types=types,
    )

    return odict
