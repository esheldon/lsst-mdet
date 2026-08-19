"""
detection coadd over bands
"""


def coadd_mbobs(mbobs):
    """
    Coadd the bands from a ngmix.MultiBandObsList

    The noise fields are coadded with the same weights as the
    images, so the output noise is a realization of the noise in
    the coadd image; either every band must have a noise field or
    none.  The summed weight map is the correct coadd weight on
    the scales where the per-band weights are calibrated, because
    the image weights are the per-band median weights

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The input data

    Returns
    -------
    ngmix.Observation, list[weights]
    """
    import numpy as np

    nband = len(mbobs)

    if nband == 1:
        return mbobs[0][0]

    assert np.all([obslist[0].has_noise() for obslist in mbobs])

    mask = mbobs[0][0].bmask.copy()

    weights = np.zeros(nband)
    good_fracs = np.zeros(nband)
    wsum = 0.0
    for iband, obslist in enumerate(mbobs):
        assert len(obslist) == 1
        obs = obslist[0]
        mask |= obs.bmask

        medwt = np.median(
            obs.weight[obs.weight > 0],
        )

        if iband == 0:
            im = obs.image * 0
            wt = obs.weight * 0
            psf_im = obs.psf.image * 0
            noise = obs.noise * 0

        wsum += medwt
        im += obs.image * medwt
        noise += obs.noise * medwt
        psf_im += obs.psf.image * medwt
        wt += obs.weight

        weights[iband] = medwt
        good_fracs[iband] = obs.meta['good_frac']

    coadd_obs = mbobs[0][0].copy()
    del coadd_obs.meta['good_frac']
    coadd_obs.meta['good_fracs'] = good_fracs

    iwsum = 1.0 / wsum
    with coadd_obs.writeable():
        coadd_obs.image = im * iwsum
        coadd_obs.weight = wt
        coadd_obs.noise = noise * iwsum

    with coadd_obs.psf.writeable():
        coadd_obs.psf.image = psf_im * iwsum

    coadd_obs.mask = mask
    return coadd_obs, weights
