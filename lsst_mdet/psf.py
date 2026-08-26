"""
psf fitting: original psfs and the mdet psfs
"""


def fit_and_set_mcal_psfs(mbobs, weights, rng):
    import numpy as np
    import ngmix

    psf_runner = ngmix.runners.PSFRunner(
        fitter=ngmix.admom.AdmomFitter(rng=rng),
        guesser=ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=1),
        set_result=True,
        ntry=4,
    )

    res = {'psf_flags': 0}
    wsum = 0.0
    Tsum = 0.0
    g1sum = 0.0
    g2sum = 0.0
    for obslist, weight in zip(mbobs, weights):
        obs = obslist[0]
        band = obs.meta['band']
        psf_obs = obs.psf

        # this will set the gmix
        assert not psf_obs.has_gmix()
        psf_res = psf_runner.go(psf_obs)
        assert psf_obs.has_gmix()

        res[f'psf_flags_{band}'] = psf_res['flags']
        res['psf_flags'] |= psf_res['flags']

        if psf_res['flags'] == 0:
            g1, g2, T = obs.psf.gmix.get_g1g2T()
            res[f'psf_T_{band}'] = T

            wsum += weight
            Tsum += weight * T
            g1sum += weight * g1
            g2sum += weight * g2
        else:
            res[f'psf_T_{band}'] = np.nan

    if res['psf_flags'] == 0:
        res['psf_T'] = Tsum / wsum
    else:
        res['psf_T'] = np.nan

    return res


def _set_mcal_psfs(st, psf_res):
    for key in psf_res:
        st[key] = psf_res[key]


def fit_and_set_psfrec(st, mbobs, rng):
    """
    Fit original psfs and set values in st.  st can have more than
    one row, setting the value for all objects in the field
    """
    import numpy as np
    import ngmix

    psf_runner = ngmix.runners.PSFRunner(
        fitter=ngmix.admom.AdmomFitter(rng=rng),
        guesser=ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=1),
        # we don't want to pass on any gmixes
        set_result=False,
        ntry=4,
    )

    flags = 0
    wsum = 0.0
    Tsum = 0.0
    g1sum = 0.0
    g2sum = 0.0
    for obslist in mbobs:
        obs = obslist[0]
        band = obs.meta['band']
        psf_obs = obs.psf

        psf_res = psf_runner.go(psf_obs)

        st[f'psfrec_flags_{band}'] = psf_res['flags']
        flags |= psf_res['flags']

        if psf_res['flags'] == 0:
            gmix = psf_res.get_gmix()
            g1, g2, T = gmix.get_g1g2T()
            st[f'psfrec_T_{band}'] = T
            st[f'psfrec_g1_{band}'] = g1
            st[f'psfrec_g2_{band}'] = g2

            weight = np.median(
                obs.weight[obs.weight > 0],
            )
            wsum += weight
            Tsum += weight * T
            g1sum += weight * g1
            g2sum += weight * g2

    if flags == 0:
        st['psfrec_T'] = Tsum / wsum
        st['psfrec_g1'] = g1sum / wsum
        st['psfrec_g2'] = g2sum / wsum

    st['psfrec_flags'] = flags
