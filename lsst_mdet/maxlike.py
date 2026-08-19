"""
maximum-likelihood-only object measurement
"""
from .colors_and_fluxes import set_colors, set_fluxes
from .defaults import BAD_BBOX, ZERO_WEIGHTS
from .util import get_stamp


def do_single_fits(mbobs, sxcat, cat, model, rng):
    """
    Fit each object independently from its own postage stamp

    Parameters
    ----------
    obs: ngmix.Observation
        The observation to fit, the detection coadd from
        do_detection_and_fits
    sxcat: array with fields
        The sep catalog
    rng: np.random.RandomState
        The random number generator for the object fitting

    Returns
    -------
    cat, keep

    cat: array with fields
        One row per kept detection
    keep: bool array
        Which rows of sxcat were kept; stamps hitting an edge are
        dropped
    """
    from ngmix import GMixFatalError

    for i in range(sxcat.size):

        try:
            stamp_mbobs = extract_stamp_mbobs(
                mbobs=mbobs,
                icat=sxcat[i],
            )
            # # flags are explicitly set
            fit_ml(
                st=cat[i],
                model=model,
                rng=rng,
                mbobs=stamp_mbobs,
            )

        except IndexError as err:
            cat['flags'][i] = BAD_BBOX
            print(f'stamp for obj {i} hit edge: {err}')
        except GMixFatalError as err:  # noqa
            cat['flags'][i] = ZERO_WEIGHTS


def fit_ml(st, model, rng, mbobs):
    """
    Fit a model to the observation using maximum likelihood.  The PSF is fit
    first, then the image is fit using a model convolved by the PSF.  Thus the
    fit is "pre-PSF".

    Parameters
    ----------
    rng: random number generator
        E.g. np.random.RandomState
    obs: ngmix.Observation
        The observation data

    Returns
    -------
    array: np.ndarray
        Array with fit data. See get_struct
    """

    # we need the stamps to have copied the gmixes
    assert all([obslist[0].psf.has_gmix() for obslist in mbobs])

    bands = [obslist[0].meta['band'] for obslist in mbobs]

    runner = _get_ml_runner(
        rng=rng,
        model=model,
        scale=mbobs[0][0].jacobian.scale,
        bands=bands,
    )

    res = runner.go(mbobs)

    st['flags'] = res['flags']
    st['numiter'] = res['nfev']
    st['group_size'] = 1

    # for these single-object fitters the shape is usable iff the
    # fit succeeded, and no deblending is involved
    st['deblend_flags'] = 0
    st['group_size'] = 1

    if res['flags'] == 0:
        st['g1'] = res['g'][0]
        st['g1_err'] = res['g_err'][0]
        st['g2'] = res['g'][1]
        st['g2_err'] = res['g_err'][1]
        st['T'] = res['T']
        st['T_err'] = res['T_err']
        set_fluxes(
            st=st,
            bands=bands,
            flux=res['flux'],
            flux_err=res['flux_err'],
        )
        set_colors(
            st=st,
            bands=bands,
            flux=res['flux'],
            flux_err=res['flux_err'],
            flux_cov=res['flux_cov'],
        )

        st['s2n'] = res['s2n']


FIT_PARS = {
    "maxfev": 2000,
    "xtol": 1.0e-5,
    "ftol": 1.0e-5,
}


def extract_stamp_mbobs(mbobs, icat, stamp_size=49):
    import ngmix

    stamp_mbobs = ngmix.MultiBandObsList()

    for iband, obslist in enumerate(mbobs):
        stamp_obs = extract_stamp_obs(
            obs=obslist[0],
            icat=icat,
            stamp_size=stamp_size,
        )

        stamp_obslist = ngmix.ObsList()
        stamp_obslist.append(stamp_obs)
        stamp_mbobs.append(stamp_obslist)

    return stamp_mbobs


def extract_stamp_obs(obs, icat, stamp_size=49):
    import ngmix

    stamp, xstart, ystart = get_stamp(
        image=obs.image,
        x=icat['x'],
        y=icat['y'],
        stamp_size=stamp_size,
    )
    noise_stamp, _, _ = get_stamp(
        image=obs.noise,
        x=icat['x'],
        y=icat['y'],
        stamp_size=stamp_size,
    )
    mfrac_stamp, _, _ = get_stamp(
        image=obs.mfrac,
        x=icat['x'],
        y=icat['y'],
        stamp_size=stamp_size,
    )
    weight, _, _ = get_stamp(
        image=obs.weight,
        x=icat['x'],
        y=icat['y'],
        stamp_size=stamp_size,
    )

    jacobian = obs.jacobian.copy()
    jacobian.set_cen(
        x=icat['x'] - xstart,
        y=icat['y'] - ystart,
    )
    stamp_obs = ngmix.Observation(
        image=stamp,
        weight=weight,
        noise=noise_stamp,
        mfrac=mfrac_stamp,
        jacobian=jacobian,
        psf=obs.psf.copy(),
        meta=obs.meta,
    )
    return stamp_obs


def _get_ml_runner(rng, scale, model, bands):
    """
    Get an ngmix runner for the model fit
    """
    import ngmix

    prior = _get_fit_prior(
        rng=rng,
        scale=scale,
        nband=len(bands),
    )
    fitter = ngmix.fitting.Fitter(
        model=model,
        prior=prior,
        use_noise_image=True,
        fit_pars=FIT_PARS.copy(),
    )

    guesser = ngmix.guessers.TPSFFluxGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
    )

    runner = ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    return runner


def _get_fit_prior(rng, scale, T_range=None, F_range=None, nband=None):
    """
    get a prior for use with the maximum likelihood fitter

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    scale: float
        Pixel scale
    T_range: (float, float), optional
        The range for the prior on T
    F_range: (float, float), optional
        Fhe range for the prior on flux
    nband: int, optional
        number of bands
    """
    import ngmix

    if T_range is None:
        T_range = [-1.0, 1.e3]
    if F_range is None:
        F_range = [-100.0, 1.e9]

    g_prior = ngmix.priors.GPriorBA(sigma=0.4, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0, cen2=0, sigma1=scale, sigma2=scale, rng=rng,
    )
    T_prior = ngmix.priors.FlatPrior(
        minval=T_range[0], maxval=T_range[1], rng=rng,
    )
    F_prior = ngmix.priors.FlatPrior(
        minval=F_range[0], maxval=F_range[1], rng=rng,
    )

    if nband is not None:
        F_prior = [F_prior] * nband

    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )

    return prior
