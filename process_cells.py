"""
TODO

- mfrac
- star mask
- add deblending with special flags, gauss_, fam_
- decide MIN_GOOD_FRAC
- decide if keeping non primary
- detection kernel (fixed or variable)
- Tfrac (requires upstream)

"""
import os
import numpy as np
from lsst.daf.butler import Butler
import rustfits
from lsst.images._geom import BoundsError
from lsst.images import Box
from lsst.images._cell_grid import CellIJ
import lsst.geom
from numba import njit


TRIM_TO_PRIMARY = True

# flags set in the input mask plans
DM_NO_DATA = 1
DM_DETECTION_EDGE = 16
DM_OUT = DM_NO_DATA | DM_DETECTION_EDGE

# processing flags
from ngmix.flags import NO_ATTEMPT  # noqa
PSF_FAILURE = 2**21
BAD_BBOX = 2**24
ZERO_WEIGHTS = 2**25

MIN_GOOD_FRAC = 0.2

OVERLAP = 50
OVERLAP_LOW = 50
OVERLAP_HIGH = 200

SKYMAP_VERS = 'lsst_cells_v2'
AP_RAD = 1.5

MFRAC_FWHM = 1.2


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--outfile', required=True)
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--mdet', action='store_true')
    return parser.parse_args()


def run_sep(obs):
    """
    Run sep (python wrapper for sextractor) on the input ngmix Observation

    The noise for the relative threshold is taken from the weight map.
    For metacal'd observations the weight map is calibrated in
    mdet.do_metacal so that 1/sqrt(median(weight)) is the measured noise
    on the detection-kernel scale, the correct calibration when the
    noise is correlated

    Parameters
    ----------
    obs: ngmix.Observation
        The observation holding image and weight map needed for detection

    Returns
    -------
    cat, seg

    cat: array with fields
        The object data.
    seg: array
        The segmentation map, with values matching the catalog
        'number' field
    """
    import numpy as np
    import sxdes

    medwt = np.median(obs.weight)
    noise = 1 / np.sqrt(medwt)

    with obs.writeable():
        objs, seg = sxdes.run_sep(
            image=obs.image,
            noise=noise,
            config=get_sx_config(),
            mask=obs.bmask,
            thresh=0.8,
        )
    return objs, seg


def get_detect_noise(noise, kernel):
    """
    Get the noise value on the detection-kernel scale, used by
    mdet.do_metacal to calibrate the weight maps for correlated noise.

    sep with filter_type 'conv' normalizes the kernel to unit sum and
    thresholds the convolved image at thresh * err.  The std of the
    convolved noise field, divided by sqrt(sum(khat^2)), gives the err
    for which thresh keeps its white-noise meaning; for uncorrelated
    noise this reduces to the pixel sigma

    Parameters
    ----------
    noise: array
        A realization of the noise in the image, e.g. the metacal'd
        noise field for a metacal'd image
    kernel: array
        The detection kernel

    Returns
    -------
    noise value for sxdes.run_sep
    """
    import numpy as np
    from scipy.ndimage import convolve

    khat = kernel / kernel.sum()
    conv = convolve(noise, khat, mode='reflect')
    return conv.std() / np.sqrt((khat**2).sum())


def get_sx_config():
    """
    The configuration for sextractor.  This is currently hard wired
    to use a detection kernel of 0.8 arcseconds assuming pixel scale
    of 0.2 (sensible for LSST)
    """
    kernel = make_kernel()

    return {
        # 1e-5 recovers peak-resolved objects in crowded
        # regions that the DES default 1e-3 merges; measured
        # to add no false detections at any tested density
        # (see docs/detection-color-filter)
        'deblend_cont': 1.0e-5,
        'deblend_nthresh': 64,
        'minarea': 4,
        'filter_type': 'conv',
        'filter_kernel': kernel,
    }


def make_kernel():
    """
    Make a detection kernel fwhm=0.8'' for 0.2'' pixels
    """
    import ngmix

    # 7x7 convolution mask of a gaussian PSF with FWHM = 4 pixels.
    # this is 0.8 arcseconds at 0.2 arcseconds per pixel
    fwhm = 0.8 / 0.2  # pixels
    T = ngmix.moments.fwhm_to_T(fwhm)
    kernel_gm = ngmix.GMixModel(
        pars=[0.0, 0.0, 0.0, 0.0, T, 1.0],
        model='gauss',
    )
    return kernel_gm.make_image([7, 7])


def get_stamp(image, x, y, stamp_size):
    """
    Extract a postage stamp from the input image.   Returned stamps are square.
    If the stamp hits an edge, IndexError is raised.

    Parameters
    ----------
    image: array
        Array from which to extract a stamp
    x: float or int
        x position in the array
    y: float or int
        y position in the array
    stamp_size: int
        The extracted stamp will have size [stamp_size, stamp_size]

    Returns
    -------
    stamp, xstart, ystart

    stamp: array
        Extracted stamp
    xstart: int
        The start x position
    ystart: int
        The start y position
    """

    assert stamp_size % 2 != 0, f'stamp size should be odd, got {stamp_size}'

    imny, imnx = image.shape
    xstart, xend = _get_bound(x, stamp_size, imsize=imnx)
    ystart, yend = _get_bound(y, stamp_size, imsize=imny)

    stamp = image[ystart:yend, xstart:xend]
    if stamp.shape[0] != stamp_size or stamp.shape[1] != stamp_size:
        raise IndexError(
            f'expected shape [{stamp_size}, {stamp_size}] '
            f'but got {stamp.shape}'
        )
    return stamp, xstart, ystart


def _get_bound(x, stamp_size, imsize):
    """
    Get the bound in one dimension for a requested stamp

    Parameters
    ----------
    x: float or int
        x position in the array
    image: array
        Array from which to extract a stamp
    y: float or int
        y position in the array
    stamp_size: int
        The extracted stamp will have size [stamp_size, stamp_size]

    """
    rx = round(x)
    xstart = rx - (stamp_size - 1) // 2
    xend = rx + (stamp_size - 1) // 2 + 1

    if xstart < 0:
        raise IndexError('out of bounds')
    if xend > imsize + 1:
        raise IndexError('out of bounds')

    return xstart, xend


def make_cell_obs(image, var, good, noise, mfrac, psf_image, jacobian):
    import ngmix

    psf_jacobian = jacobian.copy()
    psf_cen = (np.array(psf_image.shape) - 1.0) / 2.0
    psf_jacobian.set_cen(row=psf_cen[0], col=psf_cen[1])

    psf_obs = ngmix.Observation(
        psf_image,
        weight=psf_image * 0 + 1.0 / 1.0e-6 ** 2,
        jacobian=psf_jacobian,
    )

    weight = np.zeros(image.shape)
    weight[good] = 1.0 / var[good]

    bmask = np.zeros(image.shape, dtype='i4')
    bmask[~good] = 1

    return ngmix.Observation(
        image,
        weight=weight,
        bmask=bmask,
        noise=noise,
        mfrac=mfrac,
        jacobian=jacobian,
        psf=psf_obs,
    )


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


def do_single_fits(mbobs, weights, sxcat, rng):
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
    import ngmix
    from ngmix import GMixFatalError

    bands = [obslist[0].meta['band'] for obslist in mbobs]
    cat = get_struct(bands=bands, n=sxcat.size)

    mfrac_weight = ngmix.GMixModel(
        [0, 0, 0, 0, ngmix.moments.fwhm_to_T(MFRAC_FWHM), 1],
        'gauss',
    )

    psf_res = fit_and_set_mcal_psfs(
        mbobs=mbobs, weights=weights, rng=rng,
    )

    if psf_res['psf_flags'] == 0:

        for i in range(sxcat.size):

            try:
                stamp_mbobs = extract_stamp_mbobs(
                    mbobs=mbobs,
                    icat=sxcat[i],
                )
                # # flags are explicitly set
                fit_struct = fit_gauss(
                    rng=rng,
                    mbobs=stamp_mbobs,
                )

                cat[i] = fit_struct

                cat['mfrac'][i] = calculate_mfrac(
                    mbobs=stamp_mbobs,
                    mfrac_weight=mfrac_weight,
                )

            except IndexError as err:
                cat['flags'][i] = BAD_BBOX
                print(f'stamp for obj {i} hit edge: {err}')
            except GMixFatalError as err:
                cat['flags'][i] = ZERO_WEIGHTS
                print(f'obj {i}: {err}')

    else:
        cat['flags'] = PSF_FAILURE

    # set after loop because above we explictly overwrite each cat[i]
    _set_mcal_psfs(st=cat, psf_res=psf_res)

    return cat


def calculate_mfrac(mbobs, mfrac_weight):

    mfrac = mbobs[0][0].mfrac.copy()

    for iband, obslist in enumerate(mbobs):
        if iband == 0:
            mfrac = obslist[0].mfrac.copy()
        else:
            mfrac[:, :] = np.maximum(mfrac, obslist[0].mfrac)

    mfrac_obs = mbobs[0][0].copy()

    with mfrac_obs.writeable():
        mfrac_obs.image[:, :] = mfrac
        mfrac_obs.weight[:, :] = 1.0

    stats = mfrac_weight.get_weighted_sums(mfrac_obs, MFRAC_FWHM * 2)
    return stats["sums"][5] / stats["wsum"]


def fit_gauss(rng, mbobs):
    """
    Fit a Gaussian to the observation using maximum likelihood.  The PSF is fit
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

    runner = _get_gauss_runner(
        rng=rng,
        scale=mbobs[0][0].jacobian.scale,
        bands=bands,
    )

    res = runner.go(mbobs)

    st = get_struct(bands=bands)

    st['flags'] = res['flags']
    # for these single-object fitters the shape is usable iff the
    # fit succeeded, and no deblending is involved
    st['g_flags'] = res['flags']
    st['deblend_flags'] = 0

    if res['flags'] == 0:
        st['g1'] = res['g'][0]
        st['g1_err'] = res['g_err'][0]
        st['g2'] = res['g'][1]
        st['g2_err'] = res['g_err'][1]
        st['T'] = res['T']
        st['T_err'] = res['T_err']
        _set_fluxes(
            st=st,
            bands=bands,
            flux=res['flux'],
            flux_err=res['flux_err'],
        )
        _set_colors(
            st=st,
            bands=bands,
            flux=res['flux'],
            fcov=res['flux_cov'],
        )

        st['s2n'] = res['s2n']

    return st


def _set_fluxes(st, bands, flux, flux_err):
    """
    fill the flux_{band} and flux_err_{band} columns; the fitters
    return a scalar for a single band and an array for multiple
    """
    import numpy as np

    flux = np.atleast_1d(flux)
    flux_err = np.atleast_1d(flux_err)
    if flux.size != len(bands):
        raise ValueError(
            f'got {flux.size} fluxes for {len(bands)} bands'
        )
    for iband, band in enumerate(bands):
        st[f'flux_{band}'] = flux[iband]
        st[f'flux_err_{band}'] = flux_err[iband]


def _set_colors(st, bands, flux, fcov):
    """
    Set colors and errors based on the full covariance
    """
    fac = 2.5 / np.log(10)
    eps = 1.0e-7

    nband = len(bands)

    for i in range(nband - 1):
        first_band = bands[i]
        second_band = bands[i + 1]
        cname = f'{first_band}m{second_band}'

        if flux[i] > eps and flux[i + 1] > eps:
            color = -2.5 * np.log10(flux[i] / flux[i + 1])

            color_var = fac**2 * (
                fcov[i, i] / flux[i] ** 2
                + fcov[i + 1, i + 1] / flux[i + 1] ** 2
                - 2 * fcov[i, i + 1] / (flux[i] * flux[i + 1])
            )

            st[cname] = color
            st[f'{cname}_err'] = np.sqrt(color_var)


FIT_PARS = {
    "maxfev": 2000,
    "xtol": 1.0e-5,
    "ftol": 1.0e-5,
}


def _get_gauss_runner(rng, scale, bands):
    """
    Get an ngmix runner for the gaussian fit
    """
    import ngmix

    prior = _get_fit_prior(
        rng=rng,
        scale=scale,
        nband=len(bands),
    )
    fitter = ngmix.fitting.Fitter(
        model='gauss',
        prior=prior,
        use_noise_image=True,
        fit_pars=FIT_PARS.copy(),
    )

    # guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
    #     rng=rng,
    #     T=0.25,
    #     prior=prior,
    # )
    guesser = ngmix.guessers.TPSFFluxGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
    )

    # psf_prior = _get_fit_prior(
    #     rng=rng,
    #     scale=scale,
    # )

    # psf fitting with coelliptical gaussians
    # psf_fitter = ngmix.fitting.Fitter(
    #     model='gauss',
    #     prior=psf_prior,
    #     fit_pars=FIT_PARS.copy(),
    # )
    #
    # psf_guesser = ngmix.guessers.SimplePSFGuesser(
    #     rng=rng,
    #     guess_from_moms=True,
    # )
    #
    # # this runs the fitter. We set ntry=2 to retry the fit if it fails
    # psf_runner = ngmix.runners.PSFRunner(
    #     fitter=psf_fitter, guesser=psf_guesser,
    #     ntry=2,
    # )
    runner = ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    return runner

    # this bootstraps the process, first fitting psfs then the object
    # boot = ngmix.bootstrap.Bootstrapper(
    #     runner=runner,
    #     # psf_runner=psf_runner,
    #     # ignore_failed_psf=False,
    # )
    # return boot


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
    # T_prior = ngmix.priors.TwoSidedErf(
    #     minval=T_range[0],
    #     width_at_min=abs(T_range[0]) * 0.003,
    #     maxval=T_range[1],
    #     width_at_max=abs(T_range[1]) * 0.1,
    #     rng=rng,
    # )
    # F_prior = ngmix.priors.TwoSidedErf(
    #     minval=F_range[0],
    #     width_at_min=abs(F_range[0]) * 0.01,
    #     maxval=F_range[1],
    #     width_at_max=abs(F_range[1]) * 0.1,
    #     rng=rng,
    # )

    if nband is not None:
        F_prior = [F_prior] * nband

    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )

    return prior


def get_struct(bands, n=1):
    """
    Get the output structure for the single-object fitters (am,
    gauss, wmom).  The kdeblend fitter uses get_kdeblend_struct

    Parameters
    ----------
    bands: sequence of str
        The band names; flux_{band} and flux_err_{band} columns are
        created for each
    n: int, optional
        The number of rows, default 1
    """
    dtype = [
        ('mcal_step', 'U2'),
        ('cell_i', 'i2'),
        ('cell_j', 'i2'),
        ('flags', 'i4'),
        ('is_primary', bool),
        ('deblend_flags', 'i4'),
        ('mfrac', 'f4'),

        ('xcell', 'f4'),
        ('ycell', 'f4'),
        ('x', 'f4'),
        ('y', 'f4'),
        ('ra', 'f8'),
        ('dec', 'f8'),

        ('s2n', 'f4'),
        ('g_flags', 'i4'),
        ('g1', 'f4'),
        ('g1_err', 'f4'),
        ('g2', 'f4'),
        ('g2_err', 'f4'),
        ('T', 'f4'),
        ('T_err', 'f4'),
    ]

    for band in bands:
        dtype += [
            (f'flux_{band}', 'f4'),
            (f'flux_err_{band}', 'f4'),
        ]

    nband = len(bands)
    for i in range(nband - 1):
        first_band = bands[i]
        second_band = bands[i + 1]

        cname = f'{first_band}m{second_band}'

        dtype += [
            (cname, 'f4'),
            (f'{cname}_err', 'f4'),
        ]

    for band in bands:
        dtype += [
            (f'psf_flags_{band}', 'i4'),
            (f'psf_T_{band}', 'f4'),
        ]

    dtype += [
        ('psf_flags', 'i4'),
        ('psf_T', 'f4'),
    ]

    for band in bands:
        dtype += [
            (f'psfrec_flags_{band}', 'i4'),
            (f'psfrec_T_{band}', 'f4'),
            (f'psfrec_g1_{band}', 'f4'),
            (f'psfrec_g2_{band}', 'f4'),
        ]

    dtype += [
        ('psfrec_flags', 'i4'),
        ('psfrec_T', 'f4'),
        ('psfrec_g1', 'f4'),
        ('psfrec_g2', 'f4'),
    ]

    # dtype += [
    #     ('psfrec_Tfrac', 'f4'),
    #     ('psfrec_e1diff', 'f4'),
    #     ('psfrec_e2diff', 'f4'),
    # ]

    return _init_struct(dtype=dtype, n=n)


def _init_struct(dtype, n):
    """
    build the array, initializing flags to 1, color_det to 0 and
    everything else to nan
    """
    import numpy as np

    output = np.zeros(n, dtype=dtype)

    for name in output.dtype.names:
        if 'flags' in name:
            output[name] = NO_ATTEMPT
        elif name in (
            'color_det', 'numiter', 'group_size', 'cell_i', 'cell_j',
        ):
            output[name] = 0
        else:
            output[name] = np.nan

    return output


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


def pull_mbobs(deep_coadds, cell_i, cell_j, wcs):
    import ngmix

    nband = len(deep_coadds)
    cell_info = get_cell_info(nband)
    cell_info['cell_i'] = cell_i
    cell_info['cell_j'] = cell_j

    mbobs = ngmix.MultiBandObsList()

    for iband, deep_coadd in enumerate(deep_coadds):
        bbox0 = deep_coadd.grid.bbox_of(CellIJ(cell_i, cell_j))

        bbox = Box.factory[
            bbox0.y.start - OVERLAP: bbox0.y.stop + OVERLAP,
            bbox0.x.start - OVERLAP: bbox0.x.stop + OVERLAP,
        ]

        var = deep_coadd.variance[bbox].array.copy()
        mask = deep_coadd.mask[bbox].array[:, :, 0]

        good = np.isfinite(var) & (mask & DM_OUT == 0)

        w = np.where(good)
        good_frac = w[0].size / var.size
        cell_info['good_frac'][0, iband] = good_frac

        if good_frac > MIN_GOOD_FRAC:

            noise = deep_coadd.noise_realizations[0][bbox].array.copy()
            mfrac = deep_coadd.mask_fractions["rejected"][bbox].array.copy()
            image = deep_coadd.image[bbox].array.copy()

            psf = deep_coadd.psf

            xmid = 0.5 * (bbox.x.start + bbox.x.stop)
            ymid = 0.5 * (bbox.y.start + bbox.y.stop)

            try:
                psf_image = psf.compute_kernel_image(
                    x=xmid,
                    y=ymid,
                ).array
            except BoundsError as err:
                print(err)
                break

            cell_jacobian = get_cell_jacobian(
                wcs=wcs, bbox=bbox, x=xmid, y=ymid,
            )
            obs = make_cell_obs(
                image=image,
                var=var,
                good=good,
                noise=noise,
                mfrac=mfrac,
                psf_image=psf_image,
                jacobian=cell_jacobian,
            )
            obs.meta['good_frac'] = good_frac
            obs.meta['bbox'] = bbox
            obs.meta['band'] = deep_coadd.band

            noise_bad = np.isnan(noise)
            wnoise_bad = np.where(noise_bad)
            assert np.all(np.isnan(noise[wnoise_bad]))
            # assert np.all(np.isnan(image[wnoise_bad]))
            if wnoise_bad[0].size > 0:
                print(f'cell_i: {cell_i} cell_j: {cell_j} iband: {iband}')
                print(
                    f'  found {wnoise_bad[0].size} / {noise.size} nan in noise'
                )
                if False:
                    png = 'bad.jpg'
                    import matplotlib.pyplot as mplt
                    with mplt.style.context('dark_background'):
                        fig, axs = mplt.subplots(
                            ncols=2, nrows=2, figsize=(10, 10)
                        )
                        axs[0, 0].set_title('image')
                        axs[0, 0].imshow(
                            np.log10(image.clip(min=0.001)), cmap='gray'
                        )
                        axs[0, 1].set_title('noise')
                        axs[0, 1].imshow(noise, cmap='gray')
                        axs[1, 0].set_title('mask')
                        axs[1, 0].imshow(mask, cmap='gray')
                        axs[1, 1].set_title('var')
                        axs[1, 1].imshow(var, cmap='gray')
                        fig.savefig(png)

                    import IPython
                    IPython.embed()
                break

            sigma_band = get_detect_noise(
                noise=noise,
                kernel=make_kernel(),
            )
            medwt = np.median(obs.weight[obs.weight > 0])
            # print(f'    median weight: {medwt:g} sigma_band: {sigma_band:g}')
            # if not np.isfinite(sigma_band):
            #     import IPython; IPython.embed()
            obs.weight = obs.weight / (sigma_band ** 2 * medwt)

            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)
        else:
            # print(f'    good frac {good_frac} < {MIN_GOOD_FRAC}')
            break

    if len(mbobs) < len(deep_coadds):
        return None, cell_info
    else:
        cell_info['kept'] = True
        return mbobs, cell_info


def get_cell_jacobian(wcs, bbox, x, y):
    import ngmix

    dm_jac = wcs.linearizePixelToSky(
        lsst.geom.Point2D(x, y),
        lsst.geom.arcseconds,
    )
    matrix = dm_jac.getLinear().getMatrix()

    # ESS reverse engineered this convention mismatch.  No documentation
    # was found.  Don't change this unless you know what you are doing.
    return ngmix.Jacobian(
        x=x - bbox.x.start,
        y=y - bbox.y.start,
        dudx=matrix[1, 1],
        dudy=-matrix[1, 0],
        dvdx=matrix[0, 1],
        dvdy=-matrix[0, 0],
    )


def get_primary(sxcat):
    return (
        (sxcat['x'] > OVERLAP_LOW)
        & (sxcat['x'] < OVERLAP_HIGH)
        & (sxcat['y'] > OVERLAP_LOW)
        & (sxcat['y'] < OVERLAP_HIGH)
    )


def apodize_mbobs(mbobs):
    """
    Apply an apodization mask around the edge of the images in an mbobs to
    prevent FFT artifacts.

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to mask
    """

    ap_mask = np.ones_like(mbobs[0][0].image)

    _build_square_apodization_mask(AP_RAD, ap_mask)

    msk = np.where(ap_mask < 1)

    if msk[0].size > 0:
        for obslist in mbobs:
            for obs in obslist:

                # the pixels list will be reset upon exiting
                with obs.writeable():

                    obs.image[msk] *= ap_mask[msk]
                    obs.noise[msk] *= ap_mask[msk]

                    obs.bmask[msk] |= 1

                    if hasattr(obs, "mfrac"):
                        obs.mfrac[msk] = 1.0

                    if msk[0].size == obs.image.size:
                        obs.ignore_zero_weight = False

                    obs.weight[msk] = 0.0

                    if np.all(obs.weight == 0):
                        obs.ignore_zero_weight = False


@njit
def _build_square_apodization_mask(ap_rad, ap_mask):
    ap_range = get_ap_range()

    ny, nx = ap_mask.shape
    for y in range(min(ap_range + 1, ny)):
        for x in range(nx):
            ap_mask[y, x] *= _ap_kern_kern(y, ap_range, ap_rad)
            ap_mask[ny - 1 - y, x] *= _ap_kern_kern(y, ap_range, ap_rad)

    for y in range(ny):
        for x in range(min(ap_range + 1, nx)):
            ap_mask[y, x] *= _ap_kern_kern(x, ap_range, ap_rad)
            ap_mask[y, nx - 1 - x] *= _ap_kern_kern(x, ap_range, ap_rad)


@njit
def get_ap_range():
    """
    Get the range over which the the apodization kernel drops to zero

        int(6*ap_rad + 0.5)

    Parameters
    ----------
    ap_rad: float
        The apodization radius over which the kernel goes to zero

    Returns
    -------
    ap range as an integer
    """
    return int(6 * AP_RAD + 0.5)


@njit
def _ap_kern_kern(x, m, h):
    # cumulative triweight kernel
    y = (x - m) / h + 3
    if y < -3:
        return 0
    elif y > 3:
        return 1
    else:
        val = (
            -5 * y ** 7 / 69984
            + 7 * y ** 5 / 2592
            - 35 * y ** 3 / 864
            + 35 * y / 96
            + 1 / 2
        )
        return val


def do_metacal_and_process(mbobs, rng):
    import esutil as eu

    odict = do_all_metacal(mbobs=mbobs, rng=rng)

    dlist = []
    for key, mcal_mbobs in odict.items():
        st = process_one_mbobs(mbobs=mcal_mbobs, rng=rng)

        st['mcal_step'] = 'ns' if key == 'noshear' else key
        dlist.append(st)

    return eu.numpy_util.combine_arrlist(dlist)


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


def process_one_mbobs(mbobs, rng):
    detect_obs, weights = coadd_mbobs(mbobs)
    sxcat, seg = run_sep(detect_obs)

    is_primary = get_primary(sxcat)

    if TRIM_TO_PRIMARY:
        # when deblending we need to process all and trim
        # afterward
        w, = np.where(is_primary)
        sxcat = sxcat[w]
        is_primary = is_primary[w]

    cat = do_single_fits(
        mbobs=mbobs,
        weights=weights,
        sxcat=sxcat,
        rng=rng,
    )

    cat['xcell'] = sxcat['x']
    cat['ycell'] = sxcat['y']
    cat['is_primary'] = is_primary

    return cat


def get_dir(tract):
    return f'{tract:05d}'


def get_fname(tract, patch, with_mdet):

    fl = [f'cat-{tract:05d}-{patch:02d}']
    if with_mdet:
        fl += ['metacal']

    dir = get_dir(tract)
    bname = '-'.join(fl) + '.fits'
    return os.path.join(dir, bname)


def calculate_positions(bbox, wcs, cat):
    cat['x'] = cat['xcell'] + bbox.x.start
    cat['y'] = cat['ycell'] + bbox.y.start

    cat['ra'], cat['dec'] = wcs.pixelToSkyArray(
        cat['x'].astype('f8'),
        cat['y'].astype('f8'),
        degrees=True,
    )


def get_cell_info(nband, n=1):
    dtype = [
        ('tract', 'i4'),
        ('patch', 'i4'),
        ('cell_i', 'i4'),
        ('cell_j', 'i4'),
        ('good_frac', 'f4', nband),
        ('kept', bool),
    ]
    return np.zeros(n, dtype=dtype)


def write_output(fname, st, cell_info, tract, patch, seed, with_mdet):
    meta = np.zeros(1, dtype=[
        ('tract', 'i4'),
        ('patch', 'i4'),
        ('seed', 'i8'),
        ('with_mdet', bool),
        ('min_good_frac', 'f4'),
    ])
    meta['tract'] = tract
    meta['patch'] = patch
    meta['seed'] = seed
    meta['with_mdet'] = with_mdet
    meta['min_good_frac'] = MIN_GOOD_FRAC

    print('writing:', fname)
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_table(st, extname='cat', compress=True)
        fits.write_table(meta, extname='meta', compress=True)
        fits.write_table(cell_info, extname='cell_info', compress=True)


def main(tract, patch, seed, with_mdet, outfile, progress):
    from tqdm import trange
    import esutil as eu

    rng = np.random.RandomState(seed)

    butler = Butler('dp2_prep_future', collections=["LSSTCam/runs/DRP/DP2"])
    skymap = butler.get("skyMap", skymap=SKYMAP_VERS)
    tract_info = skymap[tract]
    wcs = tract_info.wcs

    dlist = []

    bands = ['r', 'i', 'z']
    fname = get_fname(tract=tract, patch=patch, with_mdet=with_mdet)
    dir = get_dir(tract)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    print(fname)

    deep_coadds = []
    for band in bands:
        data_id = {
            "band": band,
            "skymap": SKYMAP_VERS,
            "tract": tract,
            "patch": patch,
        }
        print(data_id)
        deep_coadd = butler.get('deep_coadd', dataId=data_id)
        deep_coadd.apply_background(None)
        deep_coadds.append(deep_coadd)

    if progress:
        mrng_i = trange(1, 21, desc='cell_i', ncols=80, ascii=True)
    else:
        mrng_i = range(1, 21)

    cell_info_list = []
    ncell = 0
    nkeep = 0
    for cell_i in mrng_i:
        if progress:
            mrng_j = trange(
                1, 21, desc='cell_j', ncols=80, ascii=True, leave=False,
            )
        else:
            mrng_j = range(1, 21)

        for cell_j in mrng_j:
            ncell += 1

            mbobs, cell_info = pull_mbobs(
                deep_coadds=deep_coadds,
                cell_i=cell_i,
                cell_j=cell_j,
                wcs=wcs,
            )
            cell_info['tract'] = tract
            cell_info['patch'] = patch
            cell_info_list.append(cell_info)

            if mbobs is None:
                continue

            apodize_mbobs(mbobs)

            if with_mdet:
                cat = do_metacal_and_process(mbobs=mbobs, rng=rng)
            else:
                cat = process_one_mbobs(mbobs=mbobs, rng=rng)
                cat['mcal_step'] = 'na'

            fit_and_set_psfrec(st=cat, mbobs=mbobs, rng=rng)

            calculate_positions(
                bbox=mbobs[0][0].meta['bbox'],
                wcs=wcs,
                cat=cat,
            )
            cat['cell_i'] = cell_i
            cat['cell_j'] = cell_j
            nkeep += 1
            dlist.append(cat)

        # break

    print(f'kept {nkeep}/{ncell} {nkeep / ncell:g}')

    cell_info = eu.numpy_util.combine_arrlist(cell_info_list)
    st = eu.numpy_util.combine_arrlist(dlist)

    write_output(
        fname=outfile,
        st=st,
        cell_info=cell_info,
        tract=tract,
        patch=patch,
        seed=seed,
        with_mdet=with_mdet,
    )


if __name__ == '__main__':
    _args = get_args()
    main(
        with_mdet=_args.mdet,
        seed=_args.seed,
        tract=_args.tract,
        patch=_args.patch,
        progress=_args.progress,
        outfile=_args.outfile,
    )
