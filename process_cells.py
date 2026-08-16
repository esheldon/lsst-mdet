"""
TODO

- mfrac for deblended objects
- star mask
- add deblending with special flags, gauss_, fam_
- decide MIN_GOOD_FRAC
- decide if keeping non primary
- detection kernel (fixed or variable)
- Tfrac (requires upstream)
- keep track of fitted x/y?

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
PSF_FAILURE = 2 ** 21
BAD_BBOX = 2 ** 22
ZERO_WEIGHTS = 2 ** 23
# FLAG_DEBLEND_FAILED = 2 ** 24
FLAG_NOT_CONVERGED = 2 ** 25

MIN_GOOD_FRAC = 0.2

OVERLAP = 50
OVERLAP_LOW = 50
OVERLAP_HIGH = 200

SKYMAP_VERS = 'lsst_cells_v2'
AP_RAD = 1.5

MFRAC_FWHM = 1.2

GROUP_BOX_PAD = 10

# with maxiter_type 'scaled', the sweep cap applies as configured
# up to this group size and is scaled up linearly with the member
# count beyond it: sweeps to converge grow with group size
# (Gauss-Seidel information propagates about one object per
# sweep; measured on 2000 wldb fields the converged-group median
# numiter goes 11 -> 133 and the p99 102 -> 591 from single
# objects to 17-64 members, and an extras-enabled 31-member
# cluster core needs 1034).  'fixed' is right for wide fields,
# where the long-running groups are almost all non-converging
# limit cycles and the scaled caps only multiply their cost

MAXITER_SIZE_REF = 8.0


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--outfile', required=True)
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--redo-bg', action='store_true')
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
                fit_gauss(
                    st=cat[i],
                    rng=rng,
                    mbobs=stamp_mbobs,
                )

                # cat[i] = fit_struct

                cat['mfrac'][i] = calculate_mfrac(
                    mbobs=stamp_mbobs,
                    mfrac_weight=mfrac_weight,
                )

            except IndexError as err:
                cat['flags'][i] = BAD_BBOX
                print(f'stamp for obj {i} hit edge: {err}')
            except GMixFatalError as err:  # noqa
                cat['flags'][i] = ZERO_WEIGHTS
                # print(f'obj {i}: {err}')

    else:
        cat['flags'] = PSF_FAILURE

    cat['xcell'] = sxcat['x']
    cat['ycell'] = sxcat['y']
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


def fit_gauss(st, rng, mbobs):
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
            flux_err=res['flux_err'],
            flux_cov=res['flux_cov'],
        )

        st['s2n'] = res['s2n']

    # return st


def pack_deblend_object(st, obj_res, bands, jacobian):
    """
    flags, deblend_flags, numiter set outside
    """

    g1, g2, g1_err, g2_err = _e2g(
        e1=obj_res['e1'],
        e2=obj_res['e2'],
        e1_err=obj_res['e1_err'],
        e2_err=obj_res['e2_err'],
    )

    st['g1'] = g1
    st['g1_err'] = g1_err
    st['g2'] = g2
    st['g2_err'] = g2_err
    st['T'] = obj_res['T']
    st['T_err'] = obj_res['T_err']

    _set_fluxes(
        st=st,
        bands=bands,
        flux=obj_res['flux'],
        flux_err=obj_res['flux_err'],
    )

    _set_colors(
        st=st,
        bands=bands,
        flux=obj_res['flux'],
        flux_err=obj_res['flux_err'],
        flux_cov=obj_res['flux_cov'],
    )

    st['s2n'] = obj_res['s2n']


def _e2g(e1, e2, e1_err, e2_err):
    """
    convert distortion-convention shapes to reduced shear,
    g = e / (1 + sqrt(1 - e^2)), with diagonal error propagation.
    The deblender guarantees e^2 < 1 for usable shapes (the det
    condition); the clip only guards float rounding at the boundary
    """
    import numpy as np

    if np.isfinite(e1) and np.isfinite(e2):
        u = e1 * e1 + e2 * e2
        s = np.sqrt(max(1.0 - u, 0.0))
        f = 1.0 / (1.0 + s)
        g1 = e1 * f
        g2 = e2 * f

        # dg_i/de_j = f delta_ij + 2 e_i e_j f'; f' = df/d(e^2)
        if s > 0:
            fp = 1.0 / (2 * s * (1.0 + s) ** 2)
        else:
            fp = 0.0
        g1_err = np.sqrt(
            (f + 2 * e1 * e1 * fp) ** 2 * e1_err ** 2
            + (2 * e1 * e2 * fp) ** 2 * e2_err ** 2
        )
        g2_err = np.sqrt(
            (2 * e1 * e2 * fp) ** 2 * e1_err ** 2
            + (f + 2 * e2 * e2 * fp) ** 2 * e2_err ** 2
        )
    else:
        g1, g2, g1_err, g2_err = [np.nan] * 4

    return g1, g2, g1_err, g2_err


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


def _set_colors(st, bands, flux, flux_err, flux_cov):
    """
    Set colors and errors based on the full covariance
    """
    fac = 2.5 / np.log(10)
    eps = 1.0e-7

    nband = len(bands)

    if flux_cov is None:
        flux_cov = np.diag(flux_err ** 2)

    for i in range(nband - 1):
        first_band = bands[i]
        second_band = bands[i + 1]
        cname = f'{first_band}m{second_band}'

        if flux[i] > eps and flux[i + 1] > eps:
            color = -2.5 * np.log10(flux[i] / flux[i + 1])

            color_var = fac ** 2 * (
                flux_cov[i, i] / flux[i] ** 2
                + flux_cov[i + 1, i + 1] / flux[i + 1] ** 2
                - 2 * flux_cov[i, i + 1] / (flux[i] * flux[i + 1])
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


def fit_deblend(
    mbobs,
    weights,
    sxcat,
    seg,
    rng,
    show=False,
):
    """
    Deblend and measure all detected objects

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The per-band observations to fit, each holding image,
        weight, psf and the noise field attached by do_metacal,
        from which the per-mode noise power for the flux errors is
        measured.  A single Observation is also accepted
    sxcat: array with fields
        The sep catalog from detect.run_sep on detobs
    seg: array
        The sep segmentation map, used for the fofx grouping
    rng: np.random.RandomState
        The random number generator
    show: bool, optional
        If set to True, show a kdeblend view_blend figure for each
        blend group after its fit: the region of the field bounding
        the group's stamps, with the fitted models, the seg map and
        the stamp boxes, titled with the blend group id
    extra_detections: array, optional
        (N, 2) array of (x, y) 0-offset pixel positions of extra
        objects to inject into the deblend, e.g. peaks found on
        the adaptive-null detection images.  Each position joins the
        blend group of the seg island it lands on and is fit
        jointly with that group, with the configured model (never
        dev-classified), a size guess from the smoothing scale,
        and full kdeblend measurements; the catalog gains one row
        per position, marked with color_det=1.  A position landing
        on seg background is not fit; its row gets
        flags=FLAG_EXTRA_DET_OFF_SEG.  The caller chooses which
        positions to inject (e.g. an exclusion radius against the
        sep detections).  Group modes only: extra detections have
        no sep bbox for the stamp cutting, so deblend_mode
        'stamps' raises an error
    extra_fixcen: bool array, optional
        Per extra detection, keep its center fixed at the
        injected position even when recenter is on (kdeblend
        object fixcen).  Frees injected positions whose adaptive
        centers would couple degenerately to nearby members;
        note a fixed-center extra is never duplicate-flagged
        (the duplicate test is defined by fitted centers
        converging together)

    Returns
    -------
    cat, keep

    cat: array with fields
        One row per sxcat detection, followed by one row per
        extra detection (in extra_detections order); see
        fitting.get_kdeblend_struct
    keep: bool array
        Which rows of sxcat were kept (all of them; the stamp
        cutting clips at edges rather than failing).  Length
        sxcat.size: the extra-detection rows are not covered
    """
    import numpy as np
    # from ngmix.moments import fwhm_to_T
    from ngmix.prepsfadmom.prep import choose_fwhm_smooth

    model = 'exp'
    tol = 1.0e-5
    maxiter = 500
    maxiter_type = "fixed"
    recenter = True
    full_errors = True

    ap_rad = 1.5  # pixels
    cen_sigma0 = 0.1
    e_sigma0 = 0.0
    tguess_range = (0.05, 5.0)

    bands = [obslist[0].meta['band'] for obslist in mbobs]

    for obslist in mbobs:
        if not obslist[0].has_noise():
            raise ValueError(
                'each band observation must have a noise field for '
                'the per-mode noise power'
            )

    jacobian = mbobs[0][0].jacobian
    scale = jacobian.scale
    v, u = jacobian.get_vu(row=sxcat['y'], col=sxcat['x'])

    Tguess = np.clip(
        (sxcat['x2'] + sxcat['y2']) * scale ** 2,
        tguess_range[0], tguess_range[1],
    )

    objects = [
        {
            'v': v[i],
            'u': u[i],
            'type': model,
            'Tguess': Tguess[i],
        }
        for i in range(sxcat.size)
    ]

    # one psf fit on the detection coadd fills the psf fields for
    # all objects; the smoothing scale covers the largest band psf
    # psf_runner = _get_admom_runner(rng)
    # psf_res = psf_runner.go(detobs.psf)
    psf_res = fit_and_set_mcal_psfs(
        mbobs=mbobs,
        weights=weights,
        rng=rng,
    )

    fwhm_smooth = choose_fwhm_smooth(mbobs, rng=rng)
    # Tsmooth = fwhm_to_T(fwhm_smooth)

    groups = get_groups(sxcat=sxcat, seg=seg)

    # add the extra detections: each joins the group of the seg
    # island it lands on, with the configured model and a size
    # guess from the smoothing scale (the compact-start direction
    # is the safe one, see the Tguess comment above); positions on
    # seg background are not fit and their rows flagged
    nsx = sxcat.size
    n_extra = 0
    # off_seg = []
    # if extra_detections is not None:
    #     extra_detections = np.atleast_2d(extra_detections)
    #     n_extra = len(extra_detections)
    #     groups = [list(g) for g in groups]
    #     num_to_group = {}
    #     for gid, group in enumerate(groups):
    #         for i in group:
    #             num_to_group[int(sxcat['number'][i])] = gid
    #     Tguess_inj = float(
    #         np.clip(Tsmooth, TGUESS_RANGE[0], TGUESS_RANGE[1])
    #     )
    #     dim_r, dim_c = seg.shape
    #     for k, (x, y) in enumerate(extra_detections):
    #         objects.append(dict(
    #             v=(y - jrow) * scale,
    #             u=(x - jcol) * scale,
    #             type=model,
    #             Tguess=Tguess_inj,
    #             fixcen=bool(
    #                 extra_fixcen is not None and extra_fixcen[k]
    #             ),
    #             **bdf_entries,
    #         ))
    #         ir = int(round(y))
    #         ic = int(round(x))
    #         label = (
    #             int(seg[ir, ic])
    #             if 0 <= ir < dim_r and 0 <= ic < dim_c else 0
    #         )
    #         if label in num_to_group:
    #             groups[num_to_group[label]].append(nsx + k)
    #         else:
    #             off_seg.append(nsx + k)

    cat = get_struct(bands=bands, n=nsx + n_extra)

    # cat['color_det'][nsx:] = 1
    # extras have no sep flux_auto and keep the nan init
    # cat['s2n_det'][:nsx] = s2n_det
    # for idx in off_seg:
    #     cat['flags'][idx] = FLAG_EXTRA_DET_OFF_SEG
    # cat['psf_flags'] = psf_res['flags']
    # if psf_res['flags'] == 0:
    #     cat['psf_T'] = psf_res['T']

    # the final per-group fit for the visualization: pass-2 refits
    # overwrite the pass-1 entries, so each group is shown once with
    # the measurements that landed in the catalog
    group_shows = {}

    # pass 1: fit each group with no knowledge of the rest of the
    # field
    # results = {}
    for gid, group in enumerate(groups):

        # try:
        res, gextra = fit_one_group(
            group=group,
            mbobs=mbobs,
            sxcat=sxcat,
            nsx=nsx,
            seg=seg,
            objects=objects,
            maxiter=maxiter,
            maxiter_type=maxiter_type,
            fwhm_smooth=fwhm_smooth,
            ap_rad=ap_rad,
            tol=tol,
            rng=rng,
            recenter=recenter,
            cen_sigma0=cen_sigma0,
            e_sigma0=e_sigma0,
            full_errors=full_errors,
        )

        # except Exception as err:
        #     print(f'deblend failed for group {group}: {err}')
        #     for i in group:
        #         cat['flags'][i] = FLAG_DEBLEND_FAILED
        #     continue

        for i, obj_res in zip(group, res['objects']):
            pack_deblend_object(
                st=cat[i],
                obj_res=obj_res,
                bands=bands,
                jacobian=jacobian,
            )
            # results[i] = obj_res

        cat['numiter'][group] = res['numiter']
        cat['group_size'][group] = len(group)

        if not res['converged']:
            for i in group:
                cat['flags'][i] |= FLAG_NOT_CONVERGED

        if show:
            show_group(
                mbobs=gextra['gmbobs'],
                seg=gextra['gseg'],
                objects=res['objects'],
                group=groups[gid],
                title=f'blend group {gid}',
            )

    # if show:
    #     for gid in sorted(group_shows):
    #         robjs = group_shows[gid]
    #         show_group(
    #             mbobs=mbobs,
    #             seg=seg,
    #             objects=robjs,
    #             group=groups[gid],
    #             title=f'blend group {gid}',
    #         )

    # flag extra rows whose fitted centers converged onto a sep
    # row's fitted center: nuisance components of the same object.
    # They stay in the fit -- they soak crowd light and profile
    # mismatch, improving the photometry of the real rows -- but
    # must not enter downstream selections (a fraction would pass
    # the standard cuts).  Convergence is only meaningful when the
    # centers are refit, so this requires recenter
    # if n_extra and recenter:
    #     xf, yf = cat['x_fit'], cat['y_fit']
    #     for k in range(n_extra):
    #         i = nsx + k
    #         if cat['flags'][i] != 0:
    #             continue
    #         d2 = np.nanmin(
    #             (xf[:nsx] - xf[i]) ** 2 + (yf[:nsx] - yf[i]) ** 2
    #         )
    #         if d2 < R_DUP_FIT ** 2:
    #             cat['flags'][i] |= FLAG_DUPLICATE_EXTRA

    cat['xcell'] = sxcat['x']
    cat['ycell'] = sxcat['y']
    _set_mcal_psfs(st=cat, psf_res=psf_res)
    return cat


def fit_one_group(
    group,
    mbobs,
    sxcat,
    nsx,
    seg,
    objects,
    maxiter,
    maxiter_type,
    fwhm_smooth,
    ap_rad,
    tol,
    rng,
    recenter,
    cen_sigma0,
    e_sigma0,
    full_errors,
):
    """
    cut and fit one deblend group with the configured parameters.
    Extracted from the former fit_deblend closure so external
    harnesses (e.g. the port differential rig) can drive the exact
    production per-group path.

    Returns
    -------
    res, boxes, gcut
        the kdeblend result, the per-member boxes, and the group
        cut (gmbobs, box) for group modes (None for stamps mode)
    """
    from kdeblend import deblend

    if maxiter_type == 'scaled':
        # large groups converge in proportionally more
        # sweeps; see MAXITER_SIZE_REF
        group_maxiter = int(round(
            maxiter * max(1.0, len(group) / MAXITER_SIZE_REF)
        ))
    else:
        group_maxiter = maxiter

    scale = mbobs[0][0].jacobian.scale

    # the cutout box and the member seg labels come from
    # the sep members only; extra-detection members carry
    # no sep bbox but land inside the island by
    # construction
    gmbobs, gseg, box = cut_group_mbobs(
        mbobs=mbobs,
        sxcat=sxcat,
        group=[i for i in group if i < nsx],
        seg=seg,
    )

    gobjects = [objects[i] for i in group]
    res = deblend(
        gmbobs,
        gobjects,
        fwhm_smooth=fwhm_smooth,
        ap_rad=ap_rad,
        use_noise_image=True,
        tol=tol,
        maxiter=group_maxiter,
        rng=rng,
        recenter=recenter,
        cen_sigma0=cen_sigma0,
        e_sigma0=e_sigma0,
        full_errors=full_errors,
        anchor_sigma=(
            anchor_covs(sxcat, group, nsx, scale)
            if full_errors and recenter else 0.0
        ),
    )

    gextra = {
        'gmbobs': gmbobs,
        'gseg': gseg,
        'gobjects': gobjects,
        'gbox': box,
    }
    return res, gextra


def show_group(
    mbobs,
    seg,
    objects,
    group,
    title,
    pad=10,
):
    """
    show the kdeblend view_blend figure for one blend group: the
    region of the field bounding the group's stamps, with the
    fitted models rendered, the seg map as the fourth panel and the
    stamp boxes drawn on every panel, members labeled by catalog
    index.  At most the first three bands are shown.

    was actually fit; the box interior of every shown band is
    overwritten with those pixels so the display shows what the
    fitter saw -- in group-replace mode the external objects'
    pixels are noise there, not the original data
    """
    from kdeblend import vis

    vis.view_blend(
        mbobs,
        objects,
        bands=list(range(min(len(mbobs), 3))),
        seg=seg,
        labels=[str(i) for i in group],
        title=title,
        show=True,
    )


def anchor_covs(sxcat, group, nsx, scale):
    """
    per-member anchor position covariances in arcsec^2 with
    (v, u) ordering, from the sep centroid error moments
    (erry2/errxy/errx2 in pixels^2).  The anchor noise of the
    detection positions is a real error channel for recentered
    tight blends (it can double the flux variance at the
    detection-centroid scale).  Extra-detection members have no
    sep moments and get zero (their errors stay conditional on
    the injected positions).  Non-finite or non-positive-definite
    sep moments are sanitized: bad variances zeroed, the cross
    term clamped inside the PSD bound
    """
    import numpy as np

    out = np.zeros((len(group), 2, 2))
    for k, i in enumerate(group):
        if i >= nsx:
            continue
        vv = float(sxcat['erry2'][i])
        uu = float(sxcat['errx2'][i])
        vu = float(sxcat['errxy'][i])
        if not np.isfinite(vv) or vv < 0:
            vv = 0.0
        if not np.isfinite(uu) or uu < 0:
            uu = 0.0
        lim = 0.99 * np.sqrt(vv * uu)
        if not np.isfinite(vu):
            vu = 0.0
        vu = np.clip(vu, -lim, lim)
        out[k] = scale ** 2 * np.array([
            [vv, vu], [vu, uu],
        ])
    return out


def cut_group_mbobs(mbobs, sxcat, group, seg):
    """
    cut the shared deblending image for a group from every band:
    the box bounding the union of the members' seg-footprint
    bounding boxes, padded by GROUP_BOX_PAD pixels all around and
    clipped at the field edges.  The cut keeps the sky frame of
    the field (the jacobian centers are shifted by the cut
    origin), so the object v/u offsets are unchanged.  The noise
    fields are cut along with the images.

    the pixels of objects outside the group are
    replaced with values from the 180-degree-rotated noise field
    at the same box location: the rotation preserves a stationary
    covariance exactly (C(-d) = C(d), including the metacal
    anisotropy) and is independent of both the image noise and
    the attached noise field away from the field center.  The
    attached noise fields are not modified.

    Returns
    -------
    mbobs, box
        box is (row_start, col_start, nrow, ncol) of the cut in
        the field image
    """
    import numpy as np
    import ngmix

    obs0 = mbobs[0][0]
    dim_r, dim_c = obs0.image.shape

    r0 = max(int(sxcat['ymin'][group].min()) - GROUP_BOX_PAD, 0)
    r1 = min(
        int(sxcat['ymax'][group].max()) + GROUP_BOX_PAD + 1,
        dim_r,
    )
    c0 = max(int(sxcat['xmin'][group].min()) - GROUP_BOX_PAD, 0)
    c1 = min(
        int(sxcat['xmax'][group].max()) + GROUP_BOX_PAD + 1,
        dim_c,
    )

    nrow = r1 - r0
    ncol = c1 - c0
    sl = np.s_[r0:r1, c0:c1]

    segcut = seg[sl]
    foreign = (segcut != 0) & ~np.isin(
        segcut, sxcat['number'][group],
    )
    if not foreign.any():
        foreign = None

    out = ngmix.MultiBandObsList()
    for obslist in mbobs:
        obs = obslist[0]
        jrow, jcol = obs.jacobian.get_cen()
        jacobian = obs.jacobian.copy()
        jacobian.set_cen(row=jrow - r0, col=jcol - c0)

        image = obs.image[sl].copy()
        if foreign is not None:
            image[foreign] = np.rot90(obs.noise, 2)[sl][foreign]

        gobs = ngmix.Observation(
            image=image,
            weight=obs.weight[sl].copy(),
            jacobian=jacobian,
            noise=obs.noise[sl].copy(),
            psf=obs.psf,
            # psf=_cut_psf_obs(psf_obs=obs.psf, nrow=nrow, ncol=ncol),
        )
        ol = ngmix.ObsList()
        ol.append(gobs)
        out.append(ol)

    return out, segcut, (r0, c0, nrow, ncol)


def get_groups(sxcat, seg):
    """
    group objects by the union of seg-touching links (fofx) and
    moment-relevance links

    Parameters
    ----------
    sxcat: array with fields
        The sep catalog, with the 'number' field matching the seg
        map values
    seg: array
        The sep segmentation map

    Returns
    -------
    list of lists of catalog indices
    """
    import fofx

    nobj = sxcat.size
    parent = list(range(nobj))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    fofs = fofx.get_fofs(seg)
    number_to_index = {
        number: i for i, number in enumerate(sxcat['number'])
    }
    first = {}
    for fof_id, number in zip(fofs['fof_id'], fofs['number']):
        if number in number_to_index:
            i = number_to_index[number]
            if fof_id in first:
                union(first[fof_id], i)
            else:
                first[fof_id] = i

    groups = {}
    for i in range(nobj):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


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
        ('numiter', 'i2'),
        ('group_size', 'i2'),
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


def do_metacal_and_process(mbobs, rng, show):
    import esutil as eu

    odict = do_all_metacal(mbobs=mbobs, rng=rng)

    dlist = []
    for key, mcal_mbobs in odict.items():
        st = process_one_mbobs(mbobs=mcal_mbobs, rng=rng, show=show)

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


def process_one_mbobs(mbobs, rng, show):
    detect_obs, weights = coadd_mbobs(mbobs)
    sxcat, seg = run_sep(detect_obs)

    is_primary = get_primary(sxcat)

    if TRIM_TO_PRIMARY:
        # when deblending we need to process all and trim
        # afterward
        w, = np.where(is_primary)
        sxcat = sxcat[w]
        is_primary = is_primary[w]

    if False:
        cat = do_single_fits(
            mbobs=mbobs,
            weights=weights,
            sxcat=sxcat,
            rng=rng,
        )
    else:
        cat = fit_deblend(
            mbobs=mbobs,
            weights=weights,
            sxcat=sxcat,
            seg=seg,
            rng=rng,
            show=show,
        )

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


def redo_background(deep_coadd):
    import sep

    image = deep_coadd.image.array
    var = deep_coadd.variance.array
    mask = deep_coadd.mask.array[:, :, 0]
    noise = deep_coadd.noise_realizations[0].array

    good = (
        np.isfinite(var)
        & np.isfinite(noise)
        & (mask & DM_OUT == 0)
    )
    bad = ~good

    bkg = sep.Background(image, mask=bad)
    # image -= bkg.back()

    objects, seg = sep.extract(
        image - bkg.back(),
        1.5,
        mask=bad,
        err=bkg.globalrms,
        segmentation_map=True,
    )
    new_good = good & (seg == 0)
    bad = ~new_good
    bkg = sep.Background(image, mask=bad)

    for i in range(0):
        objects, seg = sep.extract(
            image,
            1.0,
            mask=bad,
            err=bkg.globalrms,
            segmentation_map=True,
        )

        new_good = good & (seg == 0)
        bad = ~new_good
        bkg = sep.Background(image, mask=bad)

        image -= bkg.back()

    image[:, :] -= bkg.back()

    w = np.where(new_good)
    medvar = np.median(var[w])

    noise_factor = bkg.globalrms / np.sqrt(medvar)
    # noise_factor = bkg.globalrms / noise.std()
    print(f'    band: {deep_coadd.band} noise_factor: {noise_factor:g}')

    noise[:, :] *= noise_factor
    var[:, :] *= noise_factor ** 2


def redo_background_old(deep_coadd):
    import sep

    image = deep_coadd.image.array
    var = deep_coadd.variance.array
    mask = deep_coadd.mask.array[:, :, 0]
    noise = deep_coadd.noise_realizations[0].array

    good = np.isfinite(var) & (mask & DM_OUT == 0)
    w = np.where(good)

    bkg = sep.Background(image, mask=~good)
    image[:, :] -= bkg.back()

    medvar = np.median(var[w])

    noise_factor = bkg.globalrms / np.sqrt(medvar)
    print(f'    band: {deep_coadd.band} noise_factor: {noise_factor:g}')

    noise[:, :] *= noise_factor
    var[:, :] *= noise_factor ** 2


def main(tract, patch, seed, with_mdet, redo_bg, outfile, progress, show):
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
        if redo_bg:
            redo_background(deep_coadd)
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
                cat = do_metacal_and_process(mbobs=mbobs, rng=rng, show=show)
            else:
                cat = process_one_mbobs(mbobs=mbobs, rng=rng, show=show)
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
        show=_args.show,
        redo_bg=_args.redo_bg,
        outfile=_args.outfile,
    )
