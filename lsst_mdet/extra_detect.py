"""
extra-detection channels: starlet scale-2 (anull to come)
"""
from .detect import get_sx_config, make_kernel, DETECT_SETTINGS


# starlet scale-2 extra detections: the wavelet plane index
# (0-offset; plane 1 is the psf-matched band for ~4 px seeing),
# the exclusion radius vs sep and prior extras (4 px removes the
# near-degenerate pairs that destabilize the deblend; see
# docs/detection-adaptive-null), the mutual dedupe radius, and
# the number of noise realizations for the per-field threshold
# calibration
S2_JSCALE = 1
S2_EXTRA_MIN_SEP = 4.0
S2_EXTRA_DUP = 1.5
S2_NREAL = 20


def get_s2_extra_detections(
    mbobs,
    detobs,
    sxcat,
    seg,
    rng,
    prior_extras=None,
):
    """
    Extra detections from the seg-gated starlet scale-2 band: the
    psf-homogenized band images, zeroed outside the seg islands,
    are transformed to the wavelet plane matching the psf scale
    and summed with inverse-variance weights measured from the
    noise fields through the same transform.  Detection runs
    directly on that plane (no extra kernel: it re-merges close
    pairs) with the effective threshold set per field by the
    empirical quantile of pure noise through the identical
    processing, at the sep-matched per-pixel false rate.  The
    channel is color-blind scale contrast, complementary to the
    color filters; see docs/detection-adaptive-null.

    Peaks at least S2_EXTRA_MIN_SEP pixels from every sep
    detection and every prior extra (and S2_EXTRA_DUP apart) are
    returned for fit_deblend extra_detections; inject them with
    fixed centers (extra_fixcen), the validated stable setting.

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The per-band observations with noise fields
    detobs: ngmix.Observation
        The detection coadd; its jacobian frames the positions
    sxcat: array with fields
        The sep catalog from the coadd
    seg: array
        The sep segmentation map
    rng: np.random.RandomState
        For the psf size measurements and the calibration noise
        realizations
    prior_extras: (N, 2) array, optional
        Extra detections already accepted (e.g. the color-filter
        extras), excluded against like the sep catalog

    Returns
    -------
    (N, 2) array of (x, y) 0-offset pixel positions
    """
    import numpy as np
    from ngmix.moments import T_to_fwhm, fwhm_to_sigma
    import sxdes
    from scipy.stats import norm
    from scipy.ndimage import gaussian_filter

    khat = make_kernel()
    khat = khat / khat.sum()
    sep_thresh = DETECT_SETTINGS['thresh']
    p0 = norm.sf(sep_thresh / np.sqrt((khat ** 2).sum()))

    nband = len(mbobs)
    scale = detobs.jacobian.scale

    fwhms = [
        T_to_fwhm(obslist[0].psf.gmix.get_T())
        for obslist in mbobs
    ]
    # runner = _get_admom_runner(rng)
    # fwhms = []
    # for b in range(nband):
    #     res = runner.go(mbobs[b][0].psf)
    #     fwhms.append(T_to_fwhm(res['T']))

    target = max(fwhms)
    smooth_px = [
        (
            fwhm_to_sigma(np.sqrt(target ** 2 - fwhms[b] ** 2)) / scale
            if fwhms[b] < target - 1.0e-9 else 0.0
        )
        for b in range(nband)
    ]

    def homogenize(im, b):
        if smooth_px[b] > 0:
            return gaussian_filter(
                im, smooth_px[b], mode='reflect',
            )
        return im

    inseg = seg > 0
    planes, nplanes = [], []
    for b in range(nband):
        mim = homogenize(mbobs[b][0].image, b)
        mns = homogenize(mbobs[b][0].noise, b)
        planes.append(_starlet_plane(mim * inseg, S2_JSCALE))
        nplanes.append(_starlet_plane(mns, S2_JSCALE))

    wts = [1.0 / _mad_sigma(n) ** 2 for n in nplanes]
    det_im = sum(w * p for w, p in zip(wts, planes))

    # per-field effective threshold: sep's per-pixel threshold is
    # 0.8 x noise, so the quantile over pure-noise realizations
    # at 1 - p0, divided by 0.8, holds the channel to the
    # sep-matched per-pixel false rate despite the correlated
    # non-gaussian wavelet coefficients.  The realizations are
    # phase-randomized copies of each band's attached noise
    # field, preserving its correlation: metacal'd noise is
    # strongly correlated, and white realizations under-predict
    # the coefficient tail there (measured 8% false extras on
    # metacal images vs ~0 on the white-noise fields)
    vals = []
    for _ in range(S2_NREAL):
        nims = [
            homogenize(
                _phase_randomized(mbobs[b][0].noise, rng), b,
            )
            for b in range(nband)
        ]
        npl = [_starlet_plane(n, S2_JSCALE) for n in nims]
        nw = [1.0 / _mad_sigma(n) ** 2 for n in npl]
        vals.append(
            sum(w * p for w, p in zip(nw, npl)).ravel()
        )
    v = np.concatenate(vals)
    noise_eff = np.quantile(v, 1 - p0) / sep_thresh

    sx_config = dict(get_sx_config())
    sx_config['filter_kernel'] = None
    sx_config['filter_type'] = 'matched'
    cat, _ = sxdes.run_sep(
        image=det_im.astype('f4').copy(),
        noise=noise_eff, config=sx_config,
    )

    ax = sxcat['x']
    ay = sxcat['y']
    if prior_extras is not None and len(prior_extras):
        ax = np.concatenate([ax, prior_extras[:, 0]])
        ay = np.concatenate([ay, prior_extras[:, 1]])
    extras = []
    for xi, yi in zip(cat['x'], cat['y']):
        # ax can be empty when sep found nothing (heavily
        # masked cell) and there are no prior extras
        if ax.size and np.min(
            (ax - xi) ** 2 + (ay - yi) ** 2
        ) < S2_EXTRA_MIN_SEP ** 2:
            continue
        if extras and np.min([
            (xa - xi) ** 2 + (ya - yi) ** 2
            for xa, ya in extras
        ]) < S2_EXTRA_DUP ** 2:
            continue
        extras.append((xi, yi))
    return np.array(extras).reshape(-1, 2)


def _starlet_plane(image, jplane):
    """
    One plane of the generation-1 a-trous B3 starlet transform:
    w_j = c_j - c_{j+1} with the dilated [1,4,6,4,1]/16 kernel
    and reflecting boundaries, returned for plane index jplane
    (0-offset, so jplane=1 is 'scale 2')
    """
    import numpy as np
    from scipy.ndimage import convolve1d

    b3 = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    c = image.astype('f8')
    for j in range(jplane + 1):
        step = 2 ** j
        k = np.zeros(4 * step + 1)
        k[::step] = b3
        s = convolve1d(c, k, axis=0, mode='reflect')
        s = convolve1d(s, k, axis=1, mode='reflect')
        if j == jplane:
            return c - s
        c = s


def _mad_sigma(x):
    import numpy as np

    # exclude exact zeros: star-masked (apodized) noise planes
    # can be mostly zero, driving the all-pixel MAD to zero and
    # the inverse-variance weights to infinity
    v = x[x != 0]
    if v.size < 100:
        v = x
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def _phase_randomized(noise, rng):
    """
    A new realization of a stationary noise field: keep the
    amplitude spectrum, randomize the phases.  Preserves the
    field's correlation function exactly in expectation, which
    white draws do not for metacal'd (correlated) noise.
    """
    import numpy as np

    n0 = noise - noise.mean()
    f = np.fft.rfft2(n0)
    phases = np.exp(
        2j * np.pi * rng.uniform(size=f.shape)
    )
    return np.fft.irfft2(np.abs(f) * phases, s=n0.shape)
