"""
Gaia-driven bright-star subtraction and masking
"""
import numpy as np
from .defaults import DM_INTRP, DM_OUT, DM_SAT
from .gaia import gaia_pixel_positions

# every census star is subtracted with the empirical extended
# template and masked with a floored magnitude-scaled circle
# that hides the core, where a field-average template is wrong.
# The circle law follows the flagged-arm extent measured over
# 56 saturated stars in 3 patches
MASK_R15 = 45.0     # circle radius in px at G = 15
MASK_SLOPE = 0.115  # radius scales as 10^(slope * (15 - G))
MASK_RMAX = 450.0
MINRAD = 20.0       # circle floor. subtracted cores never show
STAR_MARGIN = 210   # off-patch stars whose wings still intrude
GSAT = 15.2         # G saturation threshold of these coadds
GSUB = 19.0         # subtract stars brighter than this
RUWE_MAX = 1.4      # template-star astrometric-quality guard
BG_GROW = 12        # extra star-mask margin for the background
APOD_STARS = 12.0   # taper width outside the star mask

# empirical extended star template
TMPL_HALF = 50       # measured stamp half size
TMPL_OUT_HALF = 250  # minimum halo extension half size
# per-star extent. TMPL_EXT_FACTOR times the mask radius,
# capped.  A fixed 250 px edge leaves a G~10 star's halo
# (~0.5 sigma there) unsubtracted beyond it, visible as a
# ring at the stamp edge
TMPL_EXT_FACTOR = 3.0
TMPL_OUT_MAX = 900
TMPL_NSTAR = 60
TMPL_GMIN = GSAT + 0.3  # template stars. bright but unsaturated
TMPL_GMAX = 17.5        # preferred faint limit
TMPL_GMAX_CAP = 19.0    # adaptive faint-limit cap (census depth)
TMPL_MIN_CAND = 20      # extend the faint limit below this
TMPL_MIN_STAMPS = 10    # hard minimum usable stamps
# inner (turbulence-wing) power law fallback, for corrupted
# fits, measured with the pedestal-robust joint fit
HALO_SLOPE = -4.0

# the outer aureole (atmospheric + instrumental scattering):
# a second, flatter power law that dominates beyond ~60 px.
# Measured per band from the mid-bright stars when the field
# allows, with a tiered fallback for sparse fields
AUR_GMIN = 13.0      # aureole measurement stars
AUR_GMAX = 15.5
AUR_RMAX = 250.0     # fit limit. beyond this the ambient
#                      source-carpet floor takes over
AUR_SLOPE = -2.0     # canonical scattering-aureole fallback
AUR_SLOPE_MIN = -3.5  # tier-1 fitted-slope guard
AUR_SLOPE_MAX = -1.5
AUR_MIN_STARS = 10   # tier 1 below this falls to tier 2
AUR_SLOPE_SEP = 1.0  # tier-1 slope kept this much flatter
#                      than the inner wing (degeneracy guard)
AUR_BREAK = 80.0     # tier-3 continuity radius
AUR_AMP_GUARD = 10.0  # fitted amp within this factor of the
#                       continuity value, else tier 3

# canonical per-band template constants, measured from the 15
# clean (non-crowded) fields of the 19-field DP2 shape study
# (scratchpad canonical_shape.py, ambient-referenced profiles,
# 2026-08).  Field-to-field the wing profile is a one-parameter
# family: a common shape times a per-field amplitude that
# correlates with the visit-set seeing (the turbulence halo).
# slope/ln_a: the canonical inner law at unit amplitude
# (core-normalized template units); the seeing relation for the
# amplitude prior is ln A = dlna_dfwhm * (fwhm - fwhm_ref) with
# rms prior_sig about it (raw_sig when no fwhm is available).
# The canonical aureole is NOT tabulated: it cannot be measured
# from unrestored images (the production background absorbs the
# outer wings), so the canonical route uses continuity at
# AUR_BREAK, the tier-3 convention; remeasure from restored
# production runs when a body of them exists
CANON = {
    'g': dict(
        slope=-3.800, ln_a=0.292,
        dlna_dfwhm=1.787, fwhm_ref=1.014,
        prior_sig=0.217, raw_sig=0.370,
    ),
    'r': dict(
        slope=-3.900, ln_a=0.774,
        dlna_dfwhm=1.067, fwhm_ref=1.052,
        prior_sig=0.189, raw_sig=0.206,
    ),
    'i': dict(
        slope=-3.870, ln_a=0.691,
        dlna_dfwhm=0.919, fwhm_ref=0.968,
        prior_sig=0.145, raw_sig=0.173,
    ),
    'z': dict(
        slope=-3.880, ln_a=0.666,
        dlna_dfwhm=2.431, fwhm_ref=0.936,
        prior_sig=0.120, raw_sig=0.181,
    ),
}
# the canonical sparse-field route: below TMPL_MIN_STAMPS but
# at least this many stamps, fit one amplitude against the
# canonical shape (with the seeing prior) instead of the free
# slope/pedestal/tiered-aureole machinery
CANON_MIN_STAMPS = 3

# local restoration of the stored 'object' background model.
# that model absorbs star wings and scattered
# light; adding it back around the bright stars restores the
# wing light so the template can subtract it as star flux.
# ALL saturated stars need it. with the old G < 13 cut the
# mid-bright (13-15.2) rings measured the absorbed remainder,
# the flux relation disagreed, and the amplitude floor
# over-subtracted (worst in z, where the red halo spreads the
# ring amplitudes)
RESTORE_GMAX = GSAT    # restore around stars brighter than this
RESTORE_RAD = 400.0    # full restoration within this distance
#                        of the bright-star masks
RESTORE_TAPER = 100.0  # taper width down to zero

# mask-aware preliminary background, applied after the
# restoration and before the template/subtraction. flattens
# the sky the template stack and amplitude anchors sit on,
# without chasing the (excluded) star wings
PRE_BW = 64            # background box size
PRE_GROW = 12          # exclusion beyond every star mask
PRE_GROW_BRIGHT = 128  # exclusion beyond the bright-star masks

NPASS = 3           # joint amplitude passes

# NOTE on per-star local sky references for the amplitude
# anchors (tried 2026-08, reverted): the i-band monster
# collars come from local light under the bright stars that
# the global ambient reference cannot see, but every local
# referencing scheme tested (two-band pedestal difference,
# far-band local level, bright-star-only far band) traded
# that bias for template-shape sensitivity or coupling to the
# neighbors' model errors, degrading the well-measured faint
# and mid bins or the z monsters.  The monsters are per-star
# structure-limited (a handful of stars per field, each with
# individually different surroundings); the plain ratio
# anchor with the global ambient reference is the measured
# optimum


def circle_radius(gmag):
    """
    the mask circle radius law

     magnitude-scaled with a floor and a cap

    Parameters
    ----------
    gmag: float or array
        Gaia G magnitude

    Returns
    -------
    radius in pixels, same shape as gmag
    """

    return np.minimum(
        np.maximum(
            MASK_R15 * 10 ** (MASK_SLOPE * (15.0 - gmag)),
            MINRAD,
        ),
        MASK_RMAX,
    )


def select_stars(gaia, x, y, mask0, gsub=GSUB):
    """
    Select on-patch stars that are saturated or brighter than gsub

    WE do not apply a RUWE guard.  Gaia at these depths is essentially pure
    point sources, and high-RUWE binaries are still stars we want gone.

    Also include off-patch intruders bright enough for their wings to reach in

    Referred to as "the subtract-and-mask census" in various places

    Parameters
    ----------
    gaia: array with fields
        The gaia extract (ra, dec, phot_g_mean_mag, ruwe)
    x, y: arrays
        Patch-frame pixel positions of the gaia stars
    mask0: array
        The DM mask plane, for the saturation test
    gsub: float, optional
        Census depth: unsaturated on-patch stars brighter than
        this are included

    Returns
    -------
    stars: structured array
        Fields ra, dec, x, y, G, ruwe, is_sat, on_image,
        sorted brightest first
    """

    ny, nx = mask0.shape
    sat = (mask0 & DM_SAT) != 0
    rows = []

    for k in np.argsort(gaia['phot_g_mean_mag']):

        gmag = float(gaia['phot_g_mean_mag'][k])
        ruwe = float(gaia['ruwe'][k])

        xk, yk = float(x[k]), float(y[k])
        ix, iy = int(round(xk)), int(round(yk))

        on = 0 <= ix < nx and 0 <= iy < ny

        if on:
            m = 5
            is_sat = (
                gmag < GSAT + 0.5
                and sat[max(0, iy - m):iy + m + 1,
                        max(0, ix - m):ix + m + 1].any()
            )
            if not (is_sat or gmag < gsub):
                continue
        else:
            is_sat = 0
            if gmag >= GSAT:
                continue
            if not (-STAR_MARGIN < ix < nx + STAR_MARGIN
                    and -STAR_MARGIN < iy < ny + STAR_MARGIN):
                continue

        rows.append((
            float(gaia['ra'][k]), float(gaia['dec'][k]),
            xk, yk, gmag, ruwe, int(is_sat), int(on),
        ))

    stars = np.array(rows, dtype=_get_select_stars_dtype())

    non = int(stars['on_image'].sum())

    print(
        f'    census: {non} on-patch '
        f'({int(stars["is_sat"].sum())} saturated) + '
        f'{len(stars) - non} off-patch intruders'
    )

    return stars


def _get_select_stars_dtype():
    return [
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('x', 'f8'),
        ('y', 'f8'),
        ('G', 'f4'),
        ('ruwe', 'f4'),
        ('is_sat', 'i2'),
        ('on_image', 'i2'),
    ]


def own_component_ids(comps, ix, iy):
    """
    get labels of the flagged mask components at and around a star's integer
    position, sampled at +-2 px so a slightly displaced flag still counts as
    the star's own

    Parameters
    ----------
    comps: array
        The labeled SAT/INTRP component image
    ix, iy: int
        The star's integer pixel position

    Returns
    -------
    set of int component labels (possibly empty)
    """
    ny, nx = comps.shape

    ids = set()

    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            lid = comps[
                np.clip(iy + dy, 0, ny - 1),
                np.clip(ix + dx, 0, nx - 1)
            ]
            if lid > 0:
                ids.add(int(lid))

    return ids


def build_star_mask(stars, mask0, verbose=True):
    """
    Get floored magnitude-scaled circles at every census star plus the
    SAT/INTRP components of the saturated ones

    Parameters
    ----------
    stars: structured array
        The census from select_stars
    mask0: array
        The DM mask plane, for the SAT/INTRP components
    verbose: bool, optional
        Print the masked fraction

    Returns
    -------
    starmask, comps:
        The bool star mask and the labeled SAT/INTRP component
        image (used later for the per-star own-component masks)
    """
    from scipy import ndimage

    ny, nx = mask0.shape

    comps, _ = ndimage.label((mask0 & (DM_SAT | DM_INTRP)) != 0)

    starmask = np.zeros((ny, nx), dtype=bool)
    star_ids = set()

    for st in stars:
        xk, yk = float(st['x']), float(st['y'])
        ix, iy = int(round(xk)), int(round(yk))

        if st['on_image']:
            star_ids |= own_component_ids(comps, ix, iy)

        rad = circle_radius(float(st['G']))

        ir = int(np.ceil(rad))

        y0, y1 = max(0, iy - ir), min(ny, iy + ir + 1)
        x0, x1 = max(0, ix - ir), min(nx, ix + ir + 1)

        if y1 <= y0 or x1 <= x0:
            continue

        ly, lx = np.mgrid[y0:y1, x0:x1]

        starmask[y0:y1, x0:x1] |= (
            np.hypot(ly - yk, lx - xk) <= rad
        )

    if star_ids:
        starmask |= np.isin(comps, sorted(star_ids))

    if verbose:
        print(f'    star mask fraction {starmask.mean():.3f}')

    return starmask, comps


def select_template_stars(gaia, x, y, shape):
    """
    Get indices of the template-stack stars

    bright but unsaturated, astrometrically clean, and far enough from the
    edges for a full stamp; brightest TMPL_NSTAR kept.

    The faint limit starts at TMPL_GMAX and, on sparse fields yielding fewer
    than TMPL_MIN_CAND candidates, extends in 0.5 mag steps up to
    TMPL_GMAX_CAP.  Deep-coadd cores are still very high s/n there, and the
    extended-limit stack was validated on a real sparse field (agrees with the
    bright stack to < 1 percent inside the denoise core).  At the cap the
    template stars overlap the census; benign, since the stack is built from
    the pre-subtraction image

    Parameters
    ----------
    gaia: array with fields
        The gaia extract
    x, y: arrays
        Patch-frame pixel positions of the gaia stars
    shape: (ny, nx)
        The image shape, for the edge cut

    Returns
    -------
    indices into gaia, brightest first
    """
    ny, nx = shape
    half = TMPL_HALF

    gmag = gaia['phot_g_mean_mag']
    ruwe = gaia['ruwe']

    base = (
        (gmag > TMPL_GMIN)
        & np.isfinite(ruwe) & (ruwe < RUWE_MAX)
        & (x > half + 2) & (x < nx - half - 3)
        & (y > half + 2) & (y < ny - half - 3)
    )

    gmax_t = TMPL_GMAX

    while True:
        sel = np.where(base & (gmag < gmax_t))[0]
        if sel.size >= TMPL_MIN_CAND or gmax_t >= TMPL_GMAX_CAP:
            break
        gmax_t = min(gmax_t + 0.5, TMPL_GMAX_CAP)

    if gmax_t != TMPL_GMAX:
        print(
            f'    sparse field: template faint limit '
            f'extended to G < {gmax_t:.1f} '
            f'({sel.size} candidates)'
        )

    si = np.argsort(gmag[sel])
    return sel[si][:TMPL_NSTAR]


def stack_star_stamps(image, good, seg, x, y, sel,
                      min_stamps=TMPL_MIN_STAMPS):
    """
    Get core-normalized, sub-pixel-aligned median stack of the selected stars,
    point-symmetrized.

    Other detections are NaNed out of each stamp so neighbors cannot bias the
    stack

    Parameters
    ----------
    image: array
        The patch image
    good: array
        bool usable-pixel mask
    seg: array
        The field segmentation map, for neighbor masking
    x, y: arrays
        Patch-frame pixel positions of the gaia stars
    sel: array
        Indices of the template stars

    Returns
    -------
    tmpl, nstamp:
        The (2 TMPL_HALF + 1)^2 stack and the number of stamps
        used

    Raises
    ------
    RuntimeError when fewer than min_stamps stamps are usable
    (the caller degrades to the canonical-shape route or to
    mask-only handling)
    """
    from scipy import ndimage

    half = TMPL_HALF
    stamps = []

    for k in sel:
        cx, cy = float(x[k]), float(y[k])

        # nearest integer pixel of the star center
        icx, icy = int(round(cx)), int(round(cy))

        # stamp half size with a 2 px shift margin
        m = half + 2
        cut = np.s_[icy - m:icy + m + 1, icx - m:icx + m + 1]
        stamp = image[cut].copy()

        stamp[~good[cut]] = np.nan
        segcut = seg[cut]

        # NaN every detection except the star's own segment
        central = segcut[m, m]
        stamp[(segcut != 0) & (segcut != central)] = np.nan

        # shift the sub-pixel remainder so the star sits at the
        # exact stamp center, then trim the margin
        stamp = ndimage.shift(
            stamp, (icy - cy, icx - cx), order=1, cval=np.nan,
        )
        stamp = stamp[2:-2, 2:-2]

        gy, gx = np.mgrid[-half:half + 1, -half:half + 1]
        core = np.hypot(gy, gx) < 6
        amp = np.nansum(stamp[core])

        if not amp > 0:
            continue

        stamps.append(stamp / amp)

    if len(stamps) < min_stamps:
        raise RuntimeError(
            f'only {len(stamps)} usable '
            'template stamps'
        )

    tmpl = np.nanmedian(np.array(stamps), axis=0)
    tmpl[~np.isfinite(tmpl)] = 0.0

    # point-symmetrize so the template cannot bias a center
    tmpl = 0.5 * (tmpl + tmpl[::-1, ::-1])

    return tmpl, len(stamps)


def denoise_template(tmpl):
    """
    blend the stack into its azimuthal median profile beyond
    the high-s/n core, so residual stack noise multiplied by a
    bright star's amplitude cannot imprint on the image

    Parameters
    ----------
    tmpl: array
        The measured template stack

    Returns
    -------
    tmpl, prof:
        The blended template and the azimuthal median profile
        (indexed by integer radius)
    """
    half = TMPL_HALF

    gy, gx = np.mgrid[-half:half + 1, -half:half + 1]

    rr = np.hypot(gy, gx)          # radius of each pixel
    rbin = np.round(rr).astype(int)

    prof = np.zeros(rbin.max() + 1)

    for k in range(prof.size):
        w = rbin == k
        if w.any():
            prof[k] = np.median(tmpl[w])

    # light radial smoothing of the outer profile
    if prof.size > 12:
        sm = prof.copy()
        for k in range(10, prof.size - 1):
            sm[k] = np.median(prof[max(0, k - 2):k + 3])
        prof = sm

    # this is the blend. It is a pure stack inside r=12, pure profile beyond
    # r=18

    blend = np.clip((rr - 12.0) / 6.0, 0.0, 1.0)
    tmpl = (1.0 - blend) * tmpl + blend * prof[rbin]

    return tmpl, prof


def fit_halo_slope(prof, fallback_slope=HALO_SLOPE):
    """
    Fit the slope of the halo

    This is a joint power-law plus sky-pedestal fit

        a * r^s + c

    to the well-measured 22-50 px profile.  The per-stamp sky pedestals survive
    the median stack and bias a plain log-log fit shallow (and
    background-dependent); fitting the pedestal makes the slope
    background-independent.  A corrupted fit falls back to
    fallback_slope anchored to the same radii (the canonical
    band slope when known, else HALO_SLOPE)

    Parameters
    ----------
    prof: array
        The template azimuthal profile, indexed by integer
        radius
    fallback_slope: float, optional
        Slope to anchor when the free fit is corrupted

    Returns
    -------
    slope, ln_a, pedestal:
        The power-law slope, the log of the pedestal-free
        amplitude, and the fitted sky pedestal
    """
    rfit = np.arange(22, min(51, prof.size)).astype(float)
    pfit = prof[rfit.astype(int)]

    def linfit(s):
        basis = np.vstack([rfit ** s, np.ones(rfit.size)]).T
        coef, res, *_ = np.linalg.lstsq(basis, pfit, rcond=None)
        rss = float(res[0]) if res.size else float(
            np.sum((pfit - basis @ coef) ** 2),
        )
        return rss, coef[0], coef[1]

    best = None
    for s in np.arange(-6.0, -2.0, 0.01):
        rss, a, c = linfit(s)
        if best is None or rss < best[0]:
            best = (rss, s, a, c)

    _, slope, a, ped = best

    if not (-5.0 < slope < -2.5) or not a > 0:
        print(
            f'    template slope {slope:.2f} out of range, '
            f'using {fallback_slope}'
        )

        slope = fallback_slope
        _, a, ped = linfit(slope)

        if not a > 0:
            # the last resort is anchor the fallback law to the raw
            # profile medians, ignoring the pedestal
            wpos = pfit > 0
            ped = 0.0
            a = float(np.exp(np.median(
                np.log(pfit[wpos])
                - slope * np.log(rfit[wpos]),
            )))

    return slope, float(np.log(a)), float(ped)


def template_out_half(gmag):
    """
    Get the template stamp half size

    the analytic halo extends to TMPL_EXT_FACTOR times the mask radius (floored
    at TMPL_OUT_HALF, capped at TMPL_OUT_MAX) so the brightest stars' wings are
    subtracted beyond the old fixed edge

    Parameters
    ----------
    gmag: float
        Gaia G magnitude

    Returns
    -------
    int stamp half size in pixels
    """
    return int(np.clip(
        TMPL_EXT_FACTOR * circle_radius(gmag),
        TMPL_OUT_HALF, TMPL_OUT_MAX,
    ))


def extend_template_halo(
    tmpl,
    slope,
    ln_a,
    aur_slope,
    aur_amp,
    out_half=None,
    r_blend=40.0,
    r_join=44.0,
):
    """
    embed the measured template in a larger stamp whose outer
    halo is the fitted inner power law plus the scattering
    aureole, with a smooth junction over r_blend to r_join and
    an edge taper to zero

    Parameters
    ----------
    tmpl: array
        The measured (denoised, pedestal-free) template
    slope, ln_a: float
        The inner power law from fit_halo_slope
    aur_slope, aur_amp: float
        The aureole component from fit_aureole
    out_half: int, optional
        Output stamp half size; defaults to TMPL_OUT_HALF
    r_blend, r_join: float, optional
        The junction: measured template inside r_blend, the
        analytic halo beyond r_join, linear blend between.  The
        canonical sparse route pulls the junction inward so a
        few-stamp stack only has to carry the region the
        analytic law cannot describe

    Returns
    -------
    the (2 out_half + 1)^2 extended template
    """

    half = TMPL_HALF

    if out_half is None:
        out_half = TMPL_OUT_HALF

    big = np.zeros((2 * out_half + 1, 2 * out_half + 1))
    big[
        out_half - half:out_half + half + 1,
        out_half - half:out_half + half + 1
    ] = tmpl

    gy, gx = np.mgrid[
        -out_half:out_half + 1,
        -out_half:out_half + 1
    ]

    rr = np.hypot(gy, gx)          # radius in the big stamp
    rc = np.maximum(rr, 1.0)

    halo = (
        np.exp(ln_a) * rc ** slope
        + aur_amp * rc ** aur_slope
    )

    big[rr >= r_join] = halo[rr >= r_join]

    # linear blend from the measured template into the halo
    # over the junction
    jfrac = np.clip(
        (rr - r_blend) / (r_join - r_blend), 0.0, 1.0,
    )
    inner = rr < r_join

    big[inner] = (
        (1 - jfrac[inner]) * big[inner]
        + jfrac[inner] * halo[inner]
    )

    # taper the stamp edge to zero over the last ~5 px
    taper = np.clip((out_half - 2.0 - rr) / 5.0, 0.0, 1.0)
    return big * taper


def measure_wing_profiles(image, good, seg, stars):
    """
    flux-normalized azimuthal wing profiles of the mid-bright
    (AUR_GMIN <= G < AUR_GMAX) census stars, medianed across
    stars in common log-spaced radial bins outside each star's
    own mask.  Other detections and other stars' zones are
    excluded; each star is normalized by 10^(-0.4 G) so the
    curves overlay when the wings are self-similar

    Parameters
    ----------
    image: array
        The patch image
    good: array
        bool usable-pixel mask
    seg: array
        The field segmentation map
    stars: structured array
        The census from select_stars

    Returns
    -------
    rmid, med, count, nstars:
        Radial bin centers, the median stacked profile (NaN
        where fewer than 3 stars contribute), the per-bin star
        counts, and the number of stars measured
    """
    ny, nx = image.shape

    edges = np.unique(np.round(np.logspace(
        np.log10(40.0), np.log10(AUR_RMAX + 10), 12,
    )))

    rmid = 0.5 * (edges[:-1] + edges[1:])

    sel = (
        (stars['on_image'] == 1)
        & (stars['G'] >= AUR_GMIN) & (stars['G'] < AUR_GMAX)
    )

    profs = []
    for st in stars[sel]:
        gmag = float(st['G'])
        rad = float(circle_radius(gmag))
        fnorm = 10.0 ** (-0.4 * gmag)

        icx, icy = int(round(st['x'])), int(round(st['y']))
        m = int(AUR_RMAX + 20)

        x0, x1 = max(0, icx - m), min(nx, icx + m + 1)
        y0, y1 = max(0, icy - m), min(ny, icy + m + 1)

        gy, gx = np.mgrid[y0:y1, x0:x1]

        rr = np.hypot(gy - st['y'], gx - st['x'])
        segc = seg[y0:y1, x0:x1]

        # keep the star's own detection components
        own = set(np.unique(segc[rr <= rad + 2])) - {0}
        ok = good[y0:y1, x0:x1] & (
            (segc == 0) | np.isin(segc, sorted(own))
        )

        # other stars' mask circles out
        for ot in stars:
            if (ot['x'] == st['x']) and (ot['y'] == st['y']):
                continue
            orad = float(circle_radius(float(ot['G'])))
            dx = float(ot['x']) - icx
            dy = float(ot['y']) - icy

            if abs(dx) > m + orad or abs(dy) > m + orad:
                continue

            orr = np.hypot(
                gy - ot['y'], gx - ot['x'],
            )

            ok &= orr > orad + APOD_STARS

        prof = np.full(rmid.size, np.nan)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            w = (rr >= max(lo, rad + 2)) & (rr < hi) & ok
            if w.sum() > 100:
                prof[i] = np.median(
                    image[y0:y1, x0:x1][w],
                ) / fnorm

        if np.isfinite(prof).sum() >= 4:
            profs.append(prof)

    nstars = len(profs)
    if nstars == 0:
        return (
            rmid, np.full(rmid.size, np.nan),
            np.zeros(rmid.size, dtype=int), 0
        )

    profs = np.array(profs)
    count = np.sum(np.isfinite(profs), axis=0)

    med = np.full(rmid.size, np.nan)
    wc = count >= 3

    if wc.any():
        med[wc] = np.nanmedian(profs[:, wc], axis=0)

    return rmid, med, count, nstars


def fit_aureole(rmid, med, count, nstars, slope, ln_a):
    """
    fit the outer aureole component

    Fit in template units. the amplitude b and slope s_aur of b * r^s_aur,
    added to the inner power law beyond the measured stack.

    The fit is tiered for robustness on sparse fields:

    - tier 1 (>= AUR_MIN_STARS measured stars): both slope and
      amplitude are fit; the unknown flux scale of the
      measured cloud cancels in the ratio of the two linear
      basis coefficients, so no zero point is needed.  The
      slope scan is bounded AUR_SLOPE_SEP flatter than the
      inner wing (degeneracy guard; see the comment in the
      body).
    - tier 2 (fewer stars): the slope is fixed at AUR_SLOPE
      and only the amplitude is fit.
    - tier 3 (nothing measurable): the amplitude comes from
      continuity with the inner law at AUR_BREAK.

    A measurement always beats the continuity prior: fitted
    amplitudes are clipped to within AUR_AMP_GUARD of the
    continuity value (never replaced by it), and a fit that
    runs but finds a non-positive aureole is treated as a
    measurement of zero, clipping to the lower bound.  Both
    rules exist because the prior over-subtracts on images
    whose wings were partly absorbed by earlier backgrounds

    Parameters
    ----------
    rmid, med, count: arrays
        The stacked wing profile from measure_wing_profiles
    nstars: int
        Number of stars in the measured cloud
    slope, ln_a: float
        The inner power law from fit_halo_slope

    Returns
    -------
    aur_slope, aur_amp, tier:
        The aureole power law and the tier that produced it
    """
    a_in = float(np.exp(ln_a))

    def b_continuity(s_aur):
        return a_in * AUR_BREAK ** (slope - s_aur)

    usable = (
        np.isfinite(med) & (med > 0) & (count >= 3)
        & (rmid <= AUR_RMAX)
    )

    s_aur = AUR_SLOPE
    b = None

    if usable.sum() >= 3:

        def linfit(s):
            # k_in * (inner law) + bb * r^s.  In
            # template units the aureole amplitude is bb/k_in
            r = rmid[usable]
            basis = np.vstack([
                a_in * r ** slope, r ** s,
            ]).T
            coef, res, *_ = np.linalg.lstsq(
                basis, med[usable], rcond=None,
            )
            rss = float(res[0]) if res.size else float(
                np.sum((med[usable] - basis @ coef) ** 2),
            )
            return rss, coef[0], coef[1]

        # the aureole is by definition flatter than the inner
        # wing: bound the slope scan away from the wing slope.
        # Without the bound the two-component fit can go
        # degenerate on an aureole-free cloud: a near-parallel
        # "aureole" duplicating the wing, its amplitude
        # inflated by the k_in division, passing the guard
        # because continuity diverges as the slopes converge
        lo = max(AUR_SLOPE_MIN, slope + AUR_SLOPE_SEP)

        if nstars >= AUR_MIN_STARS and lo < AUR_SLOPE_MAX:
            tier = 1
            best = None
            for s in np.arange(
                lo, AUR_SLOPE_MAX + 1e-9, 0.02,
            ):
                fit = linfit(s)
                if best is None or fit[0] < best[0]:
                    best = (fit[0], s, fit[1], fit[2])

            _, s_fit, k_in, bb = best

            if k_in > 0:
                if bb > 0:
                    s_aur, b = float(s_fit), bb / k_in
                else:
                    # no aureole light in this
                    # image; the guard clips it to the lower
                    # bound below
                    b = 0.0
        else:
            tier = 2
            _, k_in, bb = linfit(AUR_SLOPE)

            if k_in > 0:
                b = bb / k_in if bb > 0 else 0.0

    if b is None:
        # nothing measurable (or a degenerate fit): the
        # continuity prior is all that is left
        tier = 3
        b = b_continuity(s_aur)
    else:
        bc = b_continuity(s_aur)
        lo, hi = bc / AUR_AMP_GUARD, bc * AUR_AMP_GUARD

        if not (lo < b < hi):
            bclip = float(np.clip(b, lo, hi))
            print(
                f'    aureole amp {b:.2e} clipped to '
                f'{bclip:.2e} ({AUR_AMP_GUARD}x guard '
                f'about continuity {bc:.2e})'
            )
            b = bclip

    return float(s_aur), float(b), tier


def psf_cube_fwhm(psfs, scale=0.2):
    """
    median half-max FWHM of a psf stamp cube, in arcsec

    Parameters
    ----------
    psfs: array
        (ncell, ny, nx) psf stamps; failed (all-zero) stamps
        are skipped
    scale: float, optional
        Pixel scale in arcsec

    Returns
    -------
    float FWHM in arcsec, or None when nothing is measurable
    """
    c = (psfs.shape[1] - 1) / 2
    gy, gx = np.mgrid[0:psfs.shape[1], 0:psfs.shape[2]]
    rbin = np.round(np.hypot(gy - c, gx - c)).astype(int)

    fwhms = []
    for p in psfs:
        peak = p[int(c), int(c)]
        if not peak > 0:
            continue
        prof = np.array([
            p[rbin == k].mean() for k in range(12)
        ])
        below = np.flatnonzero(prof < peak / 2)
        if below.size == 0:
            continue
        k = below[0]
        f1, f2 = prof[k - 1], prof[k]
        rhalf = (k - 1) + (f1 - peak / 2) / (f1 - f2)
        fwhms.append(2 * rhalf * scale)

    if len(fwhms) == 0:
        return None
    return float(np.median(fwhms))


def measure_coadd_fwhm(deep_coadd):
    """
    median PSF FWHM of the coadd in arcsec, for the canonical
    amplitude prior

    Uses the stored psf cube when present (file mode), else
    evaluates the psf on a small interior grid (butler mode).
    Returns None when neither is available; the prior then
    centers on the canonical mean with the wider raw scatter

    Parameters
    ----------
    deep_coadd: deep_coadd
        The coadd

    Returns
    -------
    float FWHM in arcsec, or None
    """
    psfs = getattr(deep_coadd, '_psfs', None)

    if psfs is None:
        psf = getattr(deep_coadd, 'psf', None)
        bbox = getattr(deep_coadd, 'bbox', None)
        if psf is None or bbox is None:
            return None
        kims = []
        xs = np.linspace(bbox.x.start, bbox.x.stop - 1, 5)[1:-1]
        ys = np.linspace(bbox.y.start, bbox.y.stop - 1, 5)[1:-1]
        for cy in ys:
            for cx in xs:
                try:
                    kims.append(psf.compute_kernel_image(
                        x=float(cx), y=float(cy)).array)
                except Exception:
                    continue
        if len(kims) == 0:
            return None
        psfs = np.array(kims)

    return psf_cube_fwhm(psfs)


def fit_canonical_amplitude(prof, canon, fwhm):
    """
    one-amplitude fit of a sparse-field profile against the
    canonical band shape, combined with the seeing prior

    The measured stack profile is fit as
    amp * canonical_law(r) + pedestal over the 22-50 px wing.
    The log amplitude is then combined with the seeing prior
    ln A = dlna_dfwhm * (fwhm - fwhm_ref) by inverse variance;
    with no usable measurement the prior stands alone

    Parameters
    ----------
    prof: array
        The template azimuthal profile, indexed by integer
        radius
    canon: dict
        The CANON entry for the band
    fwhm: float or None
        The coadd median PSF FWHM in arcsec; None widens the
        prior to the raw field-to-field scatter about zero

    Returns
    -------
    ln_amp, pedestal:
        The posterior log amplitude relative to the canonical
        shape, and the fitted sky pedestal
    """
    rfit = np.arange(22, min(51, prof.size)).astype(float)
    pfit = prof[rfit.astype(int)]

    law = np.exp(canon['ln_a']) * rfit ** canon['slope']
    basis = np.vstack([law, np.ones(rfit.size)]).T
    coef, _, _, _ = np.linalg.lstsq(basis, pfit, rcond=None)
    amp, ped = float(coef[0]), float(coef[1])

    resid = pfit - basis @ coef
    dof = max(rfit.size - 2, 1)
    cov = (np.linalg.inv(basis.T @ basis)
           * np.sum(resid ** 2) / dof)
    amp_err = float(np.sqrt(cov[0, 0]))

    if fwhm is not None:
        lna0 = canon['dlna_dfwhm'] * (fwhm - canon['fwhm_ref'])
        sig0 = canon['prior_sig']
    else:
        lna0 = 0.0
        sig0 = canon['raw_sig']

    # an informative measurement must be significantly
    # positive AND a detectable fraction of the profile it was
    # fit to (a degenerate flat profile yields a machine-noise
    # amplitude with a spuriously tiny error)
    floor = 1.0e-3 * float(np.median(np.abs(pfit)))

    if amp > 2.0 * amp_err and amp * law[0] > floor:
        # combine with the prior by inverse variance; the
        # error floor keeps an exact fit from claiming
        # unrealistic precision against per-field systematics
        lnam = np.log(amp)
        sigm = max(amp_err / amp, 0.05)
        ln_amp = (
            (lnam / sigm ** 2 + lna0 / sig0 ** 2)
            / (1.0 / sigm ** 2 + 1.0 / sig0 ** 2)
        )
    else:
        # wing lost in the noise: the prior stands alone (the
        # star wings are there whether measured or not)
        ln_amp = lna0

    return float(ln_amp), ped


def build_template(image, good, seg, gaia, x, y, stars,
                   band=None, fwhm=None):
    """
    build the empirical extended star template

    a median stack of bright unsaturated Gaia stars centered on their predicted
    positions (registration is a few hundredths of a pixel), core-normalized,
    point-symmetrized, denoised to the azimuthal profile, and extended with a
    two-component halo

    The two components are the inner turbulence-wing power law from the stack
    (joint pedestal-robust fit) plus the flatter scattering aureole measured
    from the mid-bright stars (tiered fallback on sparse fields).  The array is
    sized for the brightest census star; each star's stamp later windows it to
    its own extent

    Parameters
    ----------
    image: array
        The patch image (restored and pre-flattened)
    good: array
        bool usable-pixel mask
    seg: array
        The field segmentation map
    gaia: array with fields
        The gaia extract
    x, y: arrays
        Patch-frame pixel positions of the gaia stars
    stars: structured array
        The census, for the aureole measurement and the
        template sizing
    band: str, optional
        The band, keying the canonical constants; None
        disables the canonical machinery entirely
    fwhm: float, optional
        The coadd median PSF FWHM in arcsec, for the canonical
        amplitude prior (measure_coadd_fwhm)

    Returns
    -------
    the extended template array

    Raises
    ------
    RuntimeError from stack_star_stamps when the field is too
    barren even for the canonical route (fewer than
    CANON_MIN_STAMPS stamps, or no canonical constants for the
    band)
    """
    sel = select_template_stars(gaia, x, y, image.shape)
    canon = CANON.get(band)

    canonical = False
    try:
        tmpl, nstamp = stack_star_stamps(
            image, good, seg, x, y, sel,
        )
    except RuntimeError:
        if canon is None:
            raise
        # the canonical sparse route: too few stamps for the
        # free fits, but enough to anchor one amplitude
        # against the canonical band shape
        tmpl, nstamp = stack_star_stamps(
            image, good, seg, x, y, sel,
            min_stamps=CANON_MIN_STAMPS,
        )
        canonical = True

    tmpl, prof = denoise_template(tmpl)

    # the template array is sized for the brightest census
    # star (plus the stamp shift margin); each star's stamp
    # windows it to its own extent
    out_half = TMPL_OUT_HALF
    if stars.size > 0:
        out_half = max(
            template_out_half(float(g)) for g in stars['G']
        )

    out_half += 2

    if canonical:
        ln_amp, ped = fit_canonical_amplitude(prof, canon, fwhm)
        tmpl = tmpl - ped

        slope = canon['slope']
        ln_a = canon['ln_a'] + ln_amp
        # the aureole cannot be measured here (nor tabulated,
        # see CANON); use continuity at the break, the tier-3
        # convention, at the fitted amplitude
        aur_slope = AUR_SLOPE
        aur_amp = float(
            np.exp(ln_a) * AUR_BREAK ** (slope - aur_slope)
        )

        # the junction moves inward: a few-stamp stack only
        # has to carry the region the analytic law cannot
        # describe
        big = extend_template_halo(
            tmpl, slope, ln_a, aur_slope, aur_amp,
            out_half=out_half, r_blend=18.0, r_join=22.0,
        )
        fstr = f'{fwhm:.2f}"' if fwhm is not None else 'n/a'
        print(
            f'    canonical template: {nstamp} stamps, '
            f'ln amp {ln_amp:+.2f} (fwhm {fstr}), '
            f'pedestal {ped:.1e}, extent {out_half}'
        )
        return big

    slope, ln_a, ped = fit_halo_slope(
        prof,
        fallback_slope=canon['slope'] if canon else HALO_SLOPE,
    )

    # the sky pedestal is additive and must not scale with a
    # star's amplitude. remove it from the measured region
    # (the halo replaces everything beyond the junction)
    tmpl = tmpl - ped

    rmid, med, count, naur = measure_wing_profiles(
        image, good, seg, stars,
    )
    aur_slope, aur_amp, tier = fit_aureole(
        rmid, med, count, naur, slope, ln_a,
    )

    big = extend_template_halo(
        tmpl, slope, ln_a, aur_slope, aur_amp,
        out_half=out_half,
    )

    print(
        f'    template: {nstamp} stamps, '
        f'halo slope {slope:.2f}, '
        f'pedestal {ped:.1e}, '
        f'extent {out_half}'
    )
    print(
        f'    aureole: slope {aur_slope:.2f} '
        f'amp {aur_amp:.2e} (tier {tier}, {naur} stars)'
    )
    return big


def field_segmentation(image, good, sig):
    """
    1.5 sigma sep segmentation map over the whole patch

    Parameters
    ----------
    image: array
        The patch image
    good: array
        bool usable-pixel mask
    sig: float
        The pixel noise for the detection threshold

    Returns
    -------
    the segmentation map (0 = sky)
    """
    import sep

    sep.set_extract_pixstack(int(1.2e7))
    sep.set_sub_object_limit(10240)

    imf = np.ascontiguousarray(image, dtype='f4')

    _, seg = sep.extract(
        imf, 1.5, err=sig, mask=~good, segmentation_map=True,
    )

    return seg


def anchor_ring(
    rad_grid,
    usable,
    local_mask,
    on_image,
    rad,
    half,
):
    """
    Get the amplitude anchor

    This is a 2-10 px band just outside the star's own mask, which is exactly
    where the subtraction has to be right

    Parameters
    ----------
    rad_grid: array
        Radius of every stamp pixel from the star
    usable: array
        bool: good pixels with nonzero template
    local_mask: array
        The star's own mask (circle plus own components)
    on_image: bool
        Whether the star center is on the image
    rad: float
        The star's mask circle radius
    half: int
        The stamp half size (the ring must sit inside it)

    Returns
    -------
    bool ring mask, or None when no usable ring exists
    """
    from scipy import ndimage

    # the ring only needs distances <= 10 px from the star's
    # own mask, so the distance transform can run on the mask's
    # padded bounding box: everything outside the pad is
    # farther than 10 px by construction, and inside the box
    # the distances are identical to a full-stamp transform
    ring = np.zeros(local_mask.shape, dtype=bool)

    if local_mask.any():
        pad = 12
        rows = np.flatnonzero(local_mask.any(axis=1))
        cols = np.flatnonzero(local_mask.any(axis=0))
        sub = np.s_[
            max(rows[0] - pad, 0):rows[-1] + pad + 1,
            max(cols[0] - pad, 0):cols[-1] + pad + 1,
        ]

        dist = ndimage.distance_transform_edt(~local_mask[sub])
        ring[sub] = (dist >= 2) & (dist <= 10)
        ring &= (rad_grid < half - 10) & usable

    if ring.sum() >= 30:
        return ring

    if not on_image:
        # off-patch intruder whose mask is off-image. anchor on
        # the nearest visible annulus outside the circle
        # radius -- closer in is core territory and measures
        # garbage
        vis = usable & (rad_grid >= max(12.0, rad))

        if vis.any():
            ring = vis & (rad_grid < rad_grid[vis].min() + 10.0)
            if ring.sum() >= 30:
                return ring

    return None


def make_star_stamp(image, good, comps, tmpl, rr, st, si):
    """
    build the per-star working set

    In the image window, sub-pixel shifted template, and amplitude anchor ring.
    The stamp extent scales with brightness (template_out_half), windowing the
    shared template array centrally; a smaller window gets its own edge taper
    so the model cannot end in a hard step

    Parameters
    ----------
    image: array
        The patch image
    good: array
        bool usable-pixel mask
    comps: array
        The labeled SAT/INTRP component image
    tmpl: array
        The shared extended template
    rr: array
        Radius grid in the template frame, same shape as tmpl
    st: record
        One census star
    si: int
        The star's census index, recorded in the entry

    Returns
    -------
    dict with keys sl (image slice), T (shifted template
    cutout), ring (anchor mask or None), A (amplitude,
    initially 0), G, idx -- or None for stars whose stamp
    barely overlaps the image
    """
    from scipy import ndimage

    ny, nx = image.shape
    tmpl_half = (tmpl.shape[0] - 1) // 2

    xk, yk = float(st['x']), float(st['y'])
    gmag = float(st['G'])

    # the shift margin must fit inside the shared array
    half = min(template_out_half(gmag), tmpl_half - 2)
    ix, iy = int(round(xk)), int(round(yk))

    # central window of the shared template, with a 2 px
    # margin for the sub-pixel shift
    m = half + 2
    twin = np.s_[
        tmpl_half - m:tmpl_half + m + 1,
        tmpl_half - m:tmpl_half + m + 1
    ]

    # stamp window clipped to the image, with the matching
    # window into the (trimmed) template frame
    y0, x0 = iy - half, ix - half

    y0c, y1c = max(0, y0), min(ny, iy + half + 1)
    x0c, x1c = max(0, x0), min(nx, ix + half + 1)

    if y1c - y0c < 40 or x1c - x0c < 40:
        return None

    img_slice = np.s_[y0c:y1c, x0c:x1c]
    tmpl_slice = np.s_[y0c - y0:y1c - y0, x0c - x0:x1c - x0]

    # template shifted to the star's sub-pixel position;
    # linear interpolation as in stack_star_stamps. outside the
    # mask the profile is smooth enough that the difference
    # from cubic is far below the noise, and it avoids the
    # spline prefilter on these large windows
    tmpl_shifted = ndimage.shift(
        tmpl[twin], (yk - iy, xk - ix), order=1, cval=0.0,
    )[2:-2, 2:-2][tmpl_slice]

    # radius of each stamp pixel from the star
    rad_grid = rr[twin][2:-2, 2:-2][tmpl_slice]

    # window edge taper, as extend_template_halo applies at
    # the full array edge
    tmpl_shifted = tmpl_shifted * np.clip(
        (half - 2.0 - rad_grid) / 5.0, 0.0, 1.0,
    )
    usable = good[img_slice] & (tmpl_shifted > 0)

    # the star's own mask, floored circle plus its own flagged
    # components (neighbors' components must not steer the ring)
    rad = circle_radius(gmag)
    local_mask = rad_grid <= rad

    if st['on_image']:
        own = own_component_ids(comps, ix, iy)
        if own:
            local_mask |= np.isin(comps[img_slice], sorted(own))

    ring = anchor_ring(
        rad_grid, usable, local_mask, st['on_image'], rad,
        half,
    )

    return {
        'sl': img_slice,
        'T': tmpl_shifted,
        'ring': ring,
        'A': 0.0,
        'G': gmag,
        'idx': si,
    }


def fit_flux_zeropoint(image, slist):
    """
    fit the fixed-slope flux relation A = 10^(zp - 0.4 G) from
    the bright-star ring amplitudes

    Parameters
    ----------
    image: array
        The patch image
    slist: list of dict
        The per-star work list from make_star_stamp

    Returns
    -------
    zp: float, or None when fewer than 3 stars contribute
    """
    zps = []

    for st in slist:
        if st['ring'] is not None and st['G'] < 15.5:
            # single-star ring amplitude estimate
            a0 = float(np.median(
                image[st['sl']][st['ring']]
                / st['T'][st['ring']],
            ))

            if a0 > 0:
                zps.append(np.log10(a0) + 0.4 * st['G'])

    zp = float(np.median(zps)) if len(zps) >= 3 else None

    if zp is not None:
        print(f'    flux zero point {zp:.2f} ({len(zps)} stars)')

    return zp


def solve_joint_amplitudes(image, slist, zp):
    """
    solve the amplitudes in NPASS Gauss-Seidel passes

    each star's ring amplitude is measured on the data minus the other stars'
    current models, so close pairs do not double count each other's halos.

    The flux relation supplies amplitudes where the ring failed and CLIPS ring
    amplitudes to [1/3, 3] times the relation. the cap kills neighbor-halo
    explosions, the floor guards pathological anchors.

    The floor clips rather than replacing with the relation outright.  low
    anchors are common (the wing-to-flux ratio is color-dependent, especially
    in z) and raising them to the relation over-subtracts

    Parameters
    ----------
    image: array
        The patch image
    slist: list of dict
        The per-star work list; each entry's 'A' is updated in
        place with the fitted amplitude
    zp: float or None
        The flux zero point from fit_flux_zeropoint

    Returns
    -------
    the summed model image, same shape as image
    """

    model = np.zeros_like(image)

    for _ in range(NPASS):
        for st in slist:
            sl = st['sl']

            # ap: amplitude predicted by the flux relation
            ap = None
            if zp is not None:
                ap = 10 ** (zp - 0.4 * st['G'])

            if st['ring'] is None:
                if ap is None or st['A'] > 0:
                    continue
                amp = ap
            else:
                # data with this star's own model restored
                resid = (
                    image[sl] - model[sl] + st['A'] * st['T']
                )
                ring = st['ring']
                amp = max(float(np.median(
                    resid[ring] / st['T'][ring],
                )), 0.0)
                if ap is not None:
                    amp = min(max(amp, ap / 3.0), 3.0 * ap)

            model[sl] += (amp - st['A']) * st['T']
            st['A'] = amp

    return model


def subtract_stars(image, var, mask0, gaia, x, y, stars, comps,
                   band=None, fwhm=None):
    """
    subtract every census star, modifying image in place

    build the template, anchor each amplitude on the robust median of
    data/template in a 2-10 px band just outside the star's own mask, refine
    jointly, guarded by the fixed-slope flux relation.  A field too barren for
    a template degrades to mask-only handling (nothing subtracted, empty work
    list returned)

    The whole solve runs on a working copy referenced to the undetected-pixel
    median: sky estimators track the mode of the pixel distribution while the
    unresolved-source carpet skews the median of blank pixels above it, so
    without the reference the anchor rings measure wing plus that ambient
    level, and an amplitude that absorbs the ambient over-subtracts everywhere
    the wing declines (a negative collar just outside every mask, worst in
    the red bands)

    Parameters
    ----------
    image: array
        The patch image (restored and pre-flattened)
    var: array
        The variance plane, for the good mask and the
        detection threshold
    mask0: array
        The DM mask plane
    gaia: array with fields
        The gaia extract
    x, y: arrays
        Patch-frame pixel positions of the gaia stars
    stars: structured array
        The census from select_stars
    comps: array
        The labeled SAT/INTRP component image from
        build_star_mask
    band: str, optional
        The band, enabling the canonical sparse-field template
        route (build_template)
    fwhm: float, optional
        The coadd median PSF FWHM in arcsec, for the canonical
        amplitude prior

    Returns
    -------
    slist: list of dict
        The per-star work list with fitted amplitudes; empty
        on the mask-only fallback
    """

    good = (
        np.isfinite(var) & (var > 0)
        & ((mask0 & DM_OUT) == 0)
    )

    # median pixel noise, for the detection threshold
    sig = float(np.sqrt(np.median(var[good])))
    seg = field_segmentation(image, good, sig)

    # the ambient sky reference (see the docstring): the
    # template wings, the aureole cloud, and every ring median
    # are measured on an ambient-free working copy; the image
    # itself is only touched by the final model subtraction
    amb = float(np.median(image[good & (seg == 0)]))
    print(f'    ambient reference {amb / sig:+.4f} sigma')
    work = image - amb

    try:
        tmpl = build_template(
            work, good, seg, gaia, x, y, stars,
            band=band, fwhm=fwhm,
        )
    except RuntimeError as err:
        # mask-only fallback. a patch too barren to build a
        # template even at the extended faint limit has next
        # to nothing worth subtracting.  Keep the masking and
        # taper (an empty work list leaves every amplitude 0)
        print(
            f'    WARNING: no star template ({err}); '
            'masking without subtraction'
        )
        return []

    tmpl_half = (tmpl.shape[0] - 1) // 2
    gy, gx = np.mgrid[
        -tmpl_half:tmpl_half + 1,
        -tmpl_half:tmpl_half + 1
    ]

    rr = np.hypot(gy, gx)  # radius grid in the template frame

    slist = []

    for si, st in enumerate(stars):
        entry = make_star_stamp(
            work, good, comps, tmpl, rr, st, si,
        )
        if entry is not None:
            slist.append(entry)

    zp = fit_flux_zeropoint(work, slist)
    model = solve_joint_amplitudes(work, slist, zp)

    image -= model
    namp = int(sum(st['A'] > 0 for st in slist))
    print(f'    subtracted {namp} of {len(slist)} stars')

    return slist


def make_star_table(stars, slist):
    """
    build the gaia_stars output table

    the census with the fitted per-band amplitude filled in for the stars that
    received stamps

    Parameters
    ----------
    stars: structured array
        The census from select_stars
    slist: list of dict
        The per-star work list from subtract_stars

    Returns
    -------
    structured array with the census fields plus A
    """
    star_table = np.zeros(len(stars), dtype=_get_star_table_dtype())
    for name in (
        'ra',
        'dec',
        'x',
        'y',
        'G',
        'ruwe',
        'is_sat',
        'on_image',
    ):
        star_table[name] = stars[name]

    for st in slist:
        star_table['A'][st['idx']] = st['A']

    return star_table


def _get_star_table_dtype():
    return [
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('x', 'f8'),
        ('y', 'f8'),
        ('G', 'f4'),
        ('ruwe', 'f4'),
        ('is_sat', 'i2'),
        ('on_image', 'i2'),
        ('A', 'f8'),
    ]


def restore_object_background(deep_coadd, dbright):
    """
    add the stored 'object' background model back to the image around the
    bright stars.

    That model absorbs star wings and scattered light; restoring it there (an
    exact undo, weight 1 within RESTORE_RAD of the bright-star masks tapering
    to 0 over RESTORE_TAPER) returns the wing light to the image so the
    template subtraction can remove it as star flux.  The preliminary and final
    backgrounds re-handle any true sky the model carried.

    A coadd without a stored model (the file-backed test path)
    is skipped with a warning

    Parameters
    ----------
    deep_coadd: deep_coadd
        The coadd; its image is modified in place
    dbright: array
        Distance from the bright-star masks, patch frame
    """
    bgs = getattr(deep_coadd, 'backgrounds', None)

    if bgs is None or 'object' not in bgs:
        print(
            '    WARNING: no stored object background '
            'model; restoration skipped'
        )
        return

    model = bgs['object'].field.render(
        deep_coadd.bbox, dtype=deep_coadd.image.array.dtype,
    ).quantity.value

    w = np.clip(
        (RESTORE_RAD + RESTORE_TAPER - dbright)
        / RESTORE_TAPER,
        0.0, 1.0,
    )

    deep_coadd.image.array[:, :] += w * model

    print(
        f'    restored object model over '
        f'{(w > 0).mean():.3f} of the patch'
    )


def preliminary_background(deep_coadd, dstar, dbright):
    """
    mask-aware preliminary background on the restored image.

    This occurs before the template build and subtraction. It flattens the sky
    that the template stack and the amplitude anchors sit on.  Star zones
    (PRE_GROW everywhere, PRE_GROW_BRIGHT around the bright stars) and
    detections are excluded from the boxes, so the model cannot chase the wing
    light the restoration just returned

    Parameters
    ----------
    deep_coadd: deep_coadd
        The coadd; its image is modified in place
    dstar: array
        Distance from the full star mask, patch frame
    dbright: array
        Distance from the bright-star masks, patch frame
    """
    import sep

    image = deep_coadd.image.array
    var = deep_coadd.variance.array
    mask0 = deep_coadd.mask.array[:, :, 0]

    good = (
        np.isfinite(var) & (var > 0)
        & ((mask0 & DM_OUT) == 0)
    )

    sig = float(np.sqrt(np.median(var[good])))
    seg = field_segmentation(image, good, sig)

    bad = (
        ~good | (seg > 0)
        | (dstar < PRE_GROW) | (dbright < PRE_GROW_BRIGHT)
    )

    bkg = sep.Background(
        np.ascontiguousarray(image, dtype='f4'),
        mask=bad, bw=PRE_BW, bh=PRE_BW,
    )

    image[:, :] -= bkg.back()

    print(
        f'    preliminary background: globalback '
        f'{bkg.globalback:.3f}'
    )


def handle_stars(
    deep_coadd, wcs, gaia, gsub=GSUB,
    subtract=True,
):
    """
    the getimages-time star handling, modifying the image in place

    Do the census and star mask; then, when subtracting, the local restoration
    of the stored object background around the bright stars, the mask-aware
    preliminary background, and the two-scale template subtraction

    Parameters
    ----------
    deep_coadd: deep_coadd
        The coadd; its image is modified in place
    wcs: ButlerWcs or FileWcs
        For the gaia pixel positions
    gaia: array with fields
        The gaia extract
    gsub: float, optional
        Census depth
    subtract: bool, optional
        False = mask-only handling (no restoration, no
        preliminary background, no subtraction, no table)

    Returns
    -------
    starmask, star_table, dstar:
        The bool star mask, the census table with fitted
        amplitudes (None when not subtracting), and the
        distance transform off the mask (for the background
        margin and the taper)
    """
    from scipy import ndimage

    mask0 = deep_coadd.mask.array[:, :, 0]

    x, y = gaia_pixel_positions(gaia, wcs, deep_coadd.bbox)

    stars = select_stars(gaia, x, y, mask0, gsub=gsub)
    starmask, comps = build_star_mask(stars, mask0)

    dstar = ndimage.distance_transform_edt(~starmask)

    star_table = None
    if subtract:
        bright = stars[stars['G'] < RESTORE_GMAX]

        if bright.size > 0:
            bsm, _ = build_star_mask(
                bright, mask0, verbose=False,
            )
            dbright = ndimage.distance_transform_edt(~bsm)
            restore_object_background(deep_coadd, dbright)
        else:
            # no bright stars: nothing to restore, and the
            # pre-pass needs no extra exclusion
            dbright = np.full(mask0.shape, np.inf)

        preliminary_background(deep_coadd, dstar, dbright)

        # for the canonical sparse-field route: the band keys
        # the canonical constants and the psf fwhm centers the
        # amplitude prior
        fwhm = measure_coadd_fwhm(deep_coadd)

        slist = subtract_stars(
            deep_coadd.image.array,
            deep_coadd.variance.array,
            mask0, gaia, x, y, stars, comps,
            band=getattr(deep_coadd, 'band', None),
            fwhm=fwhm,
        )

        star_table = make_star_table(stars, slist)

    return starmask, star_table, dstar


def apply_star_taper(deep_coadd, dstar, width=APOD_STARS):
    """
    apodize the star-mask regions

    This occurs after any background determination, in both the image and the
    noise realization so they stay statistically matched

    Parameters
    ----------
    deep_coadd: deep_coadd
        The coadd; image and noise are modified in place
    dstar: array
        Distance from the star mask, patch frame
    width: float, optional
        The taper width in pixels
    """
    from .apodize import taper_from_distance

    taper = taper_from_distance(dstar, width)
    deep_coadd.image.array[:, :] *= taper
    deep_coadd.noise_realizations[0].array[:, :] *= taper


def make_starmask_plane(starmask, dstar, apod):
    """
    build the three-valued starmask output plane

    0 = clear,

    1 = taper zone (attenuated -- masked for any measurement,
    smooth enough for FFTs)

    2 = star mask (zeroed when apod > 0)

    Parameters
    ----------
    starmask: array
        The bool star mask
    dstar: array
        Distance from the star mask, patch frame
    apod: float
        The taper width actually applied (0 = no taper zone)

    Returns
    -------
    u1 plane, same shape as starmask
    """
    plane = np.zeros(starmask.shape, dtype='u1')

    if apod > 0:
        plane[(dstar > 0) & (dstar < apod)] = 1

    plane[starmask] = 2

    return plane
