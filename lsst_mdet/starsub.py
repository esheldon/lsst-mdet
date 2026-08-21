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
MINRAD = 20.0       # circle floor: subtracted cores never show
STAR_MARGIN = 210   # off-patch stars whose wings still intrude
GSAT = 15.2         # G saturation threshold of these coadds
GSUB = 19.0         # subtract stars brighter than this
RUWE_MAX = 1.4      # template-star astrometric-quality guard
BG_GROW = 12        # extra star-mask margin for the background
APOD_STARS = 12.0   # taper width outside the star mask

# empirical extended star template
TMPL_HALF = 50       # measured stamp half size
TMPL_OUT_HALF = 250  # power-law halo extension half size
TMPL_NSTAR = 60
TMPL_GMIN = GSAT + 0.3  # template stars: bright but unsaturated
TMPL_GMAX = 17.5        # preferred faint limit
TMPL_GMAX_CAP = 19.0    # adaptive faint-limit cap (census depth)
TMPL_MIN_CAND = 20      # extend the faint limit below this
TMPL_MIN_STAMPS = 10    # hard minimum usable stamps
HALO_SLOPE = -3.7   # optics halo power law, for corrupted fits

NPASS = 3           # joint amplitude passes


def circle_radius(gmag):
    return np.minimum(
        np.maximum(
            MASK_R15 * 10 ** (MASK_SLOPE * (15.0 - gmag)),
            MINRAD,
        ),
        MASK_RMAX,
    )


def select_stars(gaia, x, y, mask0, gsub=GSUB):
    """
    the subtract-and-mask census: on-patch stars that are
    saturated or brighter than gsub (no RUWE guard -- Gaia at
    these depths is essentially pure point sources, and
    high-RUWE binaries are still stars we want gone), plus
    off-patch intruders bright enough for their wings to
    reach in.  Returns a structured array sorted brightest
    first
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
    stars = np.array(rows, dtype=[
        ('ra', 'f8'), ('dec', 'f8'),
        ('x', 'f8'), ('y', 'f8'), ('G', 'f4'), ('ruwe', 'f4'),
        ('is_sat', 'i2'), ('on_image', 'i2'),
    ])
    non = int(stars['on_image'].sum())
    print(f'    census: {non} on-patch '
          f'({int(stars["is_sat"].sum())} saturated) + '
          f'{len(stars) - non} off-patch intruders')
    return stars


def own_component_ids(comps, ix, iy):
    """
    labels of the flagged mask components at and around a
    star's integer position (comps is the labeled SAT/INTRP
    image); sampled at +-2 px so a slightly displaced flag
    still counts as the star's own
    """
    ny, nx = comps.shape
    ids = set()
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            lid = comps[np.clip(iy + dy, 0, ny - 1),
                        np.clip(ix + dx, 0, nx - 1)]
            if lid > 0:
                ids.add(int(lid))
    return ids


def build_star_mask(stars, mask0):
    """
    floored magnitude-scaled circles at every census star plus
    the SAT/INTRP components of the saturated ones
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
    print(f'    star mask fraction {starmask.mean():.3f}')
    return starmask, comps


def select_template_stars(gaia, x, y, shape):
    """
    indices of the template-stack stars: bright but
    unsaturated, astrometrically clean, and far enough from the
    edges for a full stamp; brightest TMPL_NSTAR kept.

    The faint limit starts at TMPL_GMAX and, on sparse fields
    yielding fewer than TMPL_MIN_CAND candidates, extends in
    0.5 mag steps up to TMPL_GMAX_CAP.  Deep-coadd cores are
    still very high s/n there, and the extended-limit stack
    was validated on a real sparse field (agrees with the
    bright stack to < 1 percent inside the denoise core).  At
    the cap the template stars overlap the census; benign,
    since the stack is built from the pre-subtraction image
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
        print(f'    sparse field: template faint limit '
              f'extended to G < {gmax_t:.1f} '
              f'({sel.size} candidates)')
    return sel[np.argsort(gmag[sel])][:TMPL_NSTAR]


def stack_star_stamps(image, good, seg, x, y, sel):
    """
    core-normalized, sub-pixel-aligned median stack of the
    selected stars, point-symmetrized.  Other detections are
    NaNed out of each stamp so neighbors cannot bias the stack
    """
    from scipy import ndimage

    half = TMPL_HALF
    stamps = []
    for k in sel:
        cx, cy = float(x[k]), float(y[k])
        # icx, icy: nearest integer pixel of the star center
        icx, icy = int(round(cx)), int(round(cy))
        # m: stamp half size with a 2 px shift margin
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
    if len(stamps) < TMPL_MIN_STAMPS:
        raise RuntimeError(f'only {len(stamps)} usable '
                           'template stamps')

    tmpl = np.nanmedian(np.array(stamps), axis=0)
    tmpl[~np.isfinite(tmpl)] = 0.0
    # point-symmetrize so the template cannot bias a center
    tmpl = 0.5 * (tmpl + tmpl[::-1, ::-1])
    return tmpl, len(stamps)


def denoise_template(tmpl):
    """
    blend the stack into its azimuthal median profile beyond
    the high-s/n core, so residual stack noise multiplied by a
    bright star's amplitude cannot imprint on the image.
    Returns the blended template and the radial profile
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
    # blend: pure stack inside r=12, pure profile beyond r=18
    blend = np.clip((rr - 12.0) / 6.0, 0.0, 1.0)
    tmpl = (1.0 - blend) * tmpl + blend * prof[rbin]
    return tmpl, prof


def fit_halo_slope(prof):
    """
    power-law fit to the well-measured 25-44 px profile;
    crowding can corrupt it, in which case fall back to the
    optics slope anchored to the same radii
    """
    rfit = np.arange(25, min(45, prof.size))
    pfit = prof[rfit]
    wpos = pfit > 0                # log needs positive values
    slope, ln_a = np.polyfit(
        np.log(rfit[wpos]), np.log(pfit[wpos]), 1,
    )
    if not (-5.0 < slope < -2.5):
        print(f'    template slope {slope:.2f} out of range, '
              f'using {HALO_SLOPE}')
        slope = HALO_SLOPE
        ln_a = np.median(
            np.log(pfit[wpos]) - slope * np.log(rfit[wpos]),
        )
    return slope, ln_a


def extend_template_halo(tmpl, slope, ln_a):
    """
    embed the measured template in a larger stamp whose outer
    halo is the fitted power law, with a smooth junction at
    40-44 px and an edge taper to zero
    """
    half = TMPL_HALF
    out_half = TMPL_OUT_HALF
    big = np.zeros((2 * out_half + 1, 2 * out_half + 1))
    big[out_half - half:out_half + half + 1,
        out_half - half:out_half + half + 1] = tmpl
    gy, gx = np.mgrid[-out_half:out_half + 1,
                      -out_half:out_half + 1]
    rr = np.hypot(gy, gx)          # radius in the big stamp
    halo = np.exp(ln_a) * np.maximum(rr, 1.0) ** slope
    big[rr >= 44.0] = halo[rr >= 44.0]
    # linear blend from the measured template into the halo
    # over the 40-44 px junction
    jfrac = np.clip((rr - 40.0) / 4.0, 0.0, 1.0)
    inner = rr < 44.0
    big[inner] = (
        (1 - jfrac[inner]) * big[inner]
        + jfrac[inner] * halo[inner]
    )
    # taper the stamp edge to zero over the last ~5 px
    taper = np.clip((out_half - 2.0 - rr) / 5.0, 0.0, 1.0)
    return big * taper


def build_template(image, good, seg, gaia, x, y):
    """
    empirical extended star template: median stack of bright
    unsaturated Gaia stars centered on their predicted
    positions (registration is a few hundredths of a pixel),
    core-normalized, point-symmetrized, denoised to the
    azimuthal profile, and extended with a power-law halo
    """
    sel = select_template_stars(gaia, x, y, image.shape)
    tmpl, nstamp = stack_star_stamps(image, good, seg, x, y, sel)
    tmpl, prof = denoise_template(tmpl)
    slope, ln_a = fit_halo_slope(prof)
    big = extend_template_halo(tmpl, slope, ln_a)
    print(f'    template: {nstamp} stamps, '
          f'halo slope {slope:.2f}')
    return big


def field_segmentation(image, good, sig):
    """1.5 sigma sep segmentation map over the whole patch"""
    import sep

    sep.set_extract_pixstack(int(1.2e7))
    sep.set_sub_object_limit(10240)
    imf = np.ascontiguousarray(image, dtype='f4')
    _, seg = sep.extract(
        imf, 1.5, err=sig, mask=~good, segmentation_map=True,
    )
    return seg


def anchor_ring(rad_grid, usable, local_mask, on_image, rad):
    """
    the amplitude anchor: a 2-10 px band just outside the
    star's own mask, which is exactly where the subtraction
    has to be right.  Returns None when no usable ring exists
    """
    from scipy import ndimage

    half = TMPL_OUT_HALF
    # distance of every stamp pixel from the star's own mask
    dist = ndimage.distance_transform_edt(~local_mask)
    ring = (
        (dist >= 2) & (dist <= 10)
        & (rad_grid < half - 10) & usable
    )
    if ring.sum() >= 30:
        return ring
    if not on_image:
        # off-patch intruder whose mask is off-image: anchor on
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
    per-star working set: image window, sub-pixel-shifted
    template, and amplitude anchor ring.  Returns None for
    stars whose stamp barely overlaps the image
    """
    from scipy import ndimage

    ny, nx = image.shape
    half = TMPL_OUT_HALF
    xk, yk = float(st['x']), float(st['y'])
    gmag = float(st['G'])
    ix, iy = int(round(xk)), int(round(yk))

    # stamp window clipped to the image, with the matching
    # window into the template frame
    y0, x0 = iy - half, ix - half
    y0c, y1c = max(0, y0), min(ny, iy + half + 1)
    x0c, x1c = max(0, x0), min(nx, ix + half + 1)
    if y1c - y0c < 40 or x1c - x0c < 40:
        return None
    img_slice = np.s_[y0c:y1c, x0c:x1c]
    tmpl_slice = np.s_[y0c - y0:y1c - y0, x0c - x0:x1c - x0]

    # template shifted to the star's sub-pixel position
    tmpl_shifted = ndimage.shift(
        tmpl, (yk - iy, xk - ix), order=3, cval=0.0,
    )[tmpl_slice]
    # radius of each stamp pixel from the star
    rad_grid = rr[tmpl_slice]
    usable = good[img_slice] & (tmpl_shifted > 0)

    # the star's own mask: floored circle plus its own flagged
    # components (neighbors' components must not steer the ring)
    rad = circle_radius(gmag)
    local_mask = rad_grid <= rad
    if st['on_image']:
        own = own_component_ids(comps, ix, iy)
        if own:
            local_mask |= np.isin(comps[img_slice], sorted(own))

    ring = anchor_ring(
        rad_grid, usable, local_mask, st['on_image'], rad,
    )
    return {
        'sl': img_slice, 'T': tmpl_shifted, 'ring': ring,
        'A': 0.0, 'G': gmag, 'idx': si,
    }


def fit_flux_zeropoint(image, slist):
    """
    fixed-slope flux relation A = 10^(zp - 0.4 G) from the
    bright-star ring amplitudes; None if too few stars
    """
    zps = []
    for st in slist:
        if st['ring'] is not None and st['G'] < 15.5:
            # a0: single-star ring amplitude estimate
            a0 = float(np.median(
                image[st['sl']][st['ring']]
                / st['T'][st['ring']],
            ))
            if a0 > 0:
                zps.append(np.log10(a0) + 0.4 * st['G'])
    zp = float(np.median(zps)) if len(zps) >= 3 else None
    if zp is not None:
        print(f'    flux zero point {zp:.2f} '
              f'({len(zps)} stars)')
    return zp


def solve_joint_amplitudes(image, slist, zp):
    """
    NPASS Gauss-Seidel passes: each star's ring amplitude is
    measured on the data minus the other stars' current
    models, so close pairs do not double count each other's
    halos.  The flux relation supplies amplitudes where the
    ring failed, caps ring amplitudes that land on a
    neighbor's halo (3x), and replaces anchors that measure
    anomalously low (below 1/3).  Returns the summed model
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
                    if amp < ap / 3.0:
                        # failed or background-absorbed anchor
                        # (a smooth background eats the wing
                        # signal at a bright star's circle
                        # edge); the relation is the better
                        # estimate
                        amp = ap
                    else:
                        amp = min(amp, 3.0 * ap)
            model[sl] += (amp - st['A']) * st['T']
            st['A'] = amp
    return model


def subtract_stars(image, var, mask0, gaia, x, y, stars, comps):
    """
    subtract every census star: amplitude from the robust
    median of data/template in a 2-10 px band just outside the
    star's own mask, refined jointly, guarded by the
    fixed-slope flux relation.  Modifies image in place;
    returns the per-star work list with fitted amplitudes
    """
    good = (
        np.isfinite(var) & (var > 0)
        & ((mask0 & DM_OUT) == 0)
    )
    # sig: median pixel noise, for the detection threshold
    sig = float(np.sqrt(np.median(var[good])))
    seg = field_segmentation(image, good, sig)

    try:
        tmpl = build_template(image, good, seg, gaia, x, y)
    except RuntimeError as err:
        # mask-only fallback: a patch too barren to build a
        # template even at the extended faint limit has next
        # to nothing worth subtracting.  Keep the masking and
        # taper (an empty work list leaves every amplitude 0)
        print(f'    WARNING: no star template ({err}); '
              'masking without subtraction')
        return []
    half = TMPL_OUT_HALF
    gy, gx = np.mgrid[-half:half + 1, -half:half + 1]
    rr = np.hypot(gy, gx)  # radius grid in the template frame

    slist = []
    for si, st in enumerate(stars):
        entry = make_star_stamp(
            image, good, comps, tmpl, rr, st, si,
        )
        if entry is not None:
            slist.append(entry)

    zp = fit_flux_zeropoint(image, slist)
    model = solve_joint_amplitudes(image, slist, zp)

    image -= model
    namp = int(sum(st['A'] > 0 for st in slist))
    print(f'    subtracted {namp} of {len(slist)} stars')
    return slist


def make_star_table(stars, slist):
    """
    the gaia_stars output table: the census with the fitted
    per-band amplitude filled in for the stars that received
    stamps
    """
    star_table = np.zeros(len(stars), dtype=[
        ('ra', 'f8'), ('dec', 'f8'),
        ('x', 'f8'), ('y', 'f8'), ('G', 'f4'),
        ('ruwe', 'f4'), ('is_sat', 'i2'),
        ('on_image', 'i2'), ('A', 'f8'),
    ])
    for name in ('ra', 'dec', 'x', 'y', 'G', 'ruwe',
                 'is_sat', 'on_image'):
        star_table[name] = stars[name]
    for st in slist:
        star_table['A'][st['idx']] = st['A']
    return star_table


def handle_stars(deep_coadd, wcs, gaia, gsub=GSUB,
                 subtract=True):
    """
    the getimages-time star handling: census, star mask, and
    (optionally) template subtraction, modifying the image in
    place.  Returns (starmask, star_table, dstar) with dstar
    the distance transform off the mask (None when there is no
    mask), for the background margin and the taper
    """
    from scipy import ndimage

    mask0 = deep_coadd.mask.array[:, :, 0]
    x, y = gaia_pixel_positions(gaia, wcs, deep_coadd.bbox)
    stars = select_stars(gaia, x, y, mask0, gsub=gsub)
    starmask, comps = build_star_mask(stars, mask0)

    star_table = None
    if subtract:
        slist = subtract_stars(
            deep_coadd.image.array,
            deep_coadd.variance.array,
            mask0, gaia, x, y, stars, comps,
        )
        star_table = make_star_table(stars, slist)

    dstar = ndimage.distance_transform_edt(~starmask)
    return starmask, star_table, dstar


def apply_star_taper(deep_coadd, dstar, width=APOD_STARS):
    """
    apodize the star-mask regions, AFTER any background
    determination, in both the image and the noise realization
    so they stay statistically matched
    """
    from .apodize import taper_from_distance

    taper = taper_from_distance(dstar, width)
    deep_coadd.image.array[:, :] *= taper
    deep_coadd.noise_realizations[0].array[:, :] *= taper


def make_starmask_plane(starmask, dstar, apod):
    """
    the three-valued starmask output plane: 0 = clear, 1 =
    taper zone (attenuated -- masked for any measurement,
    smooth enough for FFTs), 2 = star mask (zeroed when
    apod > 0)
    """
    plane = np.zeros(starmask.shape, dtype='u1')
    if apod > 0:
        plane[(dstar > 0) & (dstar < apod)] = 1
    plane[starmask] = 2
    return plane
