import os
import numpy as np
from lsst.daf.butler import Butler
import rustfits
from lsst.images._geom import BoundsError
# import lsst.afw.display as afwDisplay
# from lsst.images._cell_grid import CellIJ
#
# from astropy.coordinates import SkyCoord
# import astropy.units as u
# import matplotlib.pyplot as plt
# from matplotlib.collections import PatchCollection
#

SKYMAP_VERS = 'lsst_cells_v2'

# flags set in the input mask plans
DM_NO_DATA = 1
DM_SAT = 2
DM_INTRP = 4
DM_DETECTION_EDGE = 16
DM_OUT = DM_NO_DATA | DM_DETECTION_EDGE

# lsst_cells_v2 inner cell size, the fallback when the coadd's
# own cell grid cannot be introspected
CELL_SIZE = 150

# Gaia-driven bright-star subtraction and masking.  Every clean
# Gaia star down to GSUB is subtracted with an empirical
# extended template (its wings are the faint-star carpet that
# biases the background) and masked with a floored circle that
# hides the core, where a field-average template is wrong.
# Circles follow the flagged-arm extent law measured over 58
# stars in 4 fields
GAIA_EPOCH = 2016.0
OBS_EPOCH = 2025.0
MASK_R15 = 45.0     # circle radius in px at G = 15
MASK_SLOPE = 0.115  # radius scales as 10^(slope * (15 - G))
MASK_RMAX = 450.0
MINRAD = 20.0       # circle floor: subtracted cores never show
STAR_MARGIN = 210   # off-patch stars whose wings still intrude
GSAT = 15.2         # G saturation threshold of these coadds
GSUB = 18.0         # subtract stars brighter than this
RUWE_MAX = 1.4      # unsaturated census guard
GMAX = 18.0         # download depth
BG_GROW = 12        # extra star-mask margin for the background

# empirical extended star template
TMPL_HALF = 50      # measured stamp half size
TMPL_OUT_HALF = 250  # power-law halo extension half size
TMPL_NSTAR = 60
TMPL_GMIN = GSAT + 0.3  # template stars: bright but unsaturated
TMPL_GMAX = 17.5
HALO_SLOPE = -3.7   # optics halo power law, for corrupted fits

NPASS = 3           # joint amplitude passes

GAIA_TAP = 'https://gea.esac.esa.int/tap-server/tap/sync'

GAIA_ADQL = (
    'SELECT source_id, ra, dec, pmra, pmdec, parallax, '
    'phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe '
    'FROM gaiadr3.gaia_source '
    "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
    "CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {rad:.4f})) "
    'AND phot_g_mean_mag < {gmax}'
)


def fetch_gaia(wcs, bbox):
    """
    Gaia DR3 extract for this patch from the ESA TAP sync
    service (a few seconds), as a numpy structured array:
    circle centered on the patch, corner radius plus margin
    for off-patch intruders
    """
    import io
    import urllib.request
    import urllib.parse

    xmid = 0.5 * (bbox.x.start + bbox.x.stop)
    ymid = 0.5 * (bbox.y.start + bbox.y.stop)
    ctr = wcs.pixelToSky(xmid, ymid)
    corner = wcs.pixelToSky(
        float(bbox.x.start), float(bbox.y.start),
    )
    rad = ctr.separation(corner).asDegrees() + 0.02

    query = GAIA_ADQL.format(
        ra=ctr.getRa().asDegrees(),
        dec=ctr.getDec().asDegrees(),
        rad=rad,
        gmax=GMAX,
    )
    data = urllib.parse.urlencode({
        'REQUEST': 'doQuery',
        'LANG': 'ADQL',
        'FORMAT': 'csv',
        'QUERY': query,
    }).encode()
    with urllib.request.urlopen(
        GAIA_TAP, data=data, timeout=120,
    ) as resp:
        text = resp.read().decode()
    if not text.startswith('source_id'):
        raise RuntimeError(
            'unexpected TAP response: ' + text[:200],
        )
    gaia = np.genfromtxt(
        io.StringIO(text), delimiter=',', names=True,
    )
    print(f'    gaia: {gaia.size} stars')
    return gaia


def gaia_pixel_positions(gaia, wcs, bbox):
    """
    patch-frame pixel positions with proper motions propagated
    to the observation epoch
    """
    dt = OBS_EPOCH - GAIA_EPOCH
    pmra = np.nan_to_num(gaia['pmra'])
    pmdec = np.nan_to_num(gaia['pmdec'])
    cosd = np.cos(np.deg2rad(gaia['dec']))
    ra = gaia['ra'] + dt * pmra / 3.6e6 / cosd
    dec = gaia['dec'] + dt * pmdec / 3.6e6

    x, y = wcs.skyToPixelArray(ra, dec, degrees=True)
    return x - bbox.x.start, y - bbox.y.start


def circle_radius(gmag):
    return min(
        max(
            MASK_R15 * 10 ** (MASK_SLOPE * (15.0 - gmag)),
            MINRAD,
        ),
        MASK_RMAX,
    )


def select_stars(gaia, x, y, mask0):
    """
    the subtract-and-mask census: on-patch stars that are
    saturated (any RUWE) or clean point sources down to GSUB,
    plus off-patch intruders bright enough for their wings to
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
            is_bright = (
                gmag < GSUB
                and np.isfinite(ruwe) and ruwe < RUWE_MAX
            )
            if not (is_sat or is_bright):
                continue
        else:
            is_sat = 0
            if gmag >= GSAT:
                continue
            if not (-STAR_MARGIN < ix < nx + STAR_MARGIN
                    and -STAR_MARGIN < iy < ny + STAR_MARGIN):
                continue
        rows.append((
            xk, yk, gmag, ruwe, int(is_sat), int(on),
        ))
    stars = np.array(rows, dtype=[
        ('x', 'f8'), ('y', 'f8'), ('G', 'f4'), ('ruwe', 'f4'),
        ('is_sat', 'i2'), ('on_image', 'i2'),
    ])
    non = int(stars['on_image'].sum())
    print(f'    census: {non} on-patch '
          f'({int(stars["is_sat"].sum())} saturated) + '
          f'{len(stars) - non} off-patch intruders')
    return stars


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
            for dy in (-2, 0, 2):
                for dx in (-2, 0, 2):
                    lid = comps[np.clip(iy + dy, 0, ny - 1),
                                np.clip(ix + dx, 0, nx - 1)]
                    if lid > 0:
                        star_ids.add(int(lid))
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


def build_template(image, good, seg, gaia, x, y):
    """
    empirical extended star template: median stack of bright
    unsaturated Gaia stars centered on their predicted
    positions (registration is a few hundredths of a pixel),
    core-normalized, point-symmetrized, denoised to the
    azimuthal profile, and extended with a power-law halo.
    Other detections are NaNed out of each stamp so neighbors
    cannot bias the stack
    """
    from scipy import ndimage

    half = TMPL_HALF
    out_half = TMPL_OUT_HALF
    ny, nx = image.shape

    gmag = gaia['phot_g_mean_mag']
    ruwe = gaia['ruwe']
    sel = np.where(
        (gmag > TMPL_GMIN) & (gmag < TMPL_GMAX)
        & np.isfinite(ruwe) & (ruwe < RUWE_MAX)
        & (x > half + 2) & (x < nx - half - 3)
        & (y > half + 2) & (y < ny - half - 3)
    )[0]
    sel = sel[np.argsort(gmag[sel])][:TMPL_NSTAR]

    stamps = []
    for k in sel:
        cx, cy = float(x[k]), float(y[k])
        icx, icy = int(round(cx)), int(round(cy))
        m = half + 2
        stamp = image[icy - m:icy + m + 1,
                      icx - m:icx + m + 1].copy()
        stamp[~good[icy - m:icy + m + 1,
                    icx - m:icx + m + 1]] = np.nan
        segcut = seg[icy - m:icy + m + 1, icx - m:icx + m + 1]
        central = segcut[m, m]
        stamp[(segcut != 0) & (segcut != central)] = np.nan
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
    if len(stamps) < 10:
        raise RuntimeError(f'only {len(stamps)} usable '
                           'template stamps')

    tmpl = np.nanmedian(np.array(stamps), axis=0)
    tmpl[~np.isfinite(tmpl)] = 0.0
    tmpl = 0.5 * (tmpl + tmpl[::-1, ::-1])

    # denoise: blend into the azimuthal median profile beyond
    # the high-s/n core so a bright star's amplitude does not
    # amplify the stack noise
    gy, gx = np.mgrid[-half:half + 1, -half:half + 1]
    rr = np.hypot(gy, gx)
    rbin = np.round(rr).astype(int)
    prof = np.zeros(rbin.max() + 1)
    for k in range(prof.size):
        w = rbin == k
        if w.any():
            prof[k] = np.median(tmpl[w])
    if prof.size > 12:
        sm = prof.copy()
        for k in range(10, prof.size - 1):
            sm[k] = np.median(prof[max(0, k - 2):k + 3])
        prof = sm
    prof_im = prof[rbin]
    blend = np.clip((rr - 12.0) / 6.0, 0.0, 1.0)
    tmpl = (1.0 - blend) * tmpl + blend * prof_im

    # power-law halo extension; crowding can corrupt the outer
    # profile fit, in which case fall back to the optics slope
    rfit = np.arange(25, min(45, prof.size))
    pfit = prof[rfit]
    wpos = pfit > 0
    slope, ln_a = np.polyfit(
        np.log(rfit[wpos]), np.log(pfit[wpos]), 1,
    )
    if not (-5.0 < slope < -2.5):
        print(f'    template slope {slope:.2f} out of range, '
              f'using {HALO_SLOPE}')
        slope = HALO_SLOPE
        r0 = rfit[wpos]
        ln_a = np.median(
            np.log(pfit[wpos]) - slope * np.log(r0),
        )
    big = np.zeros((2 * out_half + 1, 2 * out_half + 1))
    big[out_half - half:out_half + half + 1,
        out_half - half:out_half + half + 1] = tmpl
    gyb, gxb = np.mgrid[-out_half:out_half + 1,
                        -out_half:out_half + 1]
    rrb = np.hypot(gyb, gxb)
    ext = np.exp(ln_a) * np.maximum(rrb, 1.0) ** slope
    wext = rrb >= 44.0
    big[wext] = ext[wext]
    jb = np.clip((rrb - 40.0) / 4.0, 0.0, 1.0)
    inner = rrb < 44.0
    big[inner] = (
        (1 - jb[inner]) * big[inner] + jb[inner] * ext[inner]
    )
    taper = np.clip((out_half - 2.0 - rrb) / 5.0, 0.0, 1.0)
    big = big * taper
    print(f'    template: {len(stamps)} stamps, '
          f'halo slope {slope:.2f}')
    return big


def subtract_stars(image, var, mask0, gaia, x, y, stars, comps):
    """
    subtract every census star: amplitude from the robust
    median of data/template in a 2-10 px band just outside the
    star's own mask (floored circle plus its flagged
    components), refined jointly so close pairs do not
    double-count each other's halos.  A fixed-slope flux zero
    point from the bright stars supplies amplitudes where the
    ring fails and caps ring amplitudes that land on a
    neighbor's halo.  Modifies image in place; returns the
    amplitudes
    """
    import sep
    from scipy import ndimage

    sep.set_extract_pixstack(int(1.2e7))
    sep.set_sub_object_limit(10240)

    ny, nx = image.shape
    good = (
        np.isfinite(var) & (var > 0)
        & ((mask0 & DM_OUT) == 0)
    )
    sig = float(np.sqrt(np.median(var[good])))
    imf = np.ascontiguousarray(image, dtype='f4')
    _, seg = sep.extract(
        imf, 1.5, err=sig, mask=~good, segmentation_map=True,
    )

    tmpl = build_template(image, good, seg, gaia, x, y)
    half = TMPL_OUT_HALF
    gy, gx = np.mgrid[-half:half + 1, -half:half + 1]
    rr = np.hypot(gy, gx)

    # per-star stamps and anchor rings
    slist = []
    for si, st in enumerate(stars):
        xk, yk = float(st['x']), float(st['y'])
        gmag = float(st['G'])
        ix, iy = int(round(xk)), int(round(yk))
        y0, x0 = iy - half, ix - half
        y0c, y1c = max(0, y0), min(ny, iy + half + 1)
        x0c, x1c = max(0, x0), min(nx, ix + half + 1)
        if y1c - y0c < 40 or x1c - x0c < 40:
            continue
        sl = np.s_[y0c:y1c, x0c:x1c]
        tsl = np.s_[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
        tsh = ndimage.shift(
            tmpl, (yk - iy, xk - ix), order=3, cval=0.0,
        )[tsl]
        rc = rr[tsl]
        usable = good[sl] & (tsh > 0)
        lm = rc <= circle_radius(gmag)
        if st['on_image']:
            own = set()
            for dy in (-2, 0, 2):
                for dx in (-2, 0, 2):
                    lid = comps[np.clip(iy + dy, 0, ny - 1),
                                np.clip(ix + dx, 0, nx - 1)]
                    if lid > 0:
                        own.add(int(lid))
            if own:
                lm |= np.isin(comps[sl], sorted(own))
        dmask = ndimage.distance_transform_edt(~lm)
        ring = (
            (dmask >= 2) & (dmask <= 10)
            & (rc < half - 10) & usable
        )
        if ring.sum() < 30:
            ring = None
            if not st['on_image']:
                vis = usable & (rc >= 12)
                if vis.any():
                    vring = vis & (rc < rc[vis].min() + 10.0)
                    if vring.sum() >= 30:
                        ring = vring
        slist.append({
            'sl': sl, 'T': tsh, 'ring': ring, 'A': 0.0,
            'G': gmag, 'idx': si,
        })

    # fixed-slope flux zero point from the bright-star rings
    zps = []
    for st in slist:
        if st['ring'] is not None and st['G'] < 15.5:
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

    model = np.zeros_like(image)
    for _ in range(NPASS):
        for st in slist:
            sl = st['sl']
            ap = None
            if zp is not None:
                ap = 10 ** (zp - 0.4 * st['G'])
            if st['ring'] is None:
                if ap is None or st['A'] > 0:
                    continue
                amp = ap
            else:
                resid = (
                    image[sl] - model[sl] + st['A'] * st['T']
                )
                ring = st['ring']
                amp = max(float(np.median(
                    resid[ring] / st['T'][ring],
                )), 0.0)
                if ap is not None:
                    amp = min(amp, 3.0 * ap)
            model[sl] += (amp - st['A']) * st['T']
            st['A'] = amp

    image -= model
    amps = np.array([st['A'] for st in slist])
    print(f'    subtracted {int((amps > 0).sum())} '
          f'of {len(slist)} stars')
    return slist


def get_cell_centers(deep_coadd):
    """
    tract-frame pixel centers of the coadd cells, as (xs, ys)
    1-d arrays whose outer product is the cell grid.  Tries the
    coadd's cell grid attributes; falls back to the
    lsst_cells_v2 150 px inner-cell grid over the patch bbox
    """
    bbox = deep_coadd.bbox
    try:
        from lsst.images._cell_grid import CellIJ

        grid = deep_coadd.grid

        def center(b, ax):
            a = getattr(b, ax)
            return 0.5 * (a.start + a.stop)

        # cell access as in process_cells.py pull_mbobs:
        # grid.bbox_of(CellIJ(i, j)), centers at the bbox
        # midpoint.  Probe which CellIJ axis is y rather than
        # assuming the convention
        n0, n1 = (int(v) for v in grid.grid_size)
        b00 = grid.bbox_of(CellIJ(0, 0))
        i_is_y = True
        if n0 > 1:
            b10 = grid.bbox_of(CellIJ(1, 0))
            i_is_y = b10.y.start != b00.y.start
        if i_is_y:
            nyc, nxc = n0, n1
            ys = np.array([
                center(grid.bbox_of(CellIJ(i, 0)), 'y')
                for i in range(nyc)
            ])
            xs = np.array([
                center(grid.bbox_of(CellIJ(0, j)), 'x')
                for j in range(nxc)
            ])
        else:
            nxc, nyc = n0, n1
            xs = np.array([
                center(grid.bbox_of(CellIJ(i, 0)), 'x')
                for i in range(nxc)
            ])
            ys = np.array([
                center(grid.bbox_of(CellIJ(0, j)), 'y')
                for j in range(nyc)
            ])
        csx = int(round(xs[1] - xs[0])) if nxc > 1 else CELL_SIZE
        csy = int(round(ys[1] - ys[0])) if nyc > 1 else CELL_SIZE
        return (
            xs, ys, (csx, csy), f'coadd grid {nyc} x {nxc}',
        )
    except Exception as err:
        print('cell grid introspection failed:', err)
    csx = csy = CELL_SIZE
    xs = np.arange(bbox.x.start + csx / 2 - 0.5, bbox.x.stop, csx)
    ys = np.arange(bbox.y.start + csy / 2 - 0.5, bbox.y.stop, csy)
    return xs, ys, (csx, csy), f'fallback {CELL_SIZE}px grid'


def get_wcs_header(wcs, bbox, tract, patch):
    """
    FITS WCS cards for the patch subimage: the tract wcs with
    CRPIX shifted by the patch bounding-box origin (a pure
    origin shift, so any SIP terms stay consistent)
    """
    x0 = bbox.x.start
    y0 = bbox.y.start
    md = wcs.getFitsMetadata()
    hdr = {}
    for key in md.names():
        val = md.getScalar(key)
        if key in ('SIMPLE', 'BITPIX', 'EXTEND') or \
                key.startswith('NAXIS'):
            continue
        hdr[key] = val
    hdr['CRPIX1'] = hdr['CRPIX1'] - x0
    hdr['CRPIX2'] = hdr['CRPIX2'] - y0
    # afw-style subimage origin, for going back to tract coords
    hdr['LTV1'] = -x0
    hdr['LTV2'] = -y0
    hdr['TRACT'] = tract
    hdr['PATCH'] = patch
    return hdr


def redo_background(deep_coadd, starmask=None):
    import sep

    image = deep_coadd.image.array
    var = deep_coadd.variance.array
    mask = deep_coadd.mask.array[:, :, 0]
    noise = deep_coadd.noise_realizations[0].array

    good = np.isfinite(var) & (mask & DM_OUT == 0)
    if starmask is not None:
        # bright-star vicinities carry halo/wing light the seg
        # map does not catch; keep them out of the background
        # boxes and the noise calibration
        good &= ~starmask
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
    print(f'    band: {deep_coadd.band} noise_factor: {noise_factor:g}')

    noise[:, :] *= noise_factor
    var[:, :] *= noise_factor ** 2


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument(
        '--starsub', action=argparse.BooleanOptionalAction,
        default=False,
        help='subtract the Gaia stars (empirical extended '
             'template) before any background determination',
    )
    parser.add_argument(
        '--redo-bg', action=argparse.BooleanOptionalAction,
        default=True,
        help='redo the background determination (after star '
             'subtraction when --starsub is on)',
    )
    args = parser.parse_args()

    tract = args.tract
    patch = args.patch
    print(tract)
    print(patch)

    odir = f'{tract}-images'
    if args.starsub:
        odir += '-starsub'
    if args.redo_bg:
        odir += '-redo-bg'

    # this will change
    butler = Butler('dp2_prep_future', collections=["LSSTCam/runs/DRP/DP2"])
    skymap = butler.get("skyMap", skymap=SKYMAP_VERS)
    tract_info = skymap[tract]
    wcs = tract_info.wcs

    gaia = None
    for band in ['g', 'r', 'i', 'z']:
        if not os.path.exists(odir):
            os.makedirs(odir)

        fname = f'{odir}/{tract}-{patch}-{band}.fits'

        if os.path.exists(fname):
            continue

        data_id = {
            "band": band,
            "skymap": 'lsst_cells_v2',
            "tract": tract,
            "patch": patch,
        }
        print(data_id)
        try:
            deep_coadd = butler.get('deep_coadd', dataId=data_id)
            deep_coadd.apply_background(None)

            need_gaia = args.starsub or args.redo_bg
            if gaia is None and need_gaia:
                try:
                    gaia = fetch_gaia(wcs, deep_coadd.bbox)
                except Exception as err:
                    print('    gaia download failed:', err)
                    gaia = False

            starmask = None
            star_table = None
            if gaia is not False and need_gaia:
                mask0 = deep_coadd.mask.array[:, :, 0]
                x, y = gaia_pixel_positions(
                    gaia, wcs, deep_coadd.bbox,
                )
                stars = select_stars(gaia, x, y, mask0)
                starmask, comps = build_star_mask(stars, mask0)

                if args.starsub:
                    slist = subtract_stars(
                        deep_coadd.image.array,
                        deep_coadd.variance.array,
                        mask0, gaia, x, y, stars, comps,
                    )
                    star_table = np.zeros(len(stars), dtype=[
                        ('x', 'f8'), ('y', 'f8'), ('G', 'f4'),
                        ('ruwe', 'f4'), ('is_sat', 'i2'),
                        ('on_image', 'i2'), ('A', 'f8'),
                    ])
                    for name in ('x', 'y', 'G', 'ruwe',
                                 'is_sat', 'on_image'):
                        star_table[name] = stars[name]
                    for s in slist:
                        star_table['A'][s['idx']] = s['A']
            elif need_gaia:
                print('    no gaia: star handling skipped')

            if args.redo_bg:
                smbg = None
                if starmask is not None:
                    from scipy import ndimage
                    d = ndimage.distance_transform_edt(
                        ~starmask,
                    )
                    # margin outside the mask: rim pixels are
                    # partially contaminated and must not steer
                    # the background or noise calibration
                    smbg = d < BG_GROW
                redo_background(deep_coadd, starmask=smbg)
        except Exception as err:
            print(err)
            continue

        imstd = np.nanstd(deep_coadd.image.array)
        nstd = np.nanstd(deep_coadd.noise_realizations[0].array)
        varmed = np.nanmedian(deep_coadd.variance.array)
        nmed = np.sqrt(varmed)

        psf = deep_coadd.psf
        bbox = deep_coadd.bbox
        # xmid = 0.5 * (bbox.x.start + bbox.x.stop)
        # ymid = 0.5 * (bbox.y.start + bbox.y.stop)

        # psf at every cell center.  The stamp shape comes from
        # the first successful evaluation; failed cells (edge
        # BoundsError) stay zero with ok=0 in the cell table
        xs, ys, (csx, csy), how = get_cell_centers(deep_coadd)
        print(f'psf at {ys.size} x {xs.size} cell centers ({how})')

        psf_stack = None
        cell_rows = []
        nfail = 0
        for cy in ys:
            for cx in xs:
                kim = None
                try:
                    kim = psf.compute_kernel_image(x=cx, y=cy).array
                except BoundsError:
                    nfail += 1
                if kim is not None and psf_stack is None:
                    pshape = kim.shape
                    psf_stack = np.zeros(
                        (ys.size * xs.size,) + pshape,
                        dtype='f4',
                    )
                cell_rows.append([cx, cy, kim])
        if psf_stack is None:
            print('no psf evaluation succeeded anywhere')
            png = fname + '-nobg.png'
            import matplotlib.pyplot as mplt
            with mplt.style.context('dark_background'):
                fig, ax = mplt.subplots(figsize=(10, 10))
                ax.imshow(
                    np.log10(deep_coadd.image.array.clip(min=0.001)),
                    cmap='gray',
                )
                fig.savefig(png)
            continue
        if nfail > 0:
            print(f'    {nfail} cells failed psf evaluation')

        cells = np.zeros(len(cell_rows), dtype=[
            ('cellx', 'f4'), ('celly', 'f4'), ('ok', 'i2'),
        ])
        for k, (cx, cy, kim) in enumerate(cell_rows):
            # patch-frame pixel coordinates
            cells['cellx'][k] = cx - bbox.x.start
            cells['celly'][k] = cy - bbox.y.start
            if kim is not None and kim.shape == pshape:
                psf_stack[k] = kim
                cells['ok'][k] = 1

        hdr = get_wcs_header(wcs, bbox, tract, patch)

        print('image std:', imstd)
        print('noise std:', nstd)
        print('med from plane:', nmed)
        # import IPython
        # IPython.embed()

        print('writing:', fname)
        with rustfits.FITS(fname, 'w+') as fits:
            fits.write_image(
                deep_coadd.image.array,
                extname='image',
                compress='gzip_2',
                header=hdr,
            )
            fits.write_image(
                deep_coadd.variance.array,
                extname='var',
                compress='gzip_2',
            )
            fits.write_image(
                deep_coadd.mask.array,
                extname='mask',
                compress=rustfits.Gzip2(tile_shape=[1, 3300, 1]),
            )
            fits.write_image(
                deep_coadd.noise_realizations[0].array,
                extname='noise',
                compress='gzip_2',
            )
            # (ncell, ny, nx) psf stamps, row k matching row k of
            # the psf_cells table; cell centers are patch-frame
            # pixel coordinates
            fits.write_image(
                psf_stack,
                extname='psfs',
                compress='gzip_2',
            )
            fits.write_table(
                cells,
                extname='psf_cells',
                header={
                    'CELLSX': csx, 'CELLSY': csy,
                    'NCELLX': int(xs.size),
                    'NCELLY': int(ys.size),
                },
            )
            if starmask is not None and args.starsub:
                # circles + flagged components of the census;
                # pixels here are unreliable after subtraction
                fits.write_image(
                    starmask.astype('u1'),
                    extname='starmask',
                    compress='gzip_2',
                )
            if star_table is not None:
                fits.write_table(
                    star_table,
                    extname='gaia_stars',
                    header={'GSUB': GSUB, 'MINRAD': MINRAD},
                )


main()
