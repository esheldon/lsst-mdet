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

# Gaia-driven bright-star mask for the background estimation:
# halo/wing light near bright stars biases the sep background
# boxes upward (measured on 3177-90: +0.5 counts near stars in
# z plus a -1.4 count global offset), digging a negative moat
# around every star when subtracted.  Circles follow the
# flagged-arm extent law measured over 58 stars in 4 fields
GAIA_EPOCH = 2016.0
OBS_EPOCH = 2025.0
MASK_R15 = 45.0     # circle radius in px at G = 15
MASK_SLOPE = 0.115  # radius scales as 10^(slope * (15 - G))
MASK_RMAX = 450.0
STAR_MARGIN = 210   # off-patch stars whose wings still intrude
GSAT = 15.2         # G saturation threshold of these coadds
GMAX = 18.0         # download depth
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


def build_star_mask(gaia, wcs, bbox, mask0):
    """
    magnitude-scaled circles at the Gaia bright stars (proper
    motions propagated to the observation epoch) plus the
    SAT/INTRP components of the saturated ones.  mask0 is the
    first channel of the patch mask plane; positions come from
    the tract wcs shifted to the patch frame
    """
    from scipy import ndimage

    dt = OBS_EPOCH - GAIA_EPOCH
    pmra = np.nan_to_num(gaia['pmra'])
    pmdec = np.nan_to_num(gaia['pmdec'])
    cosd = np.cos(np.deg2rad(gaia['dec']))
    ra = gaia['ra'] + dt * pmra / 3.6e6 / cosd
    dec = gaia['dec'] + dt * pmdec / 3.6e6

    x, y = wcs.skyToPixelArray(ra, dec, degrees=True)
    x = x - bbox.x.start
    y = y - bbox.y.start

    ny, nx = mask0.shape
    comps, _ = ndimage.label((mask0 & (DM_SAT | DM_INTRP)) != 0)

    starmask = np.zeros((ny, nx), dtype=bool)
    star_ids = set()
    nstar = 0
    for k in np.argsort(gaia['phot_g_mean_mag']):
        gmag = float(gaia['phot_g_mean_mag'][k])
        xk, yk = float(x[k]), float(y[k])
        ix, iy = int(round(xk)), int(round(yk))
        on = 0 <= ix < nx and 0 <= iy < ny
        if on:
            # for a background mask any bright star matters,
            # saturated in this band or not: its halo biases
            # the boxes either way
            if gmag >= GSAT + 0.5:
                continue
        else:
            if gmag >= GSAT:
                continue
            if not (-STAR_MARGIN < ix < nx + STAR_MARGIN
                    and -STAR_MARGIN < iy < ny + STAR_MARGIN):
                continue
        nstar += 1
        if on:
            for dy in (-2, 0, 2):
                for dx in (-2, 0, 2):
                    lid = comps[np.clip(iy + dy, 0, ny - 1),
                                np.clip(ix + dx, 0, nx - 1)]
                    if lid > 0:
                        star_ids.add(int(lid))
        rad = min(
            MASK_R15 * 10 ** (MASK_SLOPE * (15.0 - gmag)),
            MASK_RMAX,
        )
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
    print(f'    star mask: {nstar} stars, '
          f'fraction {starmask.mean():.3f}')
    return starmask


def get_cell_centers(deep_coadd):
    """
    tract-frame pixel centers of the coadd cells, as (xs, ys)
    1-d arrays whose outer product is the cell grid.  Tries the
    coadd's cell grid attributes; falls back to the
    lsst_cells_v2 150 px inner-cell grid over the patch bbox
    """
    bbox = deep_coadd.bbox
    grid = getattr(deep_coadd, 'grid', None)
    if grid is None:
        grid = getattr(deep_coadd, 'cell_grid', None)
    if grid is not None:
        try:
            cs = getattr(grid, 'cell_size', None)
            try:
                csx, csy = int(cs.x), int(cs.y)
            except AttributeError:
                csx = csy = int(cs)
            gb = getattr(grid, 'bbox', bbox)
            xs = np.arange(gb.x.start + csx / 2 - 0.5, gb.x.stop, csx)
            ys = np.arange(gb.y.start + csy / 2 - 0.5, gb.y.stop, csy)
            return xs, ys, (csx, csy), 'coadd grid'
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
    args = parser.parse_args()

    tract = args.tract
    patch = args.patch
    print(tract)
    print(patch)

    # this will change
    butler = Butler('dp2_prep_future', collections=["LSSTCam/runs/DRP/DP2"])
    skymap = butler.get("skyMap", skymap=SKYMAP_VERS)
    tract_info = skymap[tract]
    wcs = tract_info.wcs

    gaia = None
    for band in ['g', 'r', 'i', 'z']:
        # odir = f'{tract}-images'
        odir = f'{tract}-images-redo-bg'
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
            if gaia is None:
                try:
                    gaia = fetch_gaia(wcs, deep_coadd.bbox)
                except Exception as err:
                    print('    gaia download failed:', err)
                    gaia = False
            starmask = None
            if gaia is not False:
                starmask = build_star_mask(
                    gaia,
                    wcs,
                    deep_coadd.bbox,
                    deep_coadd.mask.array[:, :, 0],
                )
            else:
                print('    background without star mask')
            redo_background(deep_coadd, starmask=starmask)
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


main()
