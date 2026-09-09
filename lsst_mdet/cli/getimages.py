"""
cli/getimages: extract patch images from the butler to FITS,
with optional Gaia star subtraction and background
redetermination
"""
import os
import numpy as np
from lsst.daf.butler import Butler

from ..background import redo_background
from ..cells import get_cell_centers, get_tract_bounds, make_psf_cube
from ..defaults import BUTLER_COLLECTIONS, BUTLER_REPO, SKYMAP_VERS
from ..gaia import GMAX, fetch_gaia_or_none, read_gaia_file
from ..io import write_patch_files
from ..patchfiles import get_patch_filename
from ..starsub import (
    APOD_STARS,
    BG_GROW,
    GSUB,
    MINRAD,
    apply_star_taper,
    handle_stars,
    make_starmask_plane,
)
from ..wcs import get_wcs_header


def save_nobg_png(deep_coadd, fname):
    """diagnostic image for a band whose psf evaluation failed
    everywhere"""
    import matplotlib.pyplot as mplt

    png = fname + '-nobg.png'
    with mplt.style.context('dark_background'):
        fig, ax = mplt.subplots(figsize=(10, 10))
        ax.imshow(
            np.log10(deep_coadd.image.array.clip(min=0.001)),
            cmap='gray',
        )
        fig.savefig(png)


def prepare_band_stars(deep_coadd, wcs, gaia, args):
    """
    star handling and background redetermination for one band,
    modifying the coadd in place.  The star-region taper is NOT
    applied here: its distance field must be shared across the
    bands so the attenuation zones match (see main)

    Returns (dstar, star_table, skyvar): the distance transform
    off this band's star mask (None without star handling), the
    gaia census table with fitted amplitudes (None unless
    subtracting), and the sky-variance map for the pixel
    weights (None without the background redo)
    """
    star_table = None
    dstar = None
    if gaia is not None and args.starsub and args.starsub_method == 'joint':
        from lsst_starsub.starsub import handle_stars_joint, load_wing
        wing = load_wing(args.wing_pattern.format(band=deep_coadd.band))
        _, star_table, dstar = handle_stars_joint(
            deep_coadd, wcs, gaia, wing, gsub=args.gsub,
        )
    elif gaia is not None:
        _, star_table, dstar = handle_stars(
            deep_coadd, wcs, gaia,
            gsub=args.gsub, subtract=args.starsub,
        )
    elif args.starsub or args.redo_bg:
        print('    no gaia: star handling skipped')

    skyvar = None
    if args.redo_bg:
        # margin outside the mask: rim pixels are partially
        # contaminated and must not steer the background or
        # noise calibration
        smbg = None
        if dstar is not None:
            smbg = dstar < BG_GROW
        skyvar = redo_background(deep_coadd, starmask=smbg)
    else:
        print('    WARNING: no background redo: diagnostic '
              'mode only; no skyvar extension will be '
              'written, and processing will fall back to the '
              'raw variance plane for pixel weights')

    return dstar, star_table, skyvar


def main():
    args = get_args()

    tract = args.tract
    patch = args.patch

    if not os.path.exists(args.patch_dir):
        os.makedirs(args.patch_dir, exist_ok=True)

    bands = ['g', 'r', 'i', 'z']
    fnames = {
        band: get_patch_filename(
            tract=tract, patch=patch, band=band,
            patch_dir=args.patch_dir,
        )
        for band in bands
    }
    if all(os.path.exists(f) for f in fnames.values()):
        print('all band files exist')
        return

    butler = Butler(args.repo, collections=args.collections)
    skymap = butler.get("skyMap", skymap=SKYMAP_VERS)
    tract_info = skymap[tract]
    wcs = tract_info.wcs
    tract_bounds = get_tract_bounds(tract_info)

    # load every available band first: the shared star taper
    # couples the bands, so the work cannot proceed band by band
    coadds = {}
    for band in bands:
        data_id = {
            "band": band,
            "skymap": SKYMAP_VERS,
            "tract": tract,
            "patch": patch,
        }
        print(data_id)
        try:
            deep_coadd = butler.get('deep_coadd', dataId=data_id)
            deep_coadd.apply_background('object')
        except (LookupError, OSError) as err:
            # missing dataset (DatasetNotFoundError is a
            # LookupError) or unreadable artifact: skip the
            # band.  Anything else is a bug and should crash
            print(f'{type(err).__name__}: {err}')
            continue
        coadds[band] = deep_coadd

    if len(coadds) == 0:
        print('no bands available')
        return

    if all(os.path.exists(fnames[band]) for band in coadds):
        print('all available band files exist')
        return

    nexist = sum(os.path.exists(fnames[band]) for band in coadds)
    if nexist > 0:
        print(f'{nexist} band files exist; rewriting all: the '
              'shared star taper couples the bands')

    # gaia once, from the first available band's bbox (the
    # patch bbox is the same in every band).  With --starsub a
    # gaia failure must abort the run (fetch_gaia_or_none
    # raises), never silently skip the subtraction
    gaia = None
    first_coadd = next(iter(coadds.values()))
    if args.starsub or args.redo_bg:
        gmax = max(args.gsub, GMAX)
        if args.gaia_file is not None:
            # a bad file is user error: crash, never degrade
            gaia = read_gaia_file(
                args.gaia_file, wcs, first_coadd.bbox,
                gmax=gmax,
            )
        else:
            gaia = fetch_gaia_or_none(
                wcs, first_coadd.bbox, gmax=gmax,
                require=args.starsub,
            )

    # per-band star handling and background; the taper is
    # deferred until every band's mask is known.  A sparse-field
    # template failure degrades to mask-only inside
    # subtract_stars, so any error here is a bug and should
    # crash
    star_tables = {}
    skyvars = {}
    dstar_min = None
    for band, deep_coadd in coadds.items():
        print(f'star handling and background: {band}')
        dstar, star_table, skyvar = prepare_band_stars(
            deep_coadd, wcs, gaia, args,
        )
        star_tables[band] = star_table
        skyvars[band] = skyvar
        if dstar is not None:
            # the distance to the union of the per-band star
            # masks is the minimum of the per-band distances
            if dstar_min is None:
                dstar_min = dstar
            else:
                np.minimum(dstar_min, dstar, out=dstar_min)

    # one taper and one starmask plane for every band, from the
    # union distance field; apodize AFTER the background
    # determination
    apod = 0.0
    starmask_plane = None
    if args.starsub and dstar_min is not None:
        if args.apod_stars:
            for deep_coadd in coadds.values():
                apply_star_taper(
                    deep_coadd, dstar_min, width=APOD_STARS,
                )
            apod = APOD_STARS
        # dstar_min == 0 exactly on the union of the star masks
        starmask_plane = make_starmask_plane(
            dstar_min == 0, dstar_min, apod,
        )

    for band, deep_coadd in coadds.items():
        fname = fnames[band]

        # psf at every cell center
        xs, ys, (csx, csy), how = get_cell_centers(deep_coadd)
        print(f'psf at {ys.size} x {xs.size} cell centers ({how})')
        psf_stack, cells = make_psf_cube(deep_coadd, xs, ys)
        if psf_stack is None:
            print('no psf evaluation succeeded anywhere')
            save_nobg_png(deep_coadd, fname)
            continue

        hdr = get_wcs_header(wcs, deep_coadd.bbox, tract, patch)
        # tract inner sky bounds, for the primary cut and the
        # footprint trim in file-mode processing
        hdr['TRAMIN'] = tract_bounds[0]
        hdr['TRAMAX'] = tract_bounds[1]
        hdr['TDECMIN'] = tract_bounds[2]
        hdr['TDECMAX'] = tract_bounds[3]

        print('image std:', np.nanstd(deep_coadd.image.array))
        print('noise std:',
              np.nanstd(deep_coadd.noise_realizations[0].array))
        print('med from plane:',
              np.sqrt(np.nanmedian(deep_coadd.variance.array)))

        write_patch_files(
            fname=fname,
            deep_coadd=deep_coadd,
            hdr=hdr,
            psf_stack=psf_stack,
            cells=cells,
            ncellx=xs.size,
            ncelly=ys.size,
            cell_size=(csx, csy),
            starmask_plane=starmask_plane,
            apod=apod,
            star_table=star_tables[band],
            gsub=args.gsub,
            minrad=MINRAD,
            skyvar=skyvars[band],
        )


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--patch-dir', required=True)
    parser.add_argument(
        '--repo', default=BUTLER_REPO,
        help='butler repo path or alias',
    )
    parser.add_argument(
        '--collections', nargs='+', default=BUTLER_COLLECTIONS,
        help='butler collections to search',
    )
    parser.add_argument(
        '--starsub', action=argparse.BooleanOptionalAction,
        default=False,
        help='subtract the Gaia stars (empirical extended '
             'template) before any background determination',
    )
    parser.add_argument(
        '--starsub-method', default='template',
        choices=['template', 'joint'],
        help='the star subtraction: this package\'s template route '
             '(the reference) or the joint star-and-sky fit of '
             'lsst_starsub (needs --wing-pattern)',
    )
    parser.add_argument(
        '--wing-pattern',
        help='the per-band wing file for --starsub-method joint, a '
             'pattern with a {band} placeholder',
    )
    parser.add_argument(
        '--redo-bg', action=argparse.BooleanOptionalAction,
        default=True,
        help='redo the background determination (after star '
             'subtraction when --starsub is on)',
    )
    parser.add_argument(
        '--gsub', type=float, default=GSUB,
        help='subtract and mask Gaia stars brighter than '
             'this; the download depth follows it',
    )
    parser.add_argument(
        '--gaia-file',
        help='read the gaia stars from this parquet file '
             '(columns gaia_g_mag, ra, dec) instead of the '
             'TAP query',
    )
    parser.add_argument(
        '--apod-stars', action=argparse.BooleanOptionalAction,
        default=True,
        help='zero the star-mask regions in the image and '
             'noise planes with a smooth taper: hard-edged '
             'holes and raw saturated cores ring in k-space',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
