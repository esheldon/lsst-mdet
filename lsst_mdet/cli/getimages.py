"""
cli/getimages: extract patch images from the butler to FITS,
with optional Gaia star subtraction and background
redetermination
"""
import os
import numpy as np
from lsst.daf.butler import Butler

from ..background import redo_background
from ..cells import get_cell_centers, make_psf_cube
from ..defaults import BUTLER_COLLECTIONS, BUTLER_REPO, SKYMAP_VERS
from ..gaia import GMAX, fetch_gaia_or_none
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


def prepare_band(deep_coadd, wcs, gaia, args):
    """
    star handling, background redetermination, and star-region
    apodization for one band, modifying the coadd in place.

    Returns (starmask_plane, apod, star_table): the three
    valued starmask output plane (None without star handling),
    the taper width actually applied, and the gaia census
    table with fitted amplitudes (None unless subtracting)
    """
    starmask = None
    star_table = None
    dstar = None
    if gaia is not None:
        starmask, star_table, dstar = handle_stars(
            deep_coadd, wcs, gaia,
            gsub=args.gsub, subtract=args.starsub,
        )
    elif args.starsub or args.redo_bg:
        print('    no gaia: star handling skipped')

    if args.redo_bg:
        # margin outside the mask: rim pixels are partially
        # contaminated and must not steer the background or
        # noise calibration
        smbg = None
        if dstar is not None:
            smbg = dstar < BG_GROW
        redo_background(deep_coadd, starmask=smbg)

    # apodize AFTER the background determination
    apod = 0.0
    if args.starsub and args.apod_stars and dstar is not None:
        apply_star_taper(deep_coadd, dstar, width=APOD_STARS)
        apod = APOD_STARS

    starmask_plane = None
    if starmask is not None and args.starsub:
        starmask_plane = make_starmask_plane(
            starmask, dstar, apod,
        )
    return starmask_plane, apod, star_table


def main():
    args = get_args()

    tract = args.tract
    patch = args.patch

    if not os.path.exists(args.patch_dir):
        os.makedirs(args.patch_dir, exist_ok=True)

    butler = Butler(args.repo, collections=args.collections)
    skymap = butler.get("skyMap", skymap=SKYMAP_VERS)
    wcs = skymap[tract].wcs

    gaia = None
    gaia_failed = False
    for band in ['g', 'r', 'i', 'z']:
        fname = get_patch_filename(
            tract=tract, patch=patch, band=band,
            patch_dir=args.patch_dir,
        )
        if os.path.exists(fname):
            print(f'{fname} already exists')
            continue

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

        # outside the band try/except: with --starsub a gaia
        # failure must abort the run (fetch_gaia_or_none
        # raises), never silently skip the subtraction.
        # Without it, degrade once and do not retry per band
        if gaia is None and not gaia_failed \
                and (args.starsub or args.redo_bg):
            gaia = fetch_gaia_or_none(
                wcs, deep_coadd.bbox,
                gmax=max(args.gsub, GMAX),
                require=args.starsub,
            )
            if gaia is None:
                gaia_failed = True

        try:
            starmask_plane, apod, star_table = prepare_band(
                deep_coadd, wcs, gaia, args,
            )
        except RuntimeError as err:
            # the expected failure is the star template
            # (too few usable stamps on a pathological band):
            # skip the band.  Anything else should crash
            print(f'{type(err).__name__}: {err}')
            continue

        # psf at every cell center
        xs, ys, (csx, csy), how = get_cell_centers(deep_coadd)
        print(f'psf at {ys.size} x {xs.size} cell centers ({how})')
        psf_stack, cells = make_psf_cube(deep_coadd, xs, ys)
        if psf_stack is None:
            print('no psf evaluation succeeded anywhere')
            save_nobg_png(deep_coadd, fname)
            continue

        hdr = get_wcs_header(wcs, deep_coadd.bbox, tract, patch)

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
            star_table=star_table,
            gsub=args.gsub,
            minrad=MINRAD,
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
        '--apod-stars', action=argparse.BooleanOptionalAction,
        default=True,
        help='zero the star-mask regions in the image and '
             'noise planes with a smooth taper: hard-edged '
             'holes and raw saturated cores ring in k-space',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
