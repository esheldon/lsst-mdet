"""
cli/getimages
"""
from lsst.daf.butler import Butler
from lsst.images._geom import BoundsError
import numpy as np
import os
import rustfits
from ..apodize import taper_from_distance
from ..background import redo_background
from ..cells import get_cell_centers
from ..defaults import SKYMAP_VERS
from ..gaia import GMAX, fetch_gaia, gaia_pixel_positions
from ..starsub import (
    APOD_STARS,
    BG_GROW,
    GSUB,
    MINRAD,
    build_star_mask,
    select_stars,
    subtract_stars,
)
from ..wcs import get_wcs_header
from ..patchfiles import get_patch_filename


def main():
    args = get_args()

    tract = args.tract
    patch = args.patch

    if not os.path.exists(args.patch_dir):
        os.makedirs(args.patch_dir, exist_ok=True)

    # this will change
    butler = Butler('dp2_prep_future', collections=["LSSTCam/runs/DRP/DP2"])
    skymap = butler.get("skyMap", skymap=SKYMAP_VERS)
    tract_info = skymap[tract]
    wcs = tract_info.wcs

    gaia = None
    for band in ['g', 'r', 'i', 'z']:

        fname = get_patch_filename(
            tract=tract, patch=patch, band=band, patch_dir=args.patch_dir,
        )

        if os.path.exists(fname):
            print(f'{fname} already exists')
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
            # deep_coadd.apply_background(None)
            deep_coadd.apply_background('object')

            need_gaia = args.starsub or args.redo_bg
            if gaia is None and need_gaia:
                try:
                    gaia = fetch_gaia(
                        wcs, deep_coadd.bbox,
                        gmax=max(args.gsub, GMAX),
                    )
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
                stars = select_stars(
                    gaia, x, y, mask0, gsub=args.gsub,
                )
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

            dstar = None
            if starmask is not None:
                from scipy import ndimage
                dstar = ndimage.distance_transform_edt(
                    ~starmask,
                )
            if args.redo_bg:
                # margin outside the mask: rim pixels are
                # partially contaminated and must not steer
                # the background or noise calibration
                smbg = None
                if dstar is not None:
                    smbg = dstar < BG_GROW
                redo_background(deep_coadd, starmask=smbg)

            # apodize the star-mask regions AFTER the
            # background determination, in both the image and
            # the noise realization so they stay statistically
            # matched
            taper_applied = False
            if args.starsub and args.apod_stars \
                    and dstar is not None:
                taper = taper_from_distance(
                    dstar, APOD_STARS,
                )
                deep_coadd.image.array[:, :] *= taper
                nz = deep_coadd.noise_realizations[0].array
                nz[:, :] *= taper
                taper_applied = True
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
            # masked-fraction plane, needed for butler-free
            # processing (process_cells --patch-dir)
            fits.write_image(
                deep_coadd.mask_fractions['rejected'].array,
                extname='mfrac',
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
                # 2 = star mask (zeroed when APOD > 0),
                # 1 = taper zone (attenuated -- masked for any
                # measurement, smooth enough for FFTs),
                # 0 = clear
                maskout = np.zeros(starmask.shape, dtype='u1')
                if taper_applied:
                    maskout[(dstar > 0)
                            & (dstar < APOD_STARS)] = 1
                maskout[starmask] = 2
                fits.write_image(
                    maskout,
                    extname='starmask',
                    compress='gzip_2',
                    header={
                        'APOD': (
                            APOD_STARS if taper_applied
                            else 0
                        ),
                    },
                )
            if star_table is not None:
                fits.write_table(
                    star_table,
                    extname='gaia_stars',
                    header={'GSUB': args.gsub,
                            'MINRAD': MINRAD},
                )


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--patch-dir', required=True)
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
