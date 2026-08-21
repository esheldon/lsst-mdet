"""
cli/process_cells
"""
import numpy as np
from ..apodize import apodize_mbobs
from ..cells import (
    load_coadds_butler, pull_mbobs, get_cell_healsparse_polygon,
    get_tract_primary,
)
from ..defaults import BUTLER_COLLECTIONS, BUTLER_REPO
from ..hmaps import (
    make_empty_footprint,
    mask_stars_in_footprint,
    trim_footprint_to_tract_bounds,
)
from ..patchfiles import load_coadds_files
from ..io import write_output
from ..pipeline import do_metacal_and_process, process_one_mbobs
from ..psf import fit_and_set_psfrec
from ..wcs import calculate_positions


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--outfile', required=True)
    parser.add_argument(
        '--patch-dir',
        help='process from getimages FITS output instead of '
             'the butler (no LSST stack needed); --redo-bg '
             'and --starsub are refused in this mode, they '
             'are getimages-time operations',
    )
    parser.add_argument(
        '--repo', default=BUTLER_REPO,
        help='butler repo path or alias (butler mode only)',
    )
    parser.add_argument(
        '--collections', nargs='+', default=BUTLER_COLLECTIONS,
        help='butler collections to search (butler mode only)',
    )
    parser.add_argument(
        '--gaia-file',
        help='read the gaia stars from this parquet file '
             '(columns gaia_g_mag, ra, dec) instead of the '
             'TAP query (butler mode with --starsub only)',
    )
    parser.add_argument('--deblend', action='store_true')
    parser.add_argument('--s2-detect', action='store_true')
    parser.add_argument('--redo-bg', action='store_true')
    parser.add_argument(
        '--starsub', action='store_true',
        help='subtract and mask the Gaia stars at the patch '
             'level (getimages_patch machinery) before any '
             'background redo',
    )
    parser.add_argument('--mdet', action='store_true')
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def main(
    tract,
    patch,
    model,
    seed,
    with_mdet,
    redo_bg,
    starsub,
    patch_dir,
    outfile,
    deblend,
    s2_detect,
    progress,
    show,
    repo=BUTLER_REPO,
    collections=BUTLER_COLLECTIONS,
    gaia_file=None,
):
    from tqdm import trange

    rng = np.random.RandomState(seed)

    footprint = make_empty_footprint()

    dlist = []

    bands = ['r', 'i', 'z']

    if patch_dir is not None:
        if redo_bg or starsub:
            raise ValueError(
                '--redo-bg and --starsub are getimages-time '
                'operations; the patch files already carry '
                'their effects'
            )
        deep_coadds, wcs, starmask, star_table, apod, tract_bounds = (
            load_coadds_files(
                patch_dir=patch_dir, tract=tract, patch=patch,
                bands=bands,
            )
        )
    else:
        from lsst.daf.butler import Butler

        butler = Butler(repo, collections=collections)
        deep_coadds, wcs, starmask, star_table, apod, tract_bounds = (
            load_coadds_butler(
                butler=butler, tract=tract, patch=patch,
                bands=bands, redo_bg=redo_bg, starsub=starsub,
                gaia_file=gaia_file,
            )
        )

    if progress:
        mrng_i = trange(1, 21, desc='cell_i', ncols=80, ascii=True)
        # mrng_i = trange(17, 18, desc='cell_i', ncols=80, ascii=True)
    else:
        mrng_i = range(1, 21)

    cell_meta_list = []
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

            mbobs, cell_meta = pull_mbobs(
                deep_coadds=deep_coadds,
                cell_i=cell_i,
                cell_j=cell_j,
                wcs=wcs,
                starmask=starmask,
            )
            cell_meta['tract'] = tract
            cell_meta['patch'] = patch
            cell_meta_list.append(cell_meta)

            if mbobs is None:
                continue

            apodize_mbobs(mbobs)

            if with_mdet:
                cat = do_metacal_and_process(
                    mbobs=mbobs,
                    model=model,
                    deblend=deblend,
                    s2_detect=s2_detect,
                    rng=rng,
                    show=show,
                )
            else:
                cat = process_one_mbobs(
                    mbobs=mbobs,
                    model=model,
                    deblend=deblend,
                    s2_detect=s2_detect,
                    rng=rng,
                    show=show,
                )
                cat['mcal_step'] = 'na'

            fit_and_set_psfrec(st=cat, mbobs=mbobs, rng=rng)

            calculate_positions(
                bbox=mbobs[0][0].meta['bbox'],
                wcs=wcs,
                cat=cat,
            )
            cat['cell_i'] = cell_i
            cat['cell_j'] = cell_j

            footprint |= get_cell_healsparse_polygon(
                bbox=deep_coadds[0].bbox,
                cell_i=cell_i,
                cell_j=cell_j,
                wcs=wcs,
            )

            nkeep += 1
            dlist.append(cat)

        # break

    print(f'kept {nkeep}/{ncell} {nkeep / ncell:g}')

    cell_meta = np.concatenate(cell_meta_list)
    st = np.concatenate(dlist)

    # tracts overlap: primary objects must also be within the
    # tract inner boundary
    st['is_primary'] &= get_tract_primary(
        tract_bounds, st['ra'], st['dec'],
    )

    write_output(
        fname=outfile,
        st=st,
        cell_meta=cell_meta,
        tract=tract,
        patch=patch,
        model=model,
        seed=seed,
        with_mdet=with_mdet,
        redo_bg=redo_bg,
        starsub=starsub,
        deblend=deblend,
        s2_detect=s2_detect,
    )

    if star_table is not None and star_table.size > 0:
        mask_stars_in_footprint(
            footprint=footprint,
            wcs=wcs,
            bbox=deep_coadds[0].bbox,
            star_table=star_table,
            apod=apod,
        )

    # tracts overlap: trim to the inner boundary, the same
    # test as the is_primary cut
    trim_footprint_to_tract_bounds(footprint, tract_bounds)

    footprint_fname = outfile.replace('.fits', '-footprint.hsp')

    print('writing:', footprint_fname)
    footprint.write(footprint_fname, clobber=True)


def main_cli():
    _args = get_args()
    main(
        with_mdet=_args.mdet,
        seed=_args.seed,
        tract=_args.tract,
        patch=_args.patch,
        model=_args.model,
        deblend=_args.deblend,
        s2_detect=_args.s2_detect,
        redo_bg=_args.redo_bg,
        starsub=_args.starsub,
        patch_dir=_args.patch_dir,
        outfile=_args.outfile,
        progress=_args.progress,
        show=_args.show,
        repo=_args.repo,
        collections=_args.collections,
        gaia_file=_args.gaia_file,
    )


if __name__ == '__main__':
    main_cli()
