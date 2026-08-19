"""
cli/process_cells
"""
import numpy as np
import os
from ..apodize import apodize_mbobs
from ..cells import load_coadds_butler, pull_mbobs
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
):
    from tqdm import trange

    rng = np.random.RandomState(seed)

    dlist = []

    bands = ['r', 'i', 'z']

    if patch_dir is not None:
        if redo_bg or starsub:
            raise ValueError(
                '--redo-bg and --starsub are getimages-time '
                'operations; the patch files already carry '
                'their effects'
            )
        deep_coadds, wcs, starmask = load_coadds_files(
            patch_dir=patch_dir, tract=tract, patch=patch,
            bands=bands,
        )
    else:
        from lsst.daf.butler import Butler

        butler = Butler(
            'dp2_prep_future',
            collections=["LSSTCam/runs/DRP/DP2"],
        )
        deep_coadds, wcs, starmask = load_coadds_butler(
            butler=butler, tract=tract, patch=patch,
            bands=bands, redo_bg=redo_bg, starsub=starsub,
        )

    if progress:
        mrng_i = trange(1, 21, desc='cell_i', ncols=80, ascii=True)
        # mrng_i = trange(17, 18, desc='cell_i', ncols=80, ascii=True)
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
                starmask=starmask,
            )
            cell_info['tract'] = tract
            cell_info['patch'] = patch
            cell_info_list.append(cell_info)

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
            nkeep += 1
            dlist.append(cat)

        # break

    print(f'kept {nkeep}/{ncell} {nkeep / ncell:g}')

    cell_info = np.concatenate(cell_info_list)
    st = np.concatenate(dlist)

    write_output(
        fname=outfile,
        st=st,
        cell_info=cell_info,
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
    )


if __name__ == '__main__':
    main_cli()
