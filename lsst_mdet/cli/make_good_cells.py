"""
build the good-cells file: one row per processable cell, over
every tract and patch in the skymap.

A cell is good when it and its 8 neighbors all have at least
MIN_EPOCHS epochs contributing unmasked data in every band
(lsst-mdet-process-cells requires valid neighbor cells).  The
output has one row per good cell with tract, patch, cell_i,
cell_j, ra_center, dec_center.

Index conventions: cell_i is the y/row index and cell_j the
x/column index, following lsst.images CellIJ -- the same
convention as the coadd provenance table and
lsst-mdet-process-cells --cells, so the rows are directly
usable there.  The skymap getCellInfo index is the legacy
(x, y) Index2D; CellIJ.to_legacy() performs that conversion
safely.
"""
from ..defaults import BUTLER_COLLECTIONS, BUTLER_REPO, SKYMAP_VERS
from .make_slurm import NCELL_SIDE

# epochs with unmasked data required in every band, for the
# cell and each of its neighbors
MIN_EPOCHS = 3

# the full cell grid includes the unprocessed border ring
NCELL_GRID = NCELL_SIDE + 2

# patches per tract in this skymap
NPATCH = 100

DEFAULT_BANDS = ['r', 'i', 'z']


def get_cell_info_struct(tract, patch, cell_i, cell_j,
                         ra_center, dec_center):
    import numpy as np

    dtype = [
        ('tract', 'i4'),
        ('patch', 'i4'),
        ('cell_i', 'i4'),
        ('cell_j', 'i4'),
        ('ra_center', 'f8'),
        ('dec_center', 'f8'),
    ]
    cell_info = np.zeros(1, dtype=dtype)
    cell_info['tract'] = tract
    cell_info['patch'] = patch
    cell_info['cell_i'] = cell_i
    cell_info['cell_j'] = cell_j
    cell_info['ra_center'] = ra_center
    cell_info['dec_center'] = dec_center
    return cell_info


def get_good_cells_for_tract(butler, skymap, tract, bands,
                             verbose=False):
    """
    the good cells of one tract, or None when the tract has no
    usable patches
    """
    import numpy as np
    import lsst.geom
    from lsst.daf.butler import EmptyQueryResultError
    from lsst.images._cell_grid import CellIJ
    from tqdm import trange

    bstr = ', '.join(["'%s'" % b for b in bands])

    try:
        refs = butler.query_datasets(
            'deep_coadd',
            where=(
                f"band in ({bstr}) and skymap='{SKYMAP_VERS}' "
                f'and tract={tract}'
            ),
        )
    except EmptyQueryResultError:
        return None

    tract_info = skymap[tract]
    wcs = tract_info.wcs

    if verbose:
        itr = trange(
            NPATCH, desc=f'tract {tract}', leave=False,
            ascii=True, ncols=70,
        )
    else:
        itr = range(NPATCH)

    cell_info_list = []

    for patch in itr:
        patch_info = tract_info.getPatchInfo(patch)

        contrib = {}
        for band in bands:
            found_ref = None
            for ref in refs:
                if (ref.dataId['patch'] == patch
                        and ref.dataId['band'] == band):
                    found_ref = ref

            if found_ref:
                contrib[band] = butler.get(
                    found_ref.makeComponentRef('provenance')
                ).contributions
            else:
                break

        # sometimes a patch doesn't have a ref in the butler,
        # so skip it if we did not find every band
        if len(contrib) != len(bands):
            continue

        # first record all cells that have valid data.  The
        # provenance table's cell_i is the y/row index, the
        # convention of this file's output (see the module
        # docstring)
        cell_has_data = np.zeros(
            (NCELL_GRID, NCELL_GRID), dtype=np.bool_,
        )
        for cell_i in range(0, NCELL_GRID):
            for cell_j in range(0, NCELL_GRID):
                keep = True

                for band in bands:
                    msk = (
                        (contrib[band]['cell_i'] == cell_i)
                        & (contrib[band]['cell_j'] == cell_j)
                        & (contrib[band]['unmasked_fraction'] > 0)
                    )
                    if np.sum(msk) < MIN_EPOCHS:
                        keep = False

                if keep:
                    cell_has_data[cell_i, cell_j] = True

        # mdet only runs for cells whose neighbors also have
        # valid data, so only those go in the file
        for cell_i in range(1, NCELL_SIDE + 1):
            for cell_j in range(1, NCELL_SIDE + 1):
                if np.all(
                    cell_has_data[
                        cell_i - 1:cell_i + 2,
                        cell_j - 1:cell_j + 2,
                    ]
                ):
                    cij = CellIJ(i=cell_i, j=cell_j)
                    ci = patch_info.getCellInfo(cij.to_legacy())
                    bb = ci.getInnerBBox()
                    x_center = 0.5 * (bb.beginX + bb.endX)
                    y_center = 0.5 * (bb.beginY + bb.endY)

                    center_pos = lsst.geom.Point2D(
                        x_center, y_center,
                    )

                    sky_pos = wcs.pixelToSky(center_pos)
                    ra_center = sky_pos.getRa().asDegrees()
                    dec_center = sky_pos.getDec().asDegrees()

                    cell_info_list.append(get_cell_info_struct(
                        tract=tract,
                        patch=patch,
                        cell_i=cell_i,
                        cell_j=cell_j,
                        ra_center=ra_center,
                        dec_center=dec_center,
                    ))

    if len(cell_info_list) == 0:
        return None

    return np.concatenate(cell_info_list)


def go(args):
    import numpy as np
    import rustfits
    from lsst.daf.butler import Butler
    from tqdm import tqdm

    butler = Butler(args.repo, collections=args.collections)
    skymap = butler.get('skyMap', skymap=SKYMAP_VERS)

    if args.tracts is not None:
        tract_infos = [skymap[tract] for tract in args.tracts]
    else:
        tract_infos = list(skymap)

    ntracts = len(tract_infos)
    print('ntracts:', ntracts)

    cell_info_list = []

    for tract_info in tqdm(tract_infos, ascii=True, ncols=70):
        ci = get_good_cells_for_tract(
            butler=butler,
            skymap=skymap,
            tract=tract_info.getId(),
            bands=args.bands,
            verbose=True,
        )
        if ci is not None:
            cell_info_list.append(ci)

    ngot = len(cell_info_list)
    print(f'kept {ngot}/{ntracts} {ngot / ntracts:.2f}')

    cell_info = np.concatenate(cell_info_list)
    print('writing:', args.outfile)
    with rustfits.FITS(args.outfile, 'w+') as fits:
        fits.write_table(cell_info, compress=True)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--outfile', required=True,
                        help='output fits file, e.g. '
                             'good-cells-2026-08-28.fits; '
                             'required so runs are dated '
                             'deliberately')
    parser.add_argument('--bands', nargs='+',
                        default=DEFAULT_BANDS)
    parser.add_argument('--tracts', nargs='+', type=int,
                        help='only these tracts, e.g. for '
                             'testing; default is the whole '
                             'skymap')
    parser.add_argument('--repo', default=BUTLER_REPO,
                        help='butler repo path or alias')
    parser.add_argument('--collections', nargs='+',
                        default=BUTLER_COLLECTIONS)
    return parser.parse_args()


def main():
    go(get_args())


if __name__ == '__main__':
    main()
