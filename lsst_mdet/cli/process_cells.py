"""
cli/process_cells

process one patch.  The pieces are also used by the node driver
(process_node), which runs many patches on a node from a single
process that imports everything once and then forks workers:

    get_parser(): the argument parser, so the driver can accept
        the same processing options
    preload(): import everything the processing imports lazily,
        so the forked workers import nothing
    process_patch(args): run one patch from a parsed namespace
"""
import numpy as np
from ..apodize import apodize_mbobs
from ..cells import (
    load_coadds_butler, pull_mbobs, get_cell_healsparse_polygon,
    get_tract_primary,
)
from ..defaults import BUTLER_COLLECTIONS, BUTLER_REPO, SKYMAP_VERS
from ..starsub import GSUB
from ..hmaps import (
    make_empty_footprint,
    mask_stars_in_footprint,
    trim_footprint_to_tract_bounds,
)
from ..patchfiles import load_coadds_files
from ..io import write_color_image, write_output
from ..pipeline import do_metacal_and_process, process_one_mbobs
from ..psf import fit_and_set_psfrec
from ..qa import write_star_residual_qa
from ..wcs import calculate_positions

# retries for opening the butler when the registry database refuses
# the connection; see open_butler
BUTLER_NTRIES = 8
BUTLER_RETRY_SLEEP = 5.0
BUTLER_RETRY_MAX_SLEEP = 120.0


def get_parser(per_patch=True):
    """
    get the argument parser

    Parameters
    ----------
    per_patch: bool
        If True, include the arguments that identify a single patch
        (--tract, --patch, --seed, --outfile).  The node driver leaves
        these out and takes them from its job list instead
    """
    import argparse
    parser = argparse.ArgumentParser()
    if per_patch:
        parser.add_argument('--tract', type=int, required=True)
        parser.add_argument('--patch', type=int, required=True)
        parser.add_argument('--seed', type=int, required=True)
        parser.add_argument('--outfile', required=True)
    parser.add_argument('--model', required=True)
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
    gaia = parser.add_mutually_exclusive_group()
    gaia.add_argument(
        '--gaia-file',
        help='read the gaia stars from this parquet file '
             '(columns gaia_g_mag, ra, dec) instead of the '
             'TAP query (butler mode with --starsub only)',
    )
    gaia.add_argument(
        '--gaia-pattern',
        help='as --gaia-file, but a pattern with {tract} and '
             '{patch} placeholders, filled in per patch without '
             'zero padding, e.g. /path/{tract}/{patch}/gaia.parq',
    )
    parser.add_argument('--deblend', action='store_true')
    parser.add_argument('--s2-detect', action='store_true')
    parser.add_argument(
        '--redo-bg', action=argparse.BooleanOptionalAction,
        default=None,
        help='redo the background determination (butler mode; '
             'on by default there, matching getimages). '
             'Refused with --patch-dir, where the patch files '
             'already carry it',
    )
    parser.add_argument(
        '--starsub', action=argparse.BooleanOptionalAction,
        default=False,
        help='subtract and mask the Gaia stars at the patch '
             'level before the background redo (butler mode; '
             'refused with --patch-dir)',
    )
    parser.add_argument(
        '--gsub', type=float, default=GSUB,
        help='subtract and mask Gaia stars brighter than '
             'this; the download depth follows it (butler '
             'mode only)',
    )
    parser.add_argument(
        '--apod-stars', action=argparse.BooleanOptionalAction,
        default=True,
        help='zero the star-mask regions in the image and '
             'noise planes with a smooth taper (butler mode '
             'only)',
    )
    parser.add_argument('--mdet', action='store_true')
    parser.add_argument(
        '--cells', nargs='+', type=parse_cell,
        help='process only these cells, given as i,j with 1-20 '
             'for each, e.g. --cells 10,10 11,12.  Used for '
             'debugging and for the node driver warmup',
    )
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--show', action='store_true')
    return parser


def parse_cell(text):
    """
    parse a cell 'i,j' into (i, j)
    """
    import argparse
    try:
        cell_i, cell_j = (int(v) for v in text.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(f'cell must be i,j, got {text!r}')

    if not (1 <= cell_i <= 20 and 1 <= cell_j <= 20):
        raise argparse.ArgumentTypeError(
            f'cell indices must be 1-20, got {text!r}'
        )

    return cell_i, cell_j


def get_args():
    return get_parser().parse_args()


def preload(
    repo=BUTLER_REPO,
    collections=BUTLER_COLLECTIONS,
    tract=None,
    patch=None,
    band='r',
    patch_dir=None,
):
    """
    import everything main() imports lazily, so that a driver which
    calls this and then forks workers does no further imports in the
    workers.  The butler is the main offender: opening it, reading the
    skymap and reading one coadd pulls in about a thousand modules
    through the registry database layer and the dataset formatters,
    which are imported on first use.  Give a tract and patch to also
    warm the coadd formatter.

    In patch_dir mode (--patch-dir) the butler is not used and is
    skipped.

    This covers the imports but not the numba compilation, which
    happens at the first call of each jitted function; the driver
    also runs one cell of a real patch (--cells) before forking so
    the compiled code is inherited too

    Parameters
    ----------
    repo, collections: str, list
        the butler repo and collections that will be used
    tract, patch: int, optional
        a patch to load one coadd from; skipped if not given
    band: str
        band of the coadd to load, default 'r'
    patch_dir: str, optional
        if set, the processing is file based and the butler is
        not warmed
    """
    import gc

    # the direct lazy imports in the package, and what those pull
    # in at first use
    from .. import background  # noqa
    import ngmix  # noqa
    import ngmix.moments  # noqa
    import ngmix.prepsfadmom.prep  # noqa
    import ngmix.prepsfadmom.full_errors  # noqa
    import metacal  # noqa
    import kdeblend  # noqa
    import kdeblend.vis  # noqa
    import kdeblend.full_errors  # noqa
    import fofx  # noqa
    import fofx.fofs  # noqa
    import fofx.vis  # noqa
    import sxdes  # noqa
    import sxdes.runner  # noqa
    import sep  # noqa
    import threadpoolctl  # noqa
    import healsparse  # noqa
    import hpgeom  # noqa
    import pandas  # noqa
    import tqdm  # noqa
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot  # noqa
    import matplotlib.backends.backend_agg  # noqa
    from astropy.visualization import make_lupton_rgb  # noqa
    from PIL import Image
    # the format plugins, loaded on the first open/save
    Image.init()

    if patch_dir is not None:
        return

    import lsst.geom  # noqa
    import lsst.images  # noqa
    import lsst.images._cell_grid  # noqa
    import lsst.images._geom  # noqa
    with open_butler(repo, collections=collections) as butler:
        butler.get('skyMap', skymap=SKYMAP_VERS)
        if tract is not None and patch is not None:
            data_id = {
                'tract': tract, 'patch': patch, 'band': band,
                'skymap': SKYMAP_VERS,
            }
            butler.get('deep_coadd', dataId=data_id)

    # make sure the registry database connection is gone before any
    # fork: a connection inherited by a child is shared with the parent
    # and both would be talking on the same socket
    del butler
    gc.collect()


def open_butler(
    repo, collections, ntries=BUTLER_NTRIES, base_sleep=BUTLER_RETRY_SLEEP,
    max_sleep=BUTLER_RETRY_MAX_SLEEP,
):
    """
    open the butler, retrying with an exponential backoff when the
    registry database refuses the connection.  Opening the butler
    connects to the registry, and when many patches on many nodes
    start at once the DP2 pgbouncer can hit its client connection
    limit, which shows up as an OperationalError ("no more
    connections allowed").  That is transient: the loads on the
    other nodes finish in about a minute, so waiting and trying
    again succeeds.

    The sleep before try n is uniform in [0, base_sleep * 2**(n-1)],
    capped at max_sleep, so concurrent retries are spread out.  With
    the defaults the total wait before giving up is at most about
    seven minutes.  Only OperationalError is retried; anything else
    is a real error

    Parameters
    ----------
    repo, collections: str, list
        the butler repo and collections
    ntries: int
        total number of attempts
    base_sleep, max_sleep: float
        the backoff parameters, seconds

    Returns
    -------
    butler: the Butler, usable as a context manager that closes it
    """
    import random
    import time
    from lsst.daf.butler import Butler
    from sqlalchemy.exc import OperationalError

    for itry in range(1, ntries + 1):
        try:
            return Butler(repo, collections=collections)
        except OperationalError as err:
            if itry == ntries:
                raise
            sleep = random.uniform(
                0, min(base_sleep * 2**(itry - 1), max_sleep),
            )
            # the DBAPI error (psycopg2) carries the server message;
            # the sqlalchemy wrapper adds a doc link after it
            message = str(getattr(err, 'orig', None) or err).strip()
            message = message.splitlines()[-1]
            print(f'butler open failed on try {itry}/{ntries}: '
                  f'{message}; retrying in {sleep:.0f} s', flush=True)
            time.sleep(sleep)


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
    gsub=GSUB,
    apod_stars=True,
    cells=None,
):
    """
    process one patch

    Parameters
    ----------
    cells: list of (cell_i, cell_j), optional
        process only these cells, e.g. [(10, 10)].  Default is all
        20x20 cells
    """
    import time
    from tqdm import trange

    tstart = time.time()

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
        redo_bg = False
        (deep_coadds, wcs, starmask, star_table, apod,
         tract_bounds, skyvars) = load_coadds_files(
            patch_dir=patch_dir, tract=tract, patch=patch,
            bands=bands,
        )
    else:
        if redo_bg is None:
            # match the getimages default: the background is
            # redone unless explicitly disabled
            redo_bg = True

        # the butler is only needed for the load; closing it right
        # after releases its registry database connection, which is
        # a shared and limited resource (the DP2 pgbouncer refuses
        # connections at its max_client_conn), rather than holding
        # it idle for the whole processing stage
        with open_butler(repo, collections=collections) as butler:
            (deep_coadds, wcs, starmask, star_table, apod,
             tract_bounds, skyvars) = load_coadds_butler(
                butler=butler, tract=tract, patch=patch,
                bands=bands, redo_bg=redo_bg, starsub=starsub,
                gaia_file=gaia_file, gsub=gsub,
                apod_stars=apod_stars,
            )
        del butler

    # the load is the part that hits the butler and the file system;
    # reported separately so contention shows up in the logs
    tload = time.time() - tstart
    print(f'load time: {tload:.1f} s')

    if progress:
        mrng_i = trange(1, 21, desc='cell_i', ncols=80, ascii=True)
    else:
        mrng_i = range(1, 21)

    if cells is not None:
        cells = set(cells)

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
            if cells is not None and (cell_i, cell_j) not in cells:
                continue

            ncell += 1

            mbobs, cell_meta = pull_mbobs(
                deep_coadds=deep_coadds,
                cell_i=cell_i,
                cell_j=cell_j,
                wcs=wcs,
                starmask=starmask,
                skyvars=skyvars,
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

    print(f'kept {nkeep}/{ncell} {nkeep / ncell:g}')
    print(f'process time: {time.time() - tstart - tload:.1f} s')

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
        run_options=dict(
            repo=repo,
            collections=collections,
            patch_dir=patch_dir,
            gaia_file=gaia_file,
            gsub=gsub,
            apod_stars=apod_stars,
            cells=cells,
        ),
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

    # reduced-resolution color image of the final masked
    # images, non-footprint area tinted, for inspecting gross
    # problems
    write_color_image(
        outfile.replace('.fits', '-color.jpg'), deep_coadds,
        wcs=wcs, footprint=footprint,
    )

    # stacked residual profiles of the subtracted stars: flat
    # and zero means the subtraction left nothing behind.  The
    # catalog, footprint and color image are already written, so
    # a failure here (e.g. the sep sub-object overflow seen on an
    # image artifact) must not turn a finished patch into a
    # failed job: warn and go on without the QA figure
    if star_table is not None and star_table.size > 0:
        try:
            write_star_residual_qa(
                outfile.replace('.fits', '-star-residuals.png'),
                deep_coadds, star_table, starmask,
            )
        except Exception:
            import traceback
            traceback.print_exc()
            print('WARNING: star residual QA failed, '
                  'continuing without it')


def get_gaia_file(args):
    """
    the gaia file for this patch: --gaia-file as given, or
    --gaia-pattern filled in with the tract and patch
    """
    if args.gaia_pattern is not None:
        return args.gaia_pattern.format(tract=args.tract, patch=args.patch)
    return args.gaia_file


def process_patch(args):
    """
    process one patch from a parsed argument namespace, as returned
    by get_parser().parse_args() or built by the node driver from its
    common options plus the per-patch tract, patch, seed and outfile
    """
    main(
        with_mdet=args.mdet,
        seed=args.seed,
        tract=args.tract,
        patch=args.patch,
        model=args.model,
        deblend=args.deblend,
        s2_detect=args.s2_detect,
        redo_bg=args.redo_bg,
        starsub=args.starsub,
        patch_dir=args.patch_dir,
        outfile=args.outfile,
        progress=args.progress,
        show=args.show,
        repo=args.repo,
        collections=args.collections,
        gaia_file=get_gaia_file(args),
        gsub=args.gsub,
        apod_stars=args.apod_stars,
        cells=args.cells,
    )


def main_cli():
    process_patch(get_args())


if __name__ == '__main__':
    main_cli()
