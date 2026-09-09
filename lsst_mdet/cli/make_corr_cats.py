"""
assemble the input catalogs for shear correlations with treecorr,
e.g. the tangential shear around stars.

Two catalogs are written.  The galaxy catalog holds ra, dec, g1,
g2 and the weight for the standard galaxy sample: the selection
and weights of the stats machinery, driven by the same yaml config
(only the select stages are used), on the unsheared step.  The
global shear response from the sheared steps is stored in the
header as R, and the selection config in a 'config' extension.
The star catalog holds ra, dec and the G magnitude of the gaia
stars used in the processing, deduplicated across the per-tract
files and cut to the coarse fracdet neighborhood of the footprint
(so masked stars, whose own positions fall in the holes, are
kept).  Galaxies are cut to the footprint with an exact lookup,
so a footprint carrying the star exclusion holes
(lsst-mdet-make-star-exclusion) applies them here.

    lsst-mdet-make-corr-cats --run-dir . \\
        --config stats/sums_config.yaml --nproc 16
"""
import os

import numpy as np

DEFAULT_NPROC = 8
CAT_CHUNK = 200

# the footprint used for the galaxy cut: galaxies are kept only
# where the map is True (an exact lookup, unlike the coarse
# fracdet neighborhood test of the star cut), in every shear step
# so the response sums see the same position cut.  Loaded in the
# parent before the fork pool so the workers share it
SELECT_FOOTPRINT = None


def get_gals_file(run_dir):
    run = os.path.basename(os.path.abspath(run_dir))
    return os.path.join(run_dir, f'{run}-corr-gals.fits')


def get_stars_file(run_dir):
    run = os.path.basename(os.path.abspath(run_dir))
    return os.path.join(run_dir, f'{run}-corr-stars.fits')


def make_gal_cat(flist, config, nproc=1):
    """
    the galaxy correlation catalog and the response sums

    Returns
    -------
    gals: array with ra, dec, g1, g2, w
    rsums: dict of the 1p/1m weight and g1 sums
    """
    from ..hmaps import _get_pool, _pool_map

    chunks = [
        (flist[i:i + CAT_CHUNK], config)
        for i in range(0, len(flist), CAT_CHUNK)
    ]
    with _get_pool(nproc) as pool:
        results = _pool_map(pool, _gal_chunk, chunks)

    gals = np.concatenate([r[0] for r in results])
    rsums = results[0][1]
    for _, rs in results[1:]:
        for key in rsums:
            rsums[key] += rs[key]
    return gals, rsums


def _gal_chunk(task):
    import rustfits
    from .stats import apply_selection, get_weights

    fnames, config = task

    parts = []
    rsums = {'w1p': 0.0, 'g1p': 0.0, 'w1m': 0.0, 'g1m': 0.0}
    for fname in fnames:
        gals = apply_selection(rustfits.read(fname), config)
        if SELECT_FOOTPRINT is not None:
            keep = SELECT_FOOTPRINT.get_values_pos(
                gals['ra'], gals['dec'],
            )
            gals = gals[keep]
        weights = get_weights(gals)

        wns, = np.where(gals['mcal_step'] == 'ns')
        part = np.zeros(wns.size, dtype=[
            ('ra', 'f8'), ('dec', 'f8'),
            ('g1', 'f8'), ('g2', 'f8'), ('w', 'f8'),
        ])
        for col in ('ra', 'dec', 'g1', 'g2'):
            part[col] = gals[col][wns]
        part['w'] = weights[wns]
        parts.append(part)

        for step in ('1p', '1m'):
            ws, = np.where(gals['mcal_step'] == step)
            rsums[f'w{step}'] += weights[ws].sum()
            rsums[f'g{step}'] += (weights[ws] * gals['g1'][ws]).sum()

    return np.concatenate(parts), rsums


def get_response(rsums):
    return (rsums['g1p'] / rsums['w1p']
            - rsums['g1m'] / rsums['w1m']) / 0.02


def make_star_cat(gaia_pattern, footprint_file=None):
    """
    the star catalog: ra, dec and G magnitude from the per-tract
    gaia files, deduplicated on source_id and, when a footprint
    map is given, cut to the footprint
    """
    import glob
    import rustfits

    flist = sorted(glob.glob(gaia_pattern.replace('{tract:05d}', '*')))
    if len(flist) == 0:
        raise RuntimeError(f'no gaia files match {gaia_pattern}')
    print(f'{len(flist)} gaia files')

    parts = []
    for fname in flist:
        with rustfits.FITS(fname) as fits:
            parts.append(fits[1].read(
                columns=['source_id', 'ra', 'dec', 'phot_g_mean_mag'],
            ))
    data = np.concatenate(parts)

    _, iuniq = np.unique(data['source_id'], return_index=True)
    print(f'{data.size} rows, {iuniq.size} unique stars')
    data = data[iuniq]

    if footprint_file is not None:
        import healsparse
        fp = healsparse.HealSparseMap.read(footprint_file)
        # test the star's neighborhood, not its center: bright
        # stars are masked out of the footprint (their own
        # positions fall in the holes), but the measurement around
        # them is exactly the point.  A coarse fracdet keeps any
        # star within an arcminute-scale pixel that has coverage
        frac = fp.fracdet_map(4096)
        inside = frac.get_values_pos(data['ra'], data['dec']) > 0
        print(f'{inside.sum()} of {data.size} stars near the '
              'footprint')
        data = data[inside]

    stars = np.zeros(data.size, dtype=[
        ('ra', 'f8'), ('dec', 'f8'), ('gmag', 'f8'),
    ])
    stars['ra'] = data['ra']
    stars['dec'] = data['dec']
    stars['gmag'] = data['phot_g_mean_mag']
    return stars


def go(args):
    import rustfits
    from .stats import describe_selection, load_config, _write_config
    from .make_footprint import get_footprint_file, read_flist
    from .make_map import get_catalog_flist

    config = load_config(args.config)
    print('selection:')
    print(describe_selection(config))

    if args.flist is not None:
        flist = read_flist(args.flist)
    else:
        flist = get_catalog_flist(args.run_dir)
    if len(flist) == 0:
        raise RuntimeError(f'no catalogs found under {args.run_dir}')

    gals_file = args.gals_output or get_gals_file(args.run_dir)
    stars_file = args.stars_output or get_stars_file(args.run_dir)
    for fname in (gals_file, stars_file):
        if os.path.exists(fname) and not args.clobber:
            raise RuntimeError(f'{fname} exists and clobber is False')

    footprint_file = args.footprint
    if footprint_file is None:
        footprint_file = get_footprint_file(args.run_dir)
        if not os.path.exists(footprint_file):
            footprint_file = None

    stars = make_star_cat(args.gaia_pattern, footprint_file)
    print('writing:', stars_file)
    with rustfits.FITS(stars_file, 'w+') as fits:
        fits.write_table(stars, extname='stars', compress=True)

    if footprint_file is not None:
        import healsparse
        global SELECT_FOOTPRINT
        print('galaxy selection footprint:', footprint_file)
        SELECT_FOOTPRINT = healsparse.HealSparseMap.read(
            footprint_file,
        )

    print(f'{len(flist)} catalogs')
    gals, rsums = make_gal_cat(flist, config, nproc=args.nproc)
    resp = get_response(rsums)
    print(f'{gals.size} galaxies, R = {resp:.5f}')

    print('writing:', gals_file)
    with rustfits.FITS(gals_file, 'w+') as fits:
        fits.write_table(
            gals, extname='gals', compress=True,
            header={'R': resp},
        )
        _write_config(fits, config)


def get_args():
    import argparse
    from .make_gaia import GAIA_PATTERN

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--run-dir', default='.',
                        help='the run directory holding the '
                             '{tract}/{tract}-{patch}-mdet.fits '
                             'catalogs')
    parser.add_argument('--flist',
                        help='a file listing the catalogs to use, '
                             'instead of every catalog under the '
                             'run directory')
    parser.add_argument('--config', required=True,
                        help='the stats yaml config; only the '
                             'select stages are used')
    parser.add_argument('--gaia-pattern', default=GAIA_PATTERN,
                        help='the per-tract gaia star files')
    parser.add_argument('--footprint',
                        help='footprint map: stars are cut to its '
                             'coarse fracdet neighborhood, galaxies '
                             'to an exact lookup (so the star '
                             'exclusion holes apply); default the '
                             'run footprint, no cut if absent')
    parser.add_argument('--gals-output',
                        help='default <run-dir>/<run>-corr-gals.fits')
    parser.add_argument('--stars-output',
                        help='default <run-dir>/<run>-corr-stars.fits')
    parser.add_argument('--nproc', type=int, default=DEFAULT_NPROC,
                        help='processes for reading the catalogs')
    parser.add_argument('--clobber', action='store_true',
                        help='overwrite existing outputs')

    args = parser.parse_args()
    if args.nproc < 1:
        parser.error('--nproc must be >= 1')
    return args


def main():
    go(get_args())


if __name__ == '__main__':
    main()
