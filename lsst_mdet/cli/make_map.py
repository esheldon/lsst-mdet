"""
map a quantity over the sky from the per-patch catalogs: the
weighted mean per healpix pixel, written as a healsparse map and
rendered to a png with skyproj.

The selection and weights are those of the stats machinery, driven
by the same yaml config as lsst-mdet-dosums (only the select stages
are used here).  The quantity is any catalog column or derived
value (see stats.get_named_value), measured on the unsheared step.
g1 and g2 are divided by the survey-global response, computed from
the sheared steps in the same pass; everything else is a plain
weighted mean.

    lsst-mdet-make-map --run-dir . --config stats/sums_config.yaml \\
        --quantity g1 --nside 32 --nproc 16
"""
import os

import numpy as np

DEFAULT_NPROC = 8
DEFAULT_NSIDE = 32
DEFAULT_MIN_N = 1000
MAP_CHUNK = 200

# steps whose g sums give the response; ns is the measurement step
SHEAR_STEPS = ('1p', '1m', '2p', '2m')


def get_catalog_flist(run_dir):
    """
    every per-patch catalog of a run, sorted: the
    <tract>/<tract>-<patch>-mdet.fits files
    """
    import glob
    pattern = os.path.join(run_dir, '[0-9]*', '*-mdet.fits')
    return sorted(glob.glob(pattern))


def get_map_file(run_dir, quantity, nside):
    run = os.path.basename(os.path.abspath(run_dir))
    return os.path.join(run_dir, f'{run}-map-{quantity}-nside{nside}.hsp')


def make_map(flist, quantity, nside, config, nproc=1):
    """
    accumulate the per-pixel sums over the catalogs

    Parameters
    ----------
    flist: list of str
        The per-patch catalog files
    quantity: str
        Column or derived value name
    nside: int
        Healpix nside of the map (nest)
    config: dict
        The stats selection config, from stats.load_config
    nproc: int
        Processes for reading the catalogs

    Returns
    -------
    dict with the per-pixel arrays n, wsum, wq (weighted quantity
    sums, full sky length 12 * nside**2) and the global response
    sums rsums
    """
    from ..hmaps import _get_pool, _pool_map

    chunks = [
        (flist[i:i + MAP_CHUNK], quantity, nside, config)
        for i in range(0, len(flist), MAP_CHUNK)
    ]
    with _get_pool(nproc) as pool:
        results = _pool_map(pool, _map_chunk, chunks)

    total = results[0]
    for res in results[1:]:
        total['n'] += res['n']
        total['wsum'] += res['wsum']
        total['wq'] += res['wq']
        for key in total['rsums']:
            total['rsums'][key] += res['rsums'][key]
    return total


def _map_chunk(task):
    import hpgeom
    import rustfits
    from .stats import apply_selection, get_named_value, get_weights

    fnames, quantity, nside, config = task

    npix = 12 * nside ** 2
    n = np.zeros(npix, dtype='i8')
    wsum = np.zeros(npix)
    wq = np.zeros(npix)
    rsums = {f'{k}{s}': 0.0 for s in SHEAR_STEPS for k in ('w', 'g')}

    for fname in fnames:
        gals = apply_selection(rustfits.read(fname), config)
        weights = get_weights(gals)

        wns, = np.where(gals['mcal_step'] == 'ns')
        pix = hpgeom.angle_to_pixel(
            nside, gals['ra'][wns], gals['dec'][wns], nest=True,
        )
        vals = get_named_value(gals[wns], quantity)
        np.add.at(n, pix, 1)
        np.add.at(wsum, pix, weights[wns])
        np.add.at(wq, pix, weights[wns] * vals)

        # global response sums: g1 from the 1p/1m steps, g2 from
        # 2p/2m
        for step in SHEAR_STEPS:
            ws, = np.where(gals['mcal_step'] == step)
            comp = 'g1' if step[0] == '1' else 'g2'
            rsums[f'w{step}'] += weights[ws].sum()
            rsums[f'g{step}'] += (weights[ws] * gals[comp][ws]).sum()

    return {'n': n, 'wsum': wsum, 'wq': wq, 'rsums': rsums}


def get_responses(rsums):
    """
    the global responses R1, R2 from the shear step sums.  Runs
    store only the 1p/1m steps (the stats machinery measures R from
    g1 and uses it for both components); R2 falls back to R1 when
    the 2p/2m steps are absent
    """
    r1 = (rsums['g1p'] / rsums['w1p']
          - rsums['g1m'] / rsums['w1m']) / 0.02
    if rsums['w2p'] > 0 and rsums['w2m'] > 0:
        r2 = (rsums['g2p'] / rsums['w2p']
              - rsums['g2m'] / rsums['w2m']) / 0.02
    else:
        r2 = r1
    return r1, r2


def make_hsp_map(total, quantity, nside, min_n):
    """
    the healsparse map of the weighted mean, over pixels with at
    least min_n objects; g1 and g2 are divided by the global
    response
    """
    import healsparse

    pix, = np.where(total['n'] >= min_n)
    if pix.size == 0:
        raise RuntimeError(f'no pixels with at least {min_n} objects')
    vals = total['wq'][pix] / total['wsum'][pix]

    r1, r2 = get_responses(total['rsums'])
    if quantity == 'g1':
        vals /= r1
    elif quantity == 'g2':
        vals /= r2

    nside_coverage = min(32, nside)
    hsp_map = healsparse.HealSparseMap.make_empty(
        nside_coverage, nside, dtype='f8',
    )
    hsp_map[pix] = vals
    return hsp_map


def render_map(hsp_map, quantity=None, title=None, cmap=None,
               vmin=None, vmax=None, xsize=2000, label=None,
               ra_range=None, dec_range=None):
    """
    draw the map with skyproj, zoomed to the covered area or to the
    given ra/dec window, with a colorbar; shared by the png writer
    and lsst-mdet-view-map.  The quantities g1/g2 default to a
    diverging map on a symmetric scale about zero; everything else
    to the 2-98 percentile range.  ra values past 360 express a
    window crossing ra = 0; interactively the window is only the
    starting view

    Returns
    -------
    fig, sp: the matplotlib figure and the Skyproj
    """
    import matplotlib.pyplot as plt
    import skyproj

    from .plot_footprint import fit_figure_to_map

    if (ra_range is None) != (dec_range is None):
        raise ValueError('give both ra_range and dec_range, or neither')

    vals = hsp_map[hsp_map.valid_pixels]
    is_shear = quantity in ('g1', 'g2')
    if vmin is None or vmax is None:
        lo, hi = np.percentile(vals, [2, 98])
        if is_shear:
            sym = max(abs(lo), abs(hi))
            lo, hi = -sym, sym
        if vmin is None:
            vmin = lo
        if vmax is None:
            vmax = hi
    if cmap is None:
        cmap = 'RdBu_r' if is_shear else 'viridis'
    if label is None:
        label = f'{quantity}/R' if is_shear else quantity

    fig, ax = plt.subplots(figsize=(14, 7))
    if ra_range is not None:
        lon_0 = np.mean(ra_range) % 360
        sp = skyproj.McBrydeSkyproj(ax=ax, lon_0=lon_0)
        sp.draw_hspmap(
            hsp_map, zoom=False, lon_range=ra_range,
            lat_range=dec_range, xsize=xsize, vmin=vmin, vmax=vmax,
            cmap=cmap,
        )
    else:
        sp = skyproj.McBrydeSkyproj(ax=ax)
        sp.draw_hspmap(
            hsp_map, xsize=xsize, vmin=vmin, vmax=vmax, cmap=cmap,
        )

    fit_figure_to_map(fig, sp)
    sp.draw_colorbar(label=label)
    if title is not None:
        sp.ax.set_title(title, pad=30)
    return fig, sp


def plot_map(hsp_map, output, quantity, title, cmap=None,
             vmin=None, vmax=None, ra_range=None, dec_range=None):
    """
    render the map to an image file
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, _ = render_map(
        hsp_map, quantity=quantity, title=title, cmap=cmap,
        vmin=vmin, vmax=vmax, ra_range=ra_range, dec_range=dec_range,
    )
    print('writing', output)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)


def go(args):
    import hpgeom
    from .stats import describe_selection, load_config
    from .make_footprint import read_flist

    if args.plot_only:
        import healsparse

        output = args.output
        if output is None:
            output = get_map_file(args.run_dir, args.quantity, args.nside)
        if not os.path.exists(output):
            raise RuntimeError(f'no map to plot: {output}')

        hsp_map = healsparse.HealSparseMap.read(output)
        png = args.png
        if png is None:
            png = os.path.splitext(output)[0] + '.png'
        title = os.path.basename(png).replace('.png', '')
        plot_map(
            hsp_map, png,
            args.quantity, title, cmap=args.cmap,
            vmin=args.vmin, vmax=args.vmax,
            ra_range=args.ra_range, dec_range=args.dec_range,
        )
        return

    config = load_config(args.config)
    print('selection:')
    print(describe_selection(config))

    if args.flist is not None:
        flist = read_flist(args.flist)
    else:
        flist = get_catalog_flist(args.run_dir)
    if len(flist) == 0:
        raise RuntimeError(f'no catalogs found under {args.run_dir}')

    output = args.output
    if output is None:
        output = get_map_file(args.run_dir, args.quantity, args.nside)
    if os.path.exists(output) and not args.clobber:
        raise RuntimeError(f'{output} exists and clobber is False')

    print(f'{len(flist)} catalogs')
    total = make_map(
        flist, args.quantity, args.nside, config, nproc=args.nproc,
    )
    hsp_map = make_hsp_map(total, args.quantity, args.nside, args.min_n)

    r1, r2 = get_responses(total['rsums'])
    used = total['n'][hsp_map.valid_pixels]
    mean = (total['wq'].sum() / total['wsum'].sum())
    print(f'objects: {total["n"].sum()}')
    print(f'R1: {r1:.5f}  R2: {r2:.5f}')
    print(f'survey mean {args.quantity}: {mean:.6g}')
    print(f'pixels with >= {args.min_n} objects: {used.size} '
          f'(median {np.median(used):.0f} objects, '
          f'{hpgeom.nside_to_pixel_area(args.nside, degrees=True):.3g} '
          'deg^2 each)')

    print('writing', output)
    hsp_map.write(output, clobber=args.clobber)

    png = args.png
    if png is None:
        png = os.path.splitext(output)[0] + '.png'
    title = os.path.basename(output).replace('.hsp', '')
    plot_map(
        hsp_map, png, args.quantity,
        title, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
        ra_range=args.ra_range, dec_range=args.dec_range,
    )


def get_args():
    import argparse
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
    parser.add_argument('--config',
                        help='the stats yaml config; only the '
                             'select stages are used.  Required '
                             'unless --plot-only')
    parser.add_argument('--plot-only', action='store_true',
                        help='re-render the png from the existing '
                             'map instead of remaking it')
    parser.add_argument('--quantity', default='g1',
                        help='column or derived value to map; g1 '
                             'and g2 are divided by the global '
                             'response')
    parser.add_argument('--nside', type=int, default=DEFAULT_NSIDE,
                        help='healpix nside of the map')
    parser.add_argument('--min-n', type=int, default=DEFAULT_MIN_N,
                        help='keep pixels with at least this many '
                             'objects')
    parser.add_argument('--output',
                        help='output map; default <run-dir>/<run>-'
                             'map-<quantity>-nside<nside>.hsp, with '
                             'the png beside it')
    parser.add_argument('--png',
                        help='output png; default the map file '
                             'with a .png extension')
    parser.add_argument('--nproc', type=int, default=DEFAULT_NPROC,
                        help='processes for reading the catalogs')
    parser.add_argument('--vmin', type=float,
                        help='color scale minimum; default 2nd '
                             'percentile, symmetric for g1/g2')
    parser.add_argument('--vmax', type=float,
                        help='color scale maximum; default 98th '
                             'percentile, symmetric for g1/g2')
    parser.add_argument('--cmap',
                        help='matplotlib colormap; default RdBu_r '
                             'for g1/g2, viridis otherwise')
    parser.add_argument('--ra-range', type=float, nargs=2,
                        metavar=('LOW', 'HIGH'),
                        help='view window ra range for the png; '
                             'values past 360 express a range '
                             'crossing ra = 0')
    parser.add_argument('--dec-range', type=float, nargs=2,
                        metavar=('LOW', 'HIGH'),
                        help='view window dec range for the png')
    parser.add_argument('--clobber', action='store_true',
                        help='overwrite an existing output')

    args = parser.parse_args()
    if args.nproc < 1:
        parser.error('--nproc must be >= 1')
    if not args.plot_only and args.config is None:
        parser.error('--config is required unless --plot-only')
    if (args.ra_range is None) != (args.dec_range is None):
        parser.error('give both --ra-range and --dec-range, or neither')
    return args


def main():
    go(get_args())


if __name__ == '__main__':
    main()
