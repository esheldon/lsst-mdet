"""
view a healsparse map interactively with skyproj: the same
rendering as the lsst-mdet-make-map png (colorbar, zoom to the
covered area, the g1/g2 conventions via --quantity) but in a live
matplotlib window with pan and zoom.

A bit-packed map (a footprint) is shown as its fractional coverage,
at a resolution matched to the raster or set with --nside.

    lsst-mdet-view-map run-dp2-v00-map-g1-nside32.hsp \\
        --quantity g1 --vmin -0.007 --vmax 0.007
    lsst-mdet-view-map run-dp2-v00-footprint.hsp
"""
import os


def go(args):
    import healsparse
    import matplotlib.pyplot as plt

    from .make_map import render_map
    from .plot_footprint import choose_nside

    hsp_map = healsparse.HealSparseMap.read(args.fname)

    label = args.label
    if hsp_map.is_bit_packed_map:
        nside = args.nside
        if nside is None:
            nside = choose_nside(hsp_map, None, args.xsize)
        print(f'bit-packed map; showing the fracdet at nside {nside}')
        hsp_map = hsp_map.fracdet_map(nside)
        if label is None:
            label = 'coverage fraction'

    title = args.title
    if title is None:
        title = os.path.basename(args.fname)

    render_map(
        hsp_map, quantity=args.quantity, title=title, cmap=args.cmap,
        vmin=args.vmin, vmax=args.vmax, xsize=args.xsize, label=label,
        ra_range=args.ra_range, dec_range=args.dec_range,
    )
    plt.show()


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('fname', help='the healsparse map to view')
    parser.add_argument('--quantity',
                        help='the mapped quantity; g1 and g2 get '
                             'the diverging colormap, symmetric '
                             'scale and /R label of make-map')
    parser.add_argument('--vmin', type=float,
                        help='color scale minimum; default 2nd '
                             'percentile, symmetric for g1/g2')
    parser.add_argument('--vmax', type=float,
                        help='color scale maximum; default 98th '
                             'percentile, symmetric for g1/g2')
    parser.add_argument('--cmap',
                        help='matplotlib colormap; default RdBu_r '
                             'for g1/g2, viridis otherwise')
    parser.add_argument('--label',
                        help='colorbar label; default from '
                             '--quantity')
    parser.add_argument('--title',
                        help='default is the map file name')
    parser.add_argument('--ra-range', type=float, nargs=2,
                        metavar=('LOW', 'HIGH'),
                        help='starting view window ra range; values '
                             'past 360 express a range crossing '
                             'ra = 0')
    parser.add_argument('--dec-range', type=float, nargs=2,
                        metavar=('LOW', 'HIGH'),
                        help='starting view window dec range')
    parser.add_argument('--xsize', type=int, default=2000,
                        help='raster width in pixels')
    parser.add_argument('--nside', type=int,
                        help='fracdet resolution for a bit-packed '
                             'map; default matches the raster')
    args = parser.parse_args()
    if (args.ra_range is None) != (args.dec_range is None):
        parser.error('give both --ra-range and --dec-range, or neither')
    return args


def main():
    go(get_args())


if __name__ == '__main__':
    main()
