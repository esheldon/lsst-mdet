"""
plot a footprint healsparse map with skyproj: the fractional
coverage (fracdet) in a McBryde projection, optionally zoomed to an
ra/dec region.  The ra range may extend past 360 to express a range
crossing ra = 0.  The fracdet resolution defaults to matching the
raster resolution and can be set with --nside.

    lsst-mdet-plot-footprint run-dp2-v00-footprint.hsp
    lsst-mdet-plot-footprint run-dp2-v00-footprint.hsp \\
        --ra-range 280 395 --dec-range -34 5 --output wide.png
"""
import os

import numpy as np

from .make_gaia import DEFAULT_MIN_ABS_B

DEFAULT_XSIZE = 2000
DEFAULT_DPI = 150
MAX_AUTO_NSIDE = 8192


def choose_nside(hsp_map, ra_range, xsize):
    """
    the smallest power-of-two nside whose pixels are no larger than
    the raster pixels, clipped to the map's coverage and sparse
    resolutions and to MAX_AUTO_NSIDE
    """
    if ra_range is not None:
        ra_span = ra_range[1] - ra_range[0]
    else:
        ra_span = 360.0

    # a healpix pixel is about (58.6 / nside) deg on a side
    target = 58.6 * xsize / ra_span
    nside = 2 ** int(np.ceil(np.log2(target)))
    nside = max(nside, hsp_map.nside_coverage)
    nside = min(nside, hsp_map.nside_sparse, MAX_AUTO_NSIDE)
    return nside


def fit_figure_to_map(fig, sp):
    """
    resize the figure to the projected aspect of the drawn map: the
    equal-aspect projection leaves the map shorter than the axes
    cell for wide extents, and the colorbar is sized to the cell,
    so match the cell to the map
    """
    x0, x1 = sp.ax.get_xlim()
    y0, y1 = sp.ax.get_ylim()
    aspect = abs(x1 - x0) / abs(y1 - y0)
    width = fig.get_size_inches()[0]
    height = np.clip(width / aspect, 3.0, 1.5 * width)
    fig.set_size_inches(width, height)


def draw_gal_b_lines(sp, lon_0, min_abs_b):
    """
    draw the galactic latitude cut as dashed curves at b of
    +min_abs_b and -min_abs_b
    """
    from astropy.coordinates import SkyCoord

    wrap = (lon_0 + 180.0) % 360
    for i, b in enumerate((min_abs_b, -min_abs_b)):
        gl = np.linspace(0.0, 360.0, 721)
        crd = SkyCoord(
            l=gl, b=np.full(gl.size, b), frame='galactic', unit='deg',
        ).icrs
        ra = crd.ra.deg
        dec = crd.dec.deg
        # the curve winds once around the sky in ra; order it away
        # from the projection wrap so no segment crosses the wrap
        order = np.argsort((ra - wrap) % 360)
        label = f'galactic $|b| = {min_abs_b:g}^\\circ$' if i == 0 else None
        sp.ax.plot(
            ra[order], dec[order], color='gray', linestyle='dashed',
            linewidth=1, label=label,
        )
    sp.ax.legend(loc='upper left', fontsize=10)


def plot_footprint(
    fname, output, ra_range=None, dec_range=None, nside=None,
    xsize=DEFAULT_XSIZE, dpi=DEFAULT_DPI, title=None, cmap=None,
    min_abs_b=DEFAULT_MIN_ABS_B,
):
    """
    render the fracdet of a footprint map to an image file

    Parameters
    ----------
    fname: str
        The footprint healsparse map
    output: str
        The output image file
    ra_range, dec_range: [low, high], optional
        Zoom to this region; give both or neither.  ra values past
        360 express a range crossing ra = 0
    nside: int, optional
        fracdet resolution; default matches the raster resolution
    xsize: int
        Raster width in pixels
    dpi: int
        Output resolution
    title: str, optional
        Default is the map file name and the area
    cmap: str, optional
        matplotlib colormap name; default is skyproj's default
    min_abs_b: float
        Draw the galactic latitude cut lines at |b| = this value;
        <= 0 draws none.  Default is the production cut
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import healsparse
    import hpgeom
    import skyproj

    if (ra_range is None) != (dec_range is None):
        raise ValueError('give both ra_range and dec_range, or neither')

    print('reading', fname)
    hsp_map = healsparse.HealSparseMap.read(fname)

    if nside is None:
        nside = choose_nside(hsp_map, ra_range, xsize)

    print(f'fracdet at nside {nside}')
    frac = hsp_map.fracdet_map(nside)

    area = (
        frac[frac.valid_pixels].sum()
        * hpgeom.nside_to_pixel_area(nside, degrees=True)
    )
    if title is None:
        title = f'{os.path.basename(fname)} ({area:.1f} deg$^2$)'

    fig, ax = plt.subplots(figsize=(14, 7))
    if ra_range is not None:
        lon_0 = np.mean(ra_range) % 360
        sp = skyproj.McBrydeSkyproj(ax=ax, lon_0=lon_0)
        sp.draw_hspmap(
            frac, zoom=False, lon_range=ra_range, lat_range=dec_range,
            xsize=xsize, vmin=0, vmax=1, cmap=cmap,
        )
    else:
        lon_0 = 0.0
        sp = skyproj.McBrydeSkyproj(ax=ax)
        sp.draw_hspmap(frac, xsize=xsize, vmin=0, vmax=1, cmap=cmap)

    if min_abs_b > 0:
        draw_gal_b_lines(sp, lon_0, min_abs_b)

    fit_figure_to_map(fig, sp)
    sp.draw_colorbar(label='coverage fraction')
    # skyproj replaces the axes it is given; set the title on its
    # own, padded above the top ra labels
    sp.ax.set_title(title, pad=30)

    print('writing', output)
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def go(args):
    output = args.output
    if output is None:
        output = os.path.splitext(args.fname)[0] + '.png'

    plot_footprint(
        args.fname, output,
        ra_range=args.ra_range, dec_range=args.dec_range,
        nside=args.nside, xsize=args.xsize, dpi=args.dpi,
        title=args.title, cmap=args.cmap, min_abs_b=args.min_abs_b,
    )


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('fname', help='the footprint healsparse map')
    parser.add_argument('--output',
                        help='output image; default the map file '
                             'with a .png extension')
    parser.add_argument('--ra-range', type=float, nargs=2,
                        metavar=('LOW', 'HIGH'),
                        help='zoom to this ra range; values past 360 '
                             'express a range crossing ra = 0')
    parser.add_argument('--dec-range', type=float, nargs=2,
                        metavar=('LOW', 'HIGH'),
                        help='zoom to this dec range')
    parser.add_argument('--nside', type=int,
                        help='fracdet resolution; default matches '
                             'the raster resolution')
    parser.add_argument('--xsize', type=int, default=DEFAULT_XSIZE,
                        help='raster width in pixels')
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI)
    parser.add_argument('--title',
                        help='default is the map file name and area')
    parser.add_argument('--cmap',
                        help='matplotlib colormap name, e.g. inferno')
    parser.add_argument('--min-abs-b', type=float,
                        default=DEFAULT_MIN_ABS_B,
                        help='draw the galactic latitude cut lines '
                             'at |b| = this value; <= 0 draws none')

    args = parser.parse_args()
    if (args.ra_range is None) != (args.dec_range is None):
        parser.error('give both --ra-range and --dec-range, or neither')
    return args


def main():
    go(get_args())


if __name__ == '__main__':
    main()
