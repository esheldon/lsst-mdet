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


# the reference galactic latitudes drawn on the maps: the survey
# cut at |b| = 20 (dashed) and |b| = 30 (dotted)
GAL_B_LINES = (DEFAULT_MIN_ABS_B, 30.0)

# the deep fields with processed area, for map labels; the same
# centers define the wide/deep split of the null tests
DEEP_FIELDS = {
    'COSMOS': (150.1, 2.2),
    'ELAIS-S1': (9.45, -44.0),
    'ECDFS': (53.1, -28.1),
    'EDF-S': (61.0, -48.4),
}


def draw_gal_b_lines(sp, lon_0, b_values=GAL_B_LINES):
    """
    draw galactic latitude reference curves at +/- each |b| in
    b_values; the first dashed (the survey cut), later ones dotted
    """
    from astropy.coordinates import SkyCoord

    wrap = (lon_0 + 180.0) % 360
    styles = ('dashed', 'dotted', 'dashdot')
    # the x limits are inverted in the astronomy convention
    x0, x1 = sorted(sp.ax.get_xlim())
    y0, y1 = sorted(sp.ax.get_ylim())
    for k, abs_b in enumerate(np.atleast_1d(b_values)):
        style = styles[k % len(styles)]
        for b in (abs_b, -abs_b):
            gl = np.linspace(0.0, 360.0, 721)
            crd = SkyCoord(
                l=gl, b=np.full(gl.size, b), frame='galactic',
                unit='deg',
            ).icrs
            ra = crd.ra.deg
            dec = crd.dec.deg
            # keep the curve in its natural galactic-longitude
            # order (a small circle is double valued in ra) and
            # break the path where it crosses the projection wrap
            rw = (ra - wrap) % 360
            cut = np.where(np.abs(np.diff(rw)) > 180)[0] + 1
            sp.ax.plot(
                np.insert(ra, cut, np.nan),
                np.insert(dec, cut, np.nan),
                color='gray', linestyle=style, linewidth=1,
            )

            # label the curve directly, near each visible end
            px, py = sp.proj(ra, dec)
            good = (
                np.isfinite(px) & (px > x0) & (px < x1)
                & (py > y0) & (py < y1)
            )
            if not np.any(good):
                continue
            gidx = np.where(good)[0]
            gx = px[gidx]
            span = gx.max() - gx.min()
            used = []
            # stagger the label positions of successive |b|
            # values so they do not collide where the curves
            # converge
            for frac in (0.05 + 0.08 * k, 0.95 - 0.08 * k):
                j = gidx[np.argmin(np.abs(gx - (gx.min()
                                                + frac * span)))]
                if j in used:
                    continue
                used.append(j)
                sp.ax.text(
                    ra[j], dec[j], f'$b = {b:+g}^\\circ$',
                    fontsize=8, color='gray', ha='center',
                    va='bottom', clip_on=True,
                    bbox=dict(facecolor='white', alpha=0.7,
                              edgecolor='none', pad=0.5),
                )


def draw_deep_field_labels(sp, fontsize=9):
    """
    label the deep fields, the text placed beside each field; the
    white text box keeps the labels readable on any color map,
    and labels outside the drawn region are clipped
    """
    for name, (ra, dec) in DEEP_FIELDS.items():
        # anchor west of the field so the text extends away from
        # it (ra decreases to the right in the astro convention)
        sp.ax.text(
            ra - 2.6, dec, name, fontsize=fontsize,
            ha='left', va='center', color='black', clip_on=True,
            bbox=dict(facecolor='white', alpha=0.7,
                      edgecolor='none', pad=1),
        )


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

    if cmap is None:
        cmap = 'inferno'

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
        draw_gal_b_lines(sp, lon_0, (min_abs_b, 30.0))
    draw_deep_field_labels(sp)

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
