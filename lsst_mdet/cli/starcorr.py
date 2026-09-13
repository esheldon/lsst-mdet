"""
tangential shear around stars: treecorr NG correlations with the
stars as lenses and the galaxies as sources, in bins of the star G
magnitude.  The inputs are the catalogs from
lsst-mdet-make-corr-cats; gamma_t and gamma_x are divided by the
global response stored in the galaxy catalog header.

The measurement is compensated with uniform random points drawn in
the same window as the star selection (the coarse fracdet of the
run footprint): gamma_t around the randoms is subtracted with the
treecorr rg mechanism.  Errors are jackknife over treecorr kmeans
sky patches, shared between the star, random and galaxy catalogs.
The raw and random signals are stored beside the compensated one.

Alongside, the number of shear sources around the stars: the
Landy-Szalay cross-correlation of the star and source positions,
with each source counted once (nsrc, the number of sources) and by
its shear weight (wsrc, the effective number the gamma_t measurement
uses).  The star randoms are the same uniform points; the source
randoms are those of them inside the footprint map itself (the
usable area, with the star masks, no-data and diffuse regions
excluded), so the result is the source density per usable area
around the stars relative to its mean, ideally flat at zero.

The galaxy shears are in the ngmix/galsim tangent plane convention
(u increasing to the west, v to the north).

    lsst-mdet-starcorr --run-dir . --nbins 20

Writes one table extension per magnitude bin, by default to
stats/starcorr.fits in the run directory, with a pdf plot beside
it.
"""
import os

import numpy as np

DEFAULT_GMAG_EDGES = [6.0, 16.0, 18.0, 20.0, 21.0]
DEFAULT_THETA_MIN = 0.05   # arcmin
DEFAULT_THETA_MAX = 30.0   # arcmin
DEFAULT_NBINS = 20
DEFAULT_NRAND = 20_000_000
DEFAULT_NPATCH = 100
DEFAULT_RAND_SEED = 3121
RAND_NSIDE = 4096


def get_output_file(run_dir):
    return os.path.join(run_dir, 'stats', 'starcorr.fits')


def make_randoms(footprint_file, nrand, seed):
    """
    uniform random points in the window of the star selection: the
    pixels of the coarse fracdet of the footprint with any coverage
    """
    import healsparse

    print(f'{nrand} randoms from', footprint_file)
    fp = healsparse.HealSparseMap.read(footprint_file)
    frac = fp.fracdet_map(RAND_NSIDE)
    window = healsparse.HealSparseMap.make_empty(
        min(32, RAND_NSIDE), RAND_NSIDE, dtype=bool,
    )
    vpix = frac.valid_pixels
    good = vpix[frac[vpix] > 0]
    window[good] = True

    rng = np.random.RandomState(seed)
    return healsparse.make_uniform_randoms(window, nrand, rng=rng)


def in_footprint(footprint_file, ra, dec):
    """
    Find which points fall inside a footprint map.

    Used to pick the source randoms of the count correlation: the
    randoms of the coarse window that lie in the usable area itself.

    Parameters
    ----------
    footprint_file: str
        The healsparse footprint map
    ra, dec: array
        The positions, degrees

    Returns
    -------
    inside: bool array
    """
    import healsparse

    fp = healsparse.HealSparseMap.read(footprint_file)
    return np.asarray(fp.get_values_pos(ra, dec, lonlat=True), dtype=bool)


def measure_starcorr(
    stars, gals, resp, gmag_edges, theta_min, theta_max, nbins,
    rand_ra, rand_dec, npatch, rand_in_footprint=None,
):
    """
    compensated NG correlations per star magnitude bin, with
    jackknife errors over shared kmeans sky patches.

    With rand_in_footprint, also the Landy-Szalay NN
    cross-correlation of the stars with the source positions, per
    bin: DD stars x sources, DR stars x source randoms, RD star
    randoms x sources, RR star randoms x source randoms.  The star
    randoms are all of rand_ra, rand_dec, the source randoms the
    subset inside the footprint (rand_in_footprint), so the result
    is the source density per usable area around the stars
    relative to its mean.  Each source counted once (nsrc, the
    number of shear sources) and by its shear weight (wsrc)

    Returns
    -------
    list of (lo, hi, data, cov, ncov): the magnitude bin edges, a
    table with theta (arcmin), the compensated gammat and gammax,
    the jackknife err, the raw and random gammat and npairs, plus
    with the counts nsrc_xi, nsrc_err, wsrc_xi, wsrc_err; the
    jackknife covariance of gammat, and a dict of the count
    covariances by nsrc/wsrc (empty without the counts)
    """
    import treecorr

    corr_config = dict(
        min_sep=theta_min, max_sep=theta_max, nbins=nbins,
        sep_units='arcmin', var_method='jackknife',
        cross_patch_weight='match',
    )

    scat = treecorr.Catalog(
        ra=gals['ra'], dec=gals['dec'],
        g1=gals['g1'], g2=gals['g2'], w=gals['w'],
        ra_units='deg', dec_units='deg', npatch=npatch,
    )

    print('processing the randoms')
    rcat = treecorr.Catalog(
        ra=rand_ra, dec=rand_dec,
        ra_units='deg', dec_units='deg',
        patch_centers=scat.patch_centers,
    )
    rg = treecorr.NGCorrelation(**corr_config)
    rg.process(rcat, scat)

    # the source counts: the terms without the stars are shared by
    # every magnitude bin
    counts = None
    if rand_in_footprint is not None:
        print('processing the source counts')
        srcs = {
            'nsrc': treecorr.Catalog(
                ra=gals['ra'], dec=gals['dec'],
                ra_units='deg', dec_units='deg',
                patch_centers=scat.patch_centers,
            ),
            'wsrc': treecorr.Catalog(
                ra=gals['ra'], dec=gals['dec'], w=gals['w'],
                ra_units='deg', dec_units='deg',
                patch_centers=scat.patch_centers,
            ),
        }
        srand = treecorr.Catalog(
            ra=rand_ra[rand_in_footprint], dec=rand_dec[rand_in_footprint],
            ra_units='deg', dec_units='deg',
            patch_centers=scat.patch_centers,
        )
        rr = treecorr.NNCorrelation(**corr_config)
        rr.process(rcat, srand)
        rd = {}
        for key, cat in srcs.items():
            rd[key] = treecorr.NNCorrelation(**corr_config)
            rd[key].process(rcat, cat)
        counts = dict(srcs=srcs, srand=srand, rr=rr, rd=rd)

    dtype = [
        ('theta', 'f8'), ('gammat', 'f8'), ('gammax', 'f8'),
        ('err', 'f8'), ('gammat_raw', 'f8'),
        ('gammat_rand', 'f8'), ('npairs', 'f8'),
    ]
    if counts is not None:
        dtype += [(f'{key}_{col}', 'f8') for key in ('nsrc', 'wsrc')
                  for col in ('xi', 'err')]

    results = []
    for lo, hi in zip(gmag_edges[:-1], gmag_edges[1:]):
        sel, = np.where(
            (stars['gmag'] >= lo) & (stars['gmag'] < hi)
        )
        print(f'G [{lo:g}, {hi:g}): {sel.size} stars')
        lcat = treecorr.Catalog(
            ra=stars['ra'][sel], dec=stars['dec'][sel],
            ra_units='deg', dec_units='deg',
            patch_centers=scat.patch_centers,
        )
        ng = treecorr.NGCorrelation(**corr_config)
        ng.process(lcat, scat)

        raw = ng.xi.copy()
        xi, xi_im, varxi = ng.calculateXi(rg=rg)
        # full jackknife covariance of xi (gamma_t before the
        # response division)
        cov = ng.cov.copy()

        data = np.zeros(nbins, dtype=dtype)
        data['theta'] = ng.meanr
        data['gammat'] = xi / resp
        data['gammax'] = xi_im / resp
        data['err'] = np.sqrt(varxi) / resp
        data['gammat_raw'] = raw / resp
        data['gammat_rand'] = rg.xi / resp
        data['npairs'] = ng.npairs

        ncov = {}
        if counts is not None:
            dr = treecorr.NNCorrelation(**corr_config)
            dr.process(lcat, counts['srand'])
            for key, cat in counts['srcs'].items():
                dd = treecorr.NNCorrelation(**corr_config)
                dd.process(lcat, cat)
                nxi, nvar = dd.calculateXi(
                    rr=counts['rr'], dr=dr, rd=counts['rd'][key],
                )
                data[f'{key}_xi'] = nxi
                data[f'{key}_err'] = np.sqrt(nvar)
                ncov[key] = dd.cov.copy()

        results.append((lo, hi, data, cov / resp ** 2, ncov))

    return results


def plot_starcorr(results, output, ymin=None, ymax=None):
    """
    gamma_t vs theta for every magnitude bin, on one axes, and when
    measured the source counts below it (number, then weight);
    ymin, ymax apply to gamma_t
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as mplt

    has_counts = all('nsrc_xi' in r[2].dtype.names for r in results)
    panels = [('gammat', 'err', r'$\gamma_t$')]
    if has_counts:
        panels += [
            ('nsrc_xi', 'nsrc_err', r'$\xi_{\rm src}$, number'),
            ('wsrc_xi', 'wsrc_err', r'$\xi_{\rm src}$, weight'),
        ]
    fig, axes = mplt.subplots(
        len(panels), 1, figsize=(5.5, 5.5 / 1.62 * len(panels)),
        sharex=True, squeeze=False,
    )
    axes = axes[:, 0]
    for ax, (col, ecol, ylabel) in zip(axes, panels):
        ax.set(ylabel=ylabel, xscale='log')
        ax.axhline(0, color='black', lw=1)
        for lo, hi, data, *_ in results:
            ax.errorbar(
                data['theta'], data[col], data[ecol],
                marker='o', markersize=3,
                label=f'$G \\in [{lo:g}, {hi:g})$',
            )
    if ymin is not None or ymax is not None:
        axes[0].set_ylim(ymin, ymax)
    axes[-1].set_xlabel(r'$\theta$ [arcmin]')
    axes[0].legend(fontsize=8)

    print('writing:', output)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    mplt.close(fig)


def read_starcorr(fname):
    """
    the results list back from an output file, as measure_starcorr
    returns it (ncov empty for files without the source counts)
    """
    import rustfits

    results = []
    with rustfits.FITS(fname) as fits:
        names = [hdu.header.get('EXTNAME', '') for hdu in fits]
        for name in names:
            if not name.startswith('gmag') or name.endswith('_cov'):
                continue
            h = fits[name].header
            cov = None
            if f'{name}_cov' in names:
                cov = fits[f'{name}_cov'].read()
            ncov = {
                key: fits[f'{name}_{key}_cov'].read()
                for key in ('nsrc', 'wsrc') if f'{name}_{key}_cov' in names
            }
            results.append(
                (h['GMAG_LO'], h['GMAG_HI'], fits[name].read(), cov, ncov)
            )
    return results


def go(args):
    import rustfits

    output = args.output or get_output_file(args.run_dir)

    if args.plot_only:
        if not os.path.exists(output):
            raise RuntimeError(f'no output to plot: {output}')
        results = read_starcorr(output)
        plot_starcorr(
            results, os.path.splitext(output)[0] + '.pdf',
            ymin=args.ymin, ymax=args.ymax,
        )
        return

    if os.path.exists(output) and not args.clobber:
        raise RuntimeError(f'{output} exists and clobber is False')

    gals_file = args.gals
    if gals_file is None:
        from .make_corr_cats import get_gals_file
        gals_file = get_gals_file(args.run_dir)
    stars_file = args.stars
    if stars_file is None:
        from .make_corr_cats import get_stars_file
        stars_file = get_stars_file(args.run_dir)

    print('reading:', gals_file)
    with rustfits.FITS(gals_file) as fits:
        gals = fits['gals'].read()
        resp = fits['gals'].header['R']
    print(f'{gals.size} galaxies, R = {resp:.5f}')

    print('reading:', stars_file)
    stars = rustfits.read(stars_file)
    print(f'{stars.size} stars')

    footprint_file = args.footprint
    if footprint_file is None:
        from .make_footprint import get_footprint_file
        footprint_file = get_footprint_file(args.run_dir)
    rand_ra, rand_dec = make_randoms(
        footprint_file, args.nrand, args.rand_seed,
    )
    rand_in_fp = in_footprint(footprint_file, rand_ra, rand_dec)
    print(f'{rand_in_fp.sum()} of {rand_in_fp.size} randoms inside the '
          'footprint: the source randoms of the counts')

    results = measure_starcorr(
        stars=stars, gals=gals, resp=resp,
        gmag_edges=args.gmag_edges,
        theta_min=args.theta_min, theta_max=args.theta_max,
        nbins=args.nbins,
        rand_ra=rand_ra, rand_dec=rand_dec, npatch=args.npatch,
        rand_in_footprint=rand_in_fp,
    )

    print('writing:', output)
    with rustfits.FITS(output, 'w+') as fits:
        for lo, hi, data, cov, ncov in results:
            fits.write_table(
                data, extname=f'gmag_{lo:g}_{hi:g}',
                header={'R': resp, 'GMAG_LO': lo, 'GMAG_HI': hi},
                compress=True,
            )
            fits.write_image(
                cov, extname=f'gmag_{lo:g}_{hi:g}_cov',
            )
            for key, ncv in ncov.items():
                fits.write_image(
                    ncv, extname=f'gmag_{lo:g}_{hi:g}_{key}_cov',
                )

    plot_starcorr(
        results, os.path.splitext(output)[0] + '.pdf',
        ymin=args.ymin, ymax=args.ymax,
    )


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--run-dir', default='.',
                        help='the run directory; default inputs '
                             'are its corr-gals/corr-stars files')
    parser.add_argument('--gals',
                        help='the galaxy correlation catalog')
    parser.add_argument('--stars',
                        help='the star catalog')
    parser.add_argument('--output',
                        help='default <run-dir>/<run>-starcorr.fits '
                             'with the png beside it')
    parser.add_argument('--gmag-edges', type=float, nargs='+',
                        default=DEFAULT_GMAG_EDGES,
                        help='star G magnitude bin edges')
    parser.add_argument('--theta-min', type=float,
                        default=DEFAULT_THETA_MIN,
                        help='minimum separation [arcmin]')
    parser.add_argument('--theta-max', type=float,
                        default=DEFAULT_THETA_MAX,
                        help='maximum separation [arcmin]')
    parser.add_argument('--nbins', type=int, default=DEFAULT_NBINS,
                        help='number of log separation bins')
    parser.add_argument('--footprint',
                        help='footprint map for the random window; '
                             'default the run footprint')
    parser.add_argument('--nrand', type=int, default=DEFAULT_NRAND,
                        help='number of random points')
    parser.add_argument('--rand-seed', type=int,
                        default=DEFAULT_RAND_SEED,
                        help='seed for the random points')
    parser.add_argument('--npatch', type=int, default=DEFAULT_NPATCH,
                        help='kmeans sky patches for the jackknife')
    parser.add_argument('--plot-only', action='store_true',
                        help='re-render the png from the existing '
                             'output instead of re-measuring')
    parser.add_argument('--ymin', type=float,
                        help='fixed lower y limit for the plot')
    parser.add_argument('--ymax', type=float,
                        help='fixed upper y limit for the plot')
    parser.add_argument('--clobber', action='store_true',
                        help='overwrite an existing output')

    args = parser.parse_args()
    if len(args.gmag_edges) < 2:
        parser.error('need at least two --gmag-edges')
    return args


def main():
    go(get_args())


if __name__ == '__main__':
    main()
