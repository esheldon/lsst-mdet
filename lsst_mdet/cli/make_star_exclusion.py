"""
make the star exclusion map: True inside a circle around each
input star.  For masked stars (G brighter than the
subtraction/masking limit) the radius is the mask radius law
plus the taper width plus a boundary (default 4 arcsec, DES
style): objects inside carry a tangential-shear bias from their
moment aperture overlapping the attenuated star region.  Fainter
(unmasked) stars get a fixed circle (default 6 arcsec) covering
the blending with the unsubtracted star light.  AND NOT this map
with a footprint, or look object positions up in it, to apply
the exclusion.

The star catalog needs ra, dec (degrees) and a G magnitude
(gmag, phot_g_mean_mag or G); stars brighter than --gmax are
used.

    lsst-mdet-make-star-exclusion \\
        --stars run-dp2-v00-corr-stars.fits \\
        --output run-dp2-v00-star-exclusion.hsp --nproc 128
"""


def read_star_gmag(stars):
    for name in ('gmag', 'phot_g_mean_mag', 'G'):
        if name in stars.dtype.names:
            return stars[name]
    raise ValueError(
        'no G magnitude column (gmag, phot_g_mean_mag or G) in '
        f'{stars.dtype.names}'
    )


def go(args):
    import numpy as np
    import rustfits
    from ..hmaps import (
        make_circle_exclusion_map, star_exclusion_radii,
    )

    print('reading:', args.stars)
    stars = rustfits.read(args.stars)
    gmag = read_star_gmag(stars)

    keep = gmag < args.gmax
    print(f'{keep.sum()} of {stars.size} stars with G < {args.gmax}')
    stars = stars[keep]
    gmag = gmag[keep]

    ra = np.asarray(stars['ra'], dtype='f8')
    dec = np.asarray(stars['dec'], dtype='f8')
    rad = star_exclusion_radii(
        np.asarray(gmag, dtype='f8'),
        boundary=args.boundary,
        faint_radius=args.faint_radius,
    )

    if args.extra_stars is not None:
        print('reading:', args.extra_stars)
        extra = rustfits.read(args.extra_stars)
        print(f'{extra.size} extra stars at '
              f'{args.extra_radius} arcsec')
        ra = np.concatenate([ra, extra['ra'].astype('f8')])
        dec = np.concatenate([dec, extra['dec'].astype('f8')])
        rad = np.concatenate([
            rad,
            np.full(extra.size, args.extra_radius / 3600.0),
        ])

    exmap = make_circle_exclusion_map(
        ra=ra, dec=dec, rad_deg=rad, nproc=args.nproc,
    )

    area = exmap.get_valid_area(degrees=True)
    print(f'excluded pixels: {exmap.n_valid}')
    print(f'excluded area: {area:.2f} deg^2')

    print('writing:', args.output)
    exmap.write(args.output, clobber=args.clobber)


def get_args():
    import argparse
    from ..hmaps import (
        EXCLUSION_BOUNDARY, EXCLUSION_FAINT_GMAX,
        EXCLUSION_FAINT_RADIUS,
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--stars', required=True,
                        help='star catalog with ra, dec and a G '
                             'magnitude')
    parser.add_argument('--output', required=True,
                        help='output healsparse map')
    parser.add_argument('--gmax', type=float,
                        default=EXCLUSION_FAINT_GMAX,
                        help='use stars brighter than this '
                             '(default %(default)s)')
    parser.add_argument('--boundary', type=float,
                        default=EXCLUSION_BOUNDARY,
                        help='masked stars: margin beyond the mask '
                             'circle plus taper, arcsec '
                             '(default %(default)s)')
    parser.add_argument('--faint-radius', type=float,
                        default=EXCLUSION_FAINT_RADIUS,
                        help='unmasked stars: fixed circle radius, '
                             'arcsec (default %(default)s)')
    parser.add_argument('--extra-stars',
                        help='additional star catalog (ra, dec), '
                             'e.g. survey-selected stars beyond '
                             'the gaia depth, excluded with fixed '
                             'circles')
    parser.add_argument('--extra-radius', type=float,
                        default=EXCLUSION_FAINT_RADIUS,
                        help='circle radius for --extra-stars, '
                             'arcsec (default %(default)s)')
    parser.add_argument('--nproc', type=int, default=8,
                        help='processes for the circle queries')
    parser.add_argument('--clobber', action='store_true',
                        help='overwrite an existing output')

    args = parser.parse_args()
    if args.nproc < 1:
        parser.error('--nproc must be >= 1')
    return args


def main():
    go(get_args())


if __name__ == '__main__':
    main()
