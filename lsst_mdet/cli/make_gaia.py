"""
cli/make_gaia

per-tract Gaia DR3 star files from the DM reference catalog in the
butler (gaia_dr3_20230707: full sky, sharded on HTM level 7).  These
are for --gaia-pattern in lsst-mdet-process-cells and the node driver,
replacing the per-patch TAP queries with no network access at all.

One FITS file per tract, gaia-dr3-{tract:05d}.fits, holding the stars
within the tract bounding circle plus a margin, to G < --gmax.  The
default depth is deeper than the processing depth (gaia.GMAX) so that
can change without regenerating the files.

The columns follow the TAP query layout used by gaia.fetch_gaia, in
degrees, mas/yr and mas

    source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag,
    phot_bp_mean_mag, phot_rp_mean_mag, astrometric_excess_noise,
    ref_epoch

with pmra including the cos(dec) factor as in Gaia, and ref_epoch the
catalog epoch as a decimal year (2016.0 for DR3)
"""
import os

import numpy as np

from ..defaults import BUTLER_REPO, SKYMAP_VERS

REFCAT = 'gaia_dr3_20230707'
REFCAT_COLLECTION = 'refcats/dm-39298/gaia_dr3_20230707'
HTM_LEVEL = 7

# stored depth; the processing depth is gaia.GMAX
DEFAULT_GMAX = 21.0

# beyond the tract bounding circle, to cover the per-patch circles
# (patch corner radius + 0.02 degrees) at the tract edge
MARGIN_DEG = 0.05

# the DM refcat stores AB fluxes in nJy, converted from the Gaia
# instrumental fluxes with the DR2 AB zero points (table 5.3 of the
# DR2 documentation; see meas_algorithms convertRefcatManager).  The
# Gaia magnitudes themselves use the EDR3/DR3 Vega zero points
# (Riello et al. 2021), so mag_gaia = mag_AB - (ZP_AB - ZP_vega).
# Subtracted so the files carry the same magnitudes as the TAP query
# and the GMAX/GSUB thresholds keep their meaning; verified against
# TAP for 980 stars in tract 7034 patch 86
AB_NJY_ZEROPOINT = 31.4
DM_AB_ZEROPOINTS = {'g': 25.7934, 'bp': 25.3806, 'rp': 25.1161}
GAIA_VEGA_ZEROPOINTS = {
    'g': 25.6873668671, 'bp': 25.3385422158, 'rp': 24.7478955012,
}
AB_MINUS_GAIA = {
    band: DM_AB_ZEROPOINTS[band] - GAIA_VEGA_ZEROPOINTS[band]
    for band in DM_AB_ZEROPOINTS
}
MJD_J2000 = 51544.5

# where the files go by default, and the matching --gaia-pattern
GAIA_DIR = os.path.join(os.environ.get('SCRATCH', '.'), 'gaia-dr3')
GAIA_FILE_PATTERN = 'gaia-dr3-{tract:05d}.fits'
GAIA_PATTERN = os.path.join(GAIA_DIR, GAIA_FILE_PATTERN)

OUTPUT_DTYPE = [
    ('source_id', 'i8'),
    ('ra', 'f8'),
    ('dec', 'f8'),
    ('pmra', 'f8'),
    ('pmdec', 'f8'),
    ('parallax', 'f8'),
    ('phot_g_mean_mag', 'f8'),
    ('phot_bp_mean_mag', 'f8'),
    ('phot_rp_mean_mag', 'f8'),
    ('astrometric_excess_noise', 'f8'),
    ('ref_epoch', 'f8'),
]


def get_gaia_file(tract, outdir=GAIA_DIR):
    return os.path.join(outdir, GAIA_FILE_PATTERN.format(tract=tract))


def get_tract_circle(skymap, tract):
    """
    the bounding circle of the tract outer polygon, grown by the margin

    Returns
    -------
    circle: lsst.sphgeom.Circle
    """
    import lsst.sphgeom as sphgeom

    tract_info = skymap[tract]
    circle = tract_info.getOuterSkyPolygon().getBoundingCircle()
    radius = circle.getOpeningAngle().asDegrees() + MARGIN_DEG
    return sphgeom.Circle(
        circle.getCenter(), sphgeom.Angle.fromDegrees(radius),
    )


def get_shard_ids(circle):
    """
    the HTM shard ids overlapping the circle
    """
    import lsst.sphgeom as sphgeom

    ranges = sphgeom.HtmPixelization(HTM_LEVEL).envelope(circle)
    return [i for lo, hi in ranges for i in range(lo, hi)]


def flux_to_mag(flux, band):
    """
    nJy AB flux to the Gaia magnitude in the band; non-positive or
    missing flux becomes nan
    """
    flux = np.asarray(flux, dtype='f8')
    mag = np.full(flux.shape, np.nan)
    good = np.isfinite(flux) & (flux > 0)
    mag[good] = (
        -2.5 * np.log10(flux[good]) + AB_NJY_ZEROPOINT
        - AB_MINUS_GAIA[band]
    )
    return mag


def convert_shard(cat):
    """
    convert a refcat shard (SimpleCatalog) to the output layout
    """
    rad2deg = np.rad2deg(1.0)
    # rad to mas, for the proper motions (per year) and parallax
    rad2mas = rad2deg * 3600 * 1000

    out = np.zeros(len(cat), dtype=OUTPUT_DTYPE)
    out['source_id'] = cat['id']
    out['ra'] = cat['coord_ra'] * rad2deg
    out['dec'] = cat['coord_dec'] * rad2deg
    out['pmra'] = cat['pm_ra'] * rad2mas
    out['pmdec'] = cat['pm_dec'] * rad2mas
    out['parallax'] = cat['parallax'] * rad2mas
    out['phot_g_mean_mag'] = flux_to_mag(cat['phot_g_mean_flux'], 'g')
    out['phot_bp_mean_mag'] = flux_to_mag(cat['phot_bp_mean_flux'], 'bp')
    out['phot_rp_mean_mag'] = flux_to_mag(cat['phot_rp_mean_flux'], 'rp')
    out['astrometric_excess_noise'] = cat['astrometric_excess_noise']
    # MJD to decimal year; 2016.0 for DR3
    out['ref_epoch'] = np.round(
        2000.0 + (cat['epoch'] - MJD_J2000) / 365.25, 4,
    )

    return out


def in_circle(circle, ra, dec):
    """
    which of the positions (degrees) fall in the circle
    """
    import lsst.sphgeom as sphgeom

    return np.array([
        circle.contains(sphgeom.UnitVector3d(
            sphgeom.LonLat.fromDegrees(r, d)
        ))
        for r, d in zip(ra, dec)
    ], dtype=bool)


def make_tract_file(butler, skymap, tract, outfile, gmax):
    """
    read the shards overlapping the tract circle and write the stars
    in the circle brighter than gmax to outfile
    """
    import rustfits

    circle = get_tract_circle(skymap, tract)
    shard_ids = get_shard_ids(circle)

    parts = []
    for shard_id in shard_ids:
        cat = butler.get(REFCAT, htm7=shard_id)
        parts.append(convert_shard(cat))

    stars = np.concatenate(parts)

    keep = (
        in_circle(circle, stars['ra'], stars['dec'])
        & np.isfinite(stars['phot_g_mean_mag'])
        & (stars['phot_g_mean_mag'] < gmax)
    )
    stars = stars[keep]
    stars.sort(order='source_id')

    print(f'    tract {tract}: {len(shard_ids)} shards, '
          f'{stars.size} stars to G < {gmax:g}, writing {outfile}')

    # write then rename, so a partial file is never mistaken for a
    # finished one by --skip-existing
    tmpfile = outfile + '.tmp'
    rustfits.write(tmpfile, stars, extname='gaia')
    os.replace(tmpfile, outfile)


def get_tracts(args):
    """
    the tracts from --tracts or the unique tracts in the good cells
    file
    """
    import rustfits

    if args.tracts is not None:
        return sorted(set(args.tracts))

    with rustfits.FITS(args.good_cells) as fits:
        good_cells = fits[1].read(columns=['tract'])

    tracts = np.unique(good_cells['tract'])
    print(f'{tracts.size} tracts in {args.good_cells}')
    return tracts.tolist()


def go(args):
    from lsst.daf.butler import Butler

    tracts = get_tracts(args)

    os.makedirs(args.outdir, exist_ok=True)

    butler = Butler(args.repo, collections=[args.refcat_collection])
    skymap = butler.get(
        'skyMap', skymap=SKYMAP_VERS, collections=args.collections,
    )

    nskip = 0
    for i, tract in enumerate(tracts):
        outfile = get_gaia_file(tract, outdir=args.outdir)
        if args.skip_existing and os.path.exists(outfile):
            nskip += 1
            continue

        print(f'{i + 1}/{len(tracts)}')
        make_tract_file(
            butler=butler, skymap=skymap, tract=tract, outfile=outfile,
            gmax=args.gmax,
        )

    if nskip > 0:
        print(f'skipped {nskip} existing files')


def get_args():
    import argparse
    from ..defaults import BUTLER_COLLECTIONS

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    which = parser.add_mutually_exclusive_group()
    which.add_argument('--good-cells', default='good-cells.fits',
                       help='make files for the tracts in this file')
    which.add_argument('--tracts', type=int, nargs='+',
                       help='make files for these tracts')

    parser.add_argument('--outdir', default=GAIA_DIR,
                        help='output directory; the files are '
                             f'{GAIA_FILE_PATTERN}')
    parser.add_argument('--gmax', type=float, default=DEFAULT_GMAX,
                        help='store stars brighter than this in G')
    parser.add_argument('--skip-existing', action='store_true',
                        help='skip tracts whose file exists')
    parser.add_argument('--repo', default=BUTLER_REPO)
    parser.add_argument('--collections', nargs='+',
                        default=BUTLER_COLLECTIONS,
                        help='collections for the skymap')
    parser.add_argument('--refcat-collection', default=REFCAT_COLLECTION,
                        help=f'collection holding {REFCAT}')

    return parser.parse_args()


def main():
    args = get_args()
    go(args)


if __name__ == '__main__':
    main()
