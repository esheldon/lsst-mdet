"""
the healsparse keep-footprint for a patch: the union of the
processed cell polygons, with the star masks (out to the
taper) cleared and the whole thing trimmed to the tract inner
boundary
"""
import numpy as np

NSIDE_COVERAGE = 32
NSIDE = 131072


def make_empty_footprint():
    """
    Make an empty bool footprint which we will fill with
    our geometry
    """
    import healsparse
    return healsparse.HealSparseMap.make_empty(
        nside_coverage=NSIDE_COVERAGE,
        nside_sparse=NSIDE,
        dtype=bool,
        bit_packed=True,
    )


def mask_stars_in_footprint(footprint, wcs, bbox, star_table, apod):
    """
    Mask the stars in the footprint: clear the pixels of each
    census star's circle (the mask radius law plus the taper).

    OR-ing a value=False geometry cannot clear pixels, so the
    circle pixels are set False directly

    Parameters
    ----------
    footprint: healsparse map
        The footprint in which stars will be masked
    wcs: ButlerWcs or FileWcs
        Used for the pixel scale
    bbox: bbox
        The patch bounding box, tract frame
    star_table: array
        The gaia census with ra, dec, G
    apod: float
        The star taper width in pixels, added to the circle
        radii
    """
    import healsparse
    from .starsub import circle_radius

    scale = get_pixel_scale(
        wcs,
        0.5 * (bbox.x.start + bbox.x.stop),
        0.5 * (bbox.y.start + bbox.y.stop),
    )

    rad_deg = (circle_radius(star_table['G']) + apod) * scale / 3600

    for cra, cdec, crad in zip(
        star_table['ra'], star_table['dec'], rad_deg,
    ):
        circle = healsparse.Circle(
            ra=cra,
            dec=cdec,
            radius=crad,
            value=True,
        )
        footprint[circle.get_pixels(nside=NSIDE)] = False


def trim_footprint_to_tract_bounds(footprint, tract_bounds):
    """
    Clear footprint pixels whose centers fall outside the
    tract inner boundary, using the same test as the catalog
    is_primary cut.

    AND-ing a value=True geometry is a no-op (it never clears
    pixels outside the geometry), so the outside pixels are
    set False directly

    Parameters
    ----------
    footprint: healsparse map
        The footprint to trim
    tract_bounds: (ra_min, ra_max, dec_min, dec_max)
        The tract inner sky bounds in degrees
    """
    import hpgeom
    from .cells import get_tract_primary

    vp = footprint.valid_pixels
    if vp.size == 0:
        return
    ra, dec = hpgeom.pixel_to_angle(
        NSIDE, vp, nest=True, lonlat=True, degrees=True,
    )
    keep = get_tract_primary(tract_bounds, ra, dec)
    footprint[vp[~keep]] = False


def get_pixel_scale(wcs, x, y):
    """arcsec per pixel from the local linearization"""
    m = wcs.linearize_matrix(x, y)
    return float(np.sqrt(np.abs(np.linalg.det(m))))


def make_patch_polygon(ra, dec):
    """
    Make a polygon for the patch region

    Parameters
    ----------
    ra: array
        Array of ra for the corners
    dec: array
        Array of dec for the corners

    Returns
    -------
    healsparse.Polygon
    """
    import healsparse
    return healsparse.Polygon(ra=ra, dec=dec, value=True)
