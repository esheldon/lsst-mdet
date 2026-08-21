"""
healsparse mask maps for the patch masking: star mask circles
out to the taper plus the footprints of unprocessed cells
"""
import numpy as np

NSIDE_COVERAGE = 32
NSIDE = 131072


def make_mask_map(wcs, bbox, cell_info, star_table=None, apod=0.0):
    """
    boolean bit-packed healsparse map of the patch masking:
    True where area is lost to a star mask (out to the taper)
    or to a cell that was not processed

    Parameters
    ----------
    wcs: ButlerWcs or FileWcs
        Used for the cell footprints and the pixel scale
    bbox: bbox
        The patch bounding box, tract frame
    cell_info: array
        The per-cell processing info; cells with kept == False
        become Polygon footprints
    star_table: array, optional
        The gaia census (ra, dec, G); each star becomes a
        Circle from the mask radius law
    apod: float, optional
        The star taper width in pixels, added to the circle
        radii

    Returns
    -------
    healsparse.HealSparseMap
    """
    geom_list = []
    if star_table is not None and star_table.size > 0:
        geom_list += get_star_circles(
            wcs=wcs, bbox=bbox, star_table=star_table, apod=apod,
        )
    geom_list += get_missing_cell_polygons(
        wcs=wcs, bbox=bbox, cell_info=cell_info,
    )
    return geom_to_map(geom_list)


def get_star_circles(wcs, bbox, star_table, apod):
    """
    one Circle per census star: the mask radius law plus the
    taper width, converted to degrees with the pixel scale at
    the patch center
    """
    import healsparse
    from .starsub import circle_radius

    scale = get_pixel_scale(
        wcs,
        0.5 * (bbox.x.start + bbox.x.stop),
        0.5 * (bbox.y.start + bbox.y.stop),
    )
    rad_deg = (circle_radius(star_table['G']) + apod) * scale / 3600
    return [
        healsparse.Circle(ra=cra, dec=cdec, radius=crad, value=True)
        for cra, cdec, crad in zip(
            star_table['ra'], star_table['dec'], rad_deg,
        )
    ]


def get_missing_cell_polygons(wcs, bbox, cell_info):
    """
    the inner footprint of each cell with kept == False, as a
    Polygon
    """
    import healsparse
    from .defaults import CELL_SIZE

    polys = []
    w, = np.where(~cell_info['kept'])
    for k in w:
        x0 = bbox.x.start + cell_info['cell_j'][k] * CELL_SIZE
        y0 = bbox.y.start + cell_info['cell_i'][k] * CELL_SIZE
        xv = np.array(
            [x0, x0 + CELL_SIZE, x0 + CELL_SIZE, x0], dtype='f8',
        )
        yv = np.array(
            [y0, y0, y0 + CELL_SIZE, y0 + CELL_SIZE], dtype='f8',
        )
        ra, dec = wcs.pixelToSkyArray(xv, yv, degrees=True)
        polys.append(healsparse.Polygon(ra=ra, dec=dec, value=True))
    return polys


def get_pixel_scale(wcs, x, y):
    """arcsec per pixel from the local linearization"""
    m = wcs.linearize_matrix(x, y)
    return float(np.sqrt(np.abs(np.linalg.det(m))))


def geom_to_map(geom_list):
    """
    Create a healsparse map from the input list of geometry primitives

    Based on code from pizza_cutter

    Parameters
    ----------
    geom_list: [geom]
        E.g. a list of Circle or Polygon

    Returns
    -------
    hmap: healsparse.HealSparseMap
        Has the nside set in constants.py
    """
    import healsparse

    hmap = healsparse.HealSparseMap.make_empty(
        nside_coverage=NSIDE_COVERAGE,
        nside_sparse=NSIDE,
        dtype=bool,
        sentinel=False,
        bit_packed=True,
    )

    for g in geom_list:
        pixels = g.get_pixels(nside=NSIDE)
        hmap[pixels] = True

    return hmap
