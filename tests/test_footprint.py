"""
the keep-footprint machinery: tract bounds and primary cut
(including RA wrap), cell polygons (including index order),
star-circle clearing, and the tract trim
"""
import numpy as np
import pytest

from lsst_mdet.cells import (
    get_cell_healsparse_polygon,
    get_cell_primary,
    get_tract_bounds,
    get_tract_primary,
)
from lsst_mdet.defaults import (
    CELL_OVERLAP_HIGH, CELL_OVERLAP_LOW, CELL_SIZE,
)
from lsst_mdet.hmaps import (
    NSIDE,
    NSIDE_COVERAGE,
    make_empty_footprint,
    mask_stars_in_footprint,
    trim_footprint_to_tract_bounds,
)
from lsst_mdet.patchfiles import FileWcs, SimpleBox
from lsst_mdet.starsub import (
    MASK_R15, MASK_RMAX, MINRAD, circle_radius,
)

SCALE = 0.2  # arcsec/pixel
X0, Y0 = 24000, 21000
NPIX = 22 * CELL_SIZE


def make_wcs_and_bbox(ra0=53.0, dec0=-28.0):
    """
    synthetic TAN patch header (N up, E left) through the real
    FileWcs, with the patch origin at tract pixel (X0, Y0)
    """
    hdr = {
        'CTYPE1': 'RA---TAN',
        'CTYPE2': 'DEC--TAN',
        'CRVAL1': ra0,
        'CRVAL2': dec0,
        'CD1_1': -SCALE / 3600,
        'CD1_2': 0.0,
        'CD2_1': 0.0,
        'CD2_2': SCALE / 3600,
        'RADESYS': 'ICRS',
        # patch-frame CRPIX: reference at the patch center
        'CRPIX1': NPIX / 2,
        'CRPIX2': NPIX / 2,
        'LTV1': -X0,
        'LTV2': -Y0,
    }
    wcs = FileWcs(hdr)
    bbox = SimpleBox(X0, X0 + NPIX, Y0, Y0 + NPIX)
    return wcs, bbox


def cell_center_sky(wcs, bbox, cell_i, cell_j):
    x = bbox.x.start + (cell_j + 0.5) * CELL_SIZE
    y = bbox.y.start + (cell_i + 0.5) * CELL_SIZE
    ra, dec = wcs.pixelToSkyArray(
        np.array([x], dtype='f8'), np.array([y], dtype='f8'),
        degrees=True,
    )
    return float(ra[0]), float(dec[0])


def test_tract_primary_basic():
    bounds = (357.75, 359.25, -10.41, -8.93)
    ra = np.array([358.0, 357.0, 359.5, 357.75, 359.25])
    dec = np.array([-9.5, -9.5, -9.5, -9.5, -9.5])
    assert get_tract_primary(bounds, ra, dec).tolist() == [
        True, False, False, True, True,
    ]
    # dec cuts
    ra = np.full(3, 358.0)
    dec = np.array([-10.5, -10.41, -8.9])
    assert get_tract_primary(bounds, ra, dec).tolist() == [
        False, True, False,
    ]


def test_tract_primary_wrap():
    # ra range through 360: 359 -> 1
    bounds = (359.0, 1.0, -10.0, -9.0)
    ra = np.array([359.5, 0.5, 2.0, 358.5, 359.0, 1.0])
    dec = np.full(ra.size, -9.5)
    assert get_tract_primary(bounds, ra, dec).tolist() == [
        True, True, False, False, True, True,
    ]


def test_get_tract_bounds():
    class Ang:
        def __init__(self, d):
            self.d = d

        def asDegrees(self):
            return self.d

    class Interval:
        def __init__(self, a, b):
            self.a = Ang(a)
            self.b = Ang(b)

        def getA(self):
            return self.a

        def getB(self):
            return self.b

    class Box:
        def getLon(self):
            return Interval(357.75, 359.25)

        def getLat(self):
            return Interval(-10.41, -8.93)

    class TractInfo:
        inner_sky_region = Box()

    assert get_tract_bounds(TractInfo()) == (
        357.75, 359.25, -10.41, -8.93,
    )


def test_make_empty_footprint():
    fp = make_empty_footprint()
    assert fp.nside_sparse == NSIDE
    assert fp.nside_coverage == NSIDE_COVERAGE
    assert fp.dtype == np.bool_
    assert fp.valid_pixels.size == 0


def test_cell_polygon_covers_cell():
    wcs, bbox = make_wcs_and_bbox()
    fp = make_empty_footprint()
    fp |= get_cell_healsparse_polygon(
        bbox=bbox, cell_i=3, cell_j=7, wcs=wcs,
    )
    assert fp.valid_pixels.size > 0

    ra, dec = cell_center_sky(wcs, bbox, 3, 7)
    assert fp.get_values_pos(ra, dec, lonlat=True)

    # neighbors are not covered
    for ci, cj in [(2, 7), (4, 7), (3, 6), (3, 8)]:
        ra, dec = cell_center_sky(wcs, bbox, ci, cj)
        assert not fp.get_values_pos(ra, dec, lonlat=True)


def test_cell_polygon_index_order():
    # cell_i is along y, cell_j along x: the polygon of
    # (2, 5) must not cover the center of (5, 2)
    wcs, bbox = make_wcs_and_bbox()
    fp = make_empty_footprint()
    fp |= get_cell_healsparse_polygon(
        bbox=bbox, cell_i=2, cell_j=5, wcs=wcs,
    )
    ra, dec = cell_center_sky(wcs, bbox, 2, 5)
    assert fp.get_values_pos(ra, dec, lonlat=True)
    ra, dec = cell_center_sky(wcs, bbox, 5, 2)
    assert not fp.get_values_pos(ra, dec, lonlat=True)


def test_mask_stars():
    wcs, bbox = make_wcs_and_bbox()
    fp = make_empty_footprint()
    fp |= get_cell_healsparse_polygon(
        bbox=bbox, cell_i=10, cell_j=10, wcs=wcs,
    )

    sra, sdec = cell_center_sky(wcs, bbox, 10, 10)
    gmag = 16.0
    apod = 12.0
    star_table = np.zeros(1, dtype=[
        ('ra', 'f8'), ('dec', 'f8'), ('G', 'f4'),
    ])
    star_table['ra'] = sra
    star_table['dec'] = sdec
    star_table['G'] = gmag

    mask_stars_in_footprint(
        footprint=fp, wcs=wcs, bbox=bbox,
        star_table=star_table, apod=apod,
    )

    rad_deg = (circle_radius(gmag) + apod) * SCALE / 3600

    # center and points inside the circle are cleared
    assert not fp.get_values_pos(sra, sdec, lonlat=True)
    assert not fp.get_values_pos(
        sra, sdec + 0.5 * rad_deg, lonlat=True,
    )
    # points beyond the circle (still inside the cell) survive
    assert fp.get_values_pos(sra, sdec + 1.5 * rad_deg, lonlat=True)
    assert fp.get_values_pos(sra, sdec - 1.5 * rad_deg, lonlat=True)


def test_trim_footprint():
    wcs, bbox = make_wcs_and_bbox()
    fp = make_empty_footprint()
    for ci in [5, 6]:
        for cj in [5, 6]:
            fp |= get_cell_healsparse_polygon(
                bbox=bbox, cell_i=ci, cell_j=cj, wcs=wcs,
            )
    n0 = fp.valid_pixels.size

    # dec cut through the middle of the block: the boundary
    # between cell rows 5 and 6
    _, dec_cut = cell_center_sky(wcs, bbox, 6, 5)
    dec_cut -= 0.5 * CELL_SIZE * SCALE / 3600
    bounds = (0.0, 360.0, dec_cut, 90.0)
    trim_footprint_to_tract_bounds(fp, bounds)

    # half survives, up to healpix quantization along the cut
    # (1.6 arcsec pixels on a 60 arcsec block)
    n1 = fp.valid_pixels.size
    assert n1 == pytest.approx(n0 / 2, rel=0.05)

    # centers below the cut cleared, above kept
    for cj in [5, 6]:
        ra, dec = cell_center_sky(wcs, bbox, 5, cj)
        assert not fp.get_values_pos(ra, dec, lonlat=True)
        ra, dec = cell_center_sky(wcs, bbox, 6, cj)
        assert fp.get_values_pos(ra, dec, lonlat=True)


def test_trim_empty_footprint():
    fp = make_empty_footprint()
    trim_footprint_to_tract_bounds(fp, (0.0, 1.0, 0.0, 1.0))
    assert fp.valid_pixels.size == 0


def test_circle_radius():
    assert circle_radius(15.0) == pytest.approx(MASK_R15)
    # floor for faint stars, cap for bright
    assert circle_radius(25.0) == MINRAD
    assert circle_radius(0.0) == MASK_RMAX
    r = circle_radius(np.array([15.0, 25.0, 0.0]))
    assert r == pytest.approx([MASK_R15, MINRAD, MASK_RMAX])


def test_get_cell_primary():
    mid = 0.5 * (CELL_OVERLAP_LOW + CELL_OVERLAP_HIGH)
    assert get_cell_primary(mid, mid)
    assert not get_cell_primary(CELL_OVERLAP_LOW - 10.0, mid)
    assert not get_cell_primary(CELL_OVERLAP_HIGH + 10.0, mid)
    assert not get_cell_primary(mid, CELL_OVERLAP_LOW - 10.0)
    assert not get_cell_primary(mid, CELL_OVERLAP_HIGH + 10.0)
    x = np.array([mid, 0.0, 260.0])
    y = np.array([mid, mid, mid])
    assert get_cell_primary(x, y).tolist() == [True, False, False]
