"""
the keep-footprint machinery: tract bounds and primary cut
(including RA wrap), cell polygons (including index order),
star-circle and image-mask clearing, the coadd star mask's
no-data extension, and the tract trim
"""
import os

import numpy as np
import pytest

from lsst_mdet.cells import (
    get_cell_healsparse_polygon,
    get_cell_primary,
    get_tract_bounds,
    get_tract_primary,
)
from lsst_mdet.defaults import (
    CELL_OVERLAP_HIGH, CELL_OVERLAP_LOW, CELL_SIZE, DM_NO_DATA, DM_SAT,
)
from lsst_mdet.hmaps import (
    NSIDE,
    NSIDE_COVERAGE,
    cat_footprints,
    get_footprint_flist,
    make_empty_footprint,
    mask_pixels_in_footprint,
    mask_stars_in_footprint,
    trim_footprint_to_tract_bounds,
)
from lsst_mdet.patchfiles import FileWcs, SimpleBox
from lsst_mdet.starsub import (
    MASK_R15, MASK_RMAX, MINRAD, _get_select_stars_dtype,
    build_star_mask, circle_radius,
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


def test_mask_pixels():
    import hpgeom

    wcs, bbox = make_wcs_and_bbox()
    fp = make_empty_footprint()
    fp |= get_cell_healsparse_polygon(
        bbox=bbox, cell_i=10, cell_j=10, wcs=wcs,
    )
    n0 = fp.valid_pixels.size

    # an empty mask changes nothing
    mask_pixels_in_footprint(
        footprint=fp, wcs=wcs, bbox=bbox,
        pixmask=np.zeros((NPIX, NPIX), dtype=bool),
    )
    assert fp.valid_pixels.size == n0

    # a 40 px (8 arcsec) block at the cell center
    c0 = int(10.5 * CELL_SIZE)
    pixmask = np.zeros((NPIX, NPIX), dtype=bool)
    pixmask[c0 - 20:c0 + 20, c0 - 20:c0 + 20] = True
    mask_pixels_in_footprint(
        footprint=fp, wcs=wcs, bbox=bbox, pixmask=pixmask,
    )

    ra, dec = cell_center_sky(wcs, bbox, 10, 10)
    assert not fp.get_values_pos(ra, dec, lonlat=True)

    # the cleared area is the block, with at most one healpix
    # pixel of margin on each side
    side = np.sqrt(hpgeom.nside_to_pixel_area(NSIDE, degrees=True)) * 3600
    block = 40 * SCALE
    removed = n0 - fp.valid_pixels.size
    assert (block / side) ** 2 <= removed <= ((block + 2 * side) / side) ** 2

    # the rest of the cell survives
    x = np.array([bbox.x.start + c0 + 50], dtype='f8')
    y = np.array([bbox.y.start + c0], dtype='f8')
    ra, dec = wcs.pixelToSkyArray(x, y, degrees=True)
    assert fp.get_values_pos(ra[0], dec[0], lonlat=True)


def test_star_mask_coadd_nodata():
    shape = (400, 400)
    stars = np.zeros(1, dtype=_get_select_stars_dtype())
    stars['x'] = 200.0
    stars['y'] = 200.0
    stars['G'] = 16.0
    stars['on_image'] = 1
    rad = circle_radius(16.0)

    mask0 = np.zeros(shape, dtype='u1')
    # a no-data strip starting inside the circle, running outward
    mask0[198:203, 220:320] |= DM_NO_DATA
    # a detached no-data blob
    mask0[50:60, 350:360] |= DM_NO_DATA
    # a saturated-only line through the star, past the circle
    mask0[100:300, 199:202] |= DM_SAT

    # coadd: the touching no-data joins, the detached blob and the
    # saturated line beyond the circle do not
    sm, _ = build_star_mask(stars, mask0, verbose=False, coadd=True)
    assert sm[200, 200]
    assert sm[200, 310]
    assert not sm[55, 355]
    assert not sm[int(200 - rad - 20), 200]

    # single exposure: the saturated line is the star's own
    # component, the no-data-only strip is not flagged
    sm, _ = build_star_mask(stars, mask0, verbose=False, coadd=False)
    assert sm[int(200 - rad - 20), 200]
    assert not sm[200, 310]


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


def make_block_footprint(blocks):
    """
    a footprint with pixel ranges [lo, hi) set inside the given
    coverage pixels: {cov_pixel: (lo, hi)}
    """
    nfine = (NSIDE // NSIDE_COVERAGE) ** 2
    fp = make_empty_footprint()
    for cov_pixel, (lo, hi) in blocks.items():
        fp[np.arange(cov_pixel * nfine + lo, cov_pixel * nfine + hi)] = True
    return fp


@pytest.mark.parametrize('nproc', [1, 2])
def test_cat_footprints(tmp_path, nproc):
    """
    the streamed union equals the in-memory union: three per-patch
    maps, two sharing a coverage pixel with overlapping pixels, one
    far away, with a file order that is not coverage pixel order
    """
    import healsparse

    maps = {
        # tract 00002 sorts after 00001 but holds the low pixels
        '00002/00002-00010-mdet-footprint.hsp': make_block_footprint(
            {100: (0, 5000), 101: (1000, 3000)},
        ),
        '00002/00002-00011-mdet-footprint.hsp': make_block_footprint(
            {101: (2000, 4000), 102: (0, 16)},
        ),
        '00001/00001-00003-mdet-footprint.hsp': make_block_footprint(
            {5000: (7, 8000)},
        ),
    }
    for relname, fp in maps.items():
        fname = tmp_path / relname
        fname.parent.mkdir(exist_ok=True)
        fp.write(str(fname), clobber=True)

    flist = get_footprint_flist(str(tmp_path))
    assert len(flist) == 3
    assert os.path.basename(flist[0]).startswith('00001')

    expected = np.unique(np.concatenate(
        [fp.valid_pixels for fp in maps.values()],
    ))
    n_inputs = sum(fp.valid_pixels.size for fp in maps.values())
    assert n_inputs - expected.size == 1000  # the overlap in 101

    outfile = str(tmp_path / 'footprint.hsp')
    res = cat_footprints(flist, outfile, nproc=nproc)
    assert res['nfile'] == 3
    assert res['ncov'] == 4
    assert res['n_valid'] == expected.size
    assert res['n_valid_inputs'] == n_inputs
    assert not os.path.exists(outfile + '.incomplete')

    total = healsparse.HealSparseMap.read(outfile)
    assert total.is_bit_packed_map
    assert total.nside_sparse == NSIDE
    assert total.nside_coverage == NSIDE_COVERAGE
    assert np.where(total.coverage_mask)[0].tolist() == [100, 101, 102, 5000]
    assert np.array_equal(total.valid_pixels, expected)

    # partial reads of the reshaped output
    nfine = (NSIDE // NSIDE_COVERAGE) ** 2
    part = healsparse.HealSparseMap.read(outfile, pixels=[101])
    in101 = expected[(expected // nfine) == 101]
    assert np.array_equal(part.valid_pixels, in101)

    with pytest.raises(RuntimeError):
        cat_footprints(flist, outfile, nproc=nproc)
    res = cat_footprints(flist, outfile, nproc=nproc, clobber=True)
    assert res['n_valid'] == expected.size


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
