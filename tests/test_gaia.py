"""
the file gaia input must reproduce the fetch_gaia layout:
same patch circle and gmax cuts, neutral fills for the columns
the file does not carry, proper motions when it does
"""
import numpy as np
import pytest

from lsst_mdet.gaia import (
    GMAX, gaia_from_columns, gaia_pixel_positions,
    read_gaia_file,
)

from test_footprint import NPIX, X0, Y0, make_wcs_and_bbox


def make_columns():
    """
    stars around the synthetic patch: two inside the circle,
    one outside it, one inside but fainter than GMAX
    """
    wcs, bbox = make_wcs_and_bbox()
    ra0, dec0 = wcs.pixelToSkyArray(
        np.array([X0 + NPIX / 2], dtype='f8'),
        np.array([Y0 + NPIX / 2], dtype='f8'),
        degrees=True,
    )
    ra0, dec0 = float(ra0[0]), float(dec0[0])
    ra = np.array([ra0, ra0 + 0.05, ra0 + 0.5, ra0])
    dec = np.array([dec0, dec0, dec0, dec0 + 0.01])
    gmag = np.array([12.0, 18.0, 12.0, 25.0])
    return wcs, bbox, ra, dec, gmag, ra0, dec0


def test_gaia_from_columns():
    wcs, bbox, ra, dec, gmag, ra0, dec0 = make_columns()

    gaia = gaia_from_columns(
        ra=ra, dec=dec, gmag=gmag, wcs=wcs, bbox=bbox,
    )

    # the outside-circle star and the faint star are cut
    assert gaia.size == 2
    assert gaia['ra'] == pytest.approx([ra0, ra0 + 0.05])
    assert gaia['phot_g_mean_mag'] == pytest.approx([12.0, 18.0])

    # neutral fills for the missing catalog columns
    assert np.all(gaia['pmra'] == 0)
    assert np.all(gaia['pmdec'] == 0)
    assert np.all(gaia['ruwe'] == 1.0)

    # a higher gmax keeps the faint star
    gaia = gaia_from_columns(
        ra=ra, dec=dec, gmag=gmag, wcs=wcs, bbox=bbox, gmax=30.0,
    )
    assert gaia.size == 3


def test_gaia_from_columns_pm():
    # proper motions are carried through, and indexed with the
    # same cuts as the positions
    wcs, bbox, ra, dec, gmag, _, _ = make_columns()
    pmra = np.array([1.0, 2.0, 3.0, 4.0])
    pmdec = -pmra

    gaia = gaia_from_columns(
        ra=ra, dec=dec, gmag=gmag, wcs=wcs, bbox=bbox,
        pmra=pmra, pmdec=pmdec,
    )
    assert gaia['pmra'] == pytest.approx([1.0, 2.0])
    assert gaia['pmdec'] == pytest.approx([-1.0, -2.0])


def test_positions_roundtrip():
    # with zero proper motion the pixel positions come straight
    # from the wcs: the patch-center star lands at the center
    wcs, bbox, ra, dec, gmag, _, _ = make_columns()
    gaia = gaia_from_columns(
        ra=ra, dec=dec, gmag=gmag, wcs=wcs, bbox=bbox,
    )
    x, y = gaia_pixel_positions(gaia, wcs, bbox)
    assert x[0] == pytest.approx(NPIX / 2, abs=1e-6)
    assert y[0] == pytest.approx(NPIX / 2, abs=1e-6)


def test_read_gaia_parquet(tmp_path):
    pd = pytest.importorskip('pandas')

    wcs, bbox, ra, dec, gmag, _, _ = make_columns()
    fname = str(tmp_path / 'gaia.parquet')
    # extra columns, as in a general matched catalog, are
    # ignored; gaia_g_mag is the matched-catalog name for G
    df = pd.DataFrame({
        'ra': ra,
        'dec': dec,
        'gaia_g_mag': gmag,
        'other': np.arange(ra.size),
    })
    try:
        df.to_parquet(fname)
    except ImportError:
        pytest.skip('no parquet engine')

    gaia = read_gaia_file(fname, wcs, bbox)
    expected = gaia_from_columns(
        ra=ra, dec=dec, gmag=gmag, wcs=wcs, bbox=bbox,
    )
    assert np.array_equal(gaia, expected)
    assert gaia.dtype == expected.dtype

    # the gmax passthrough
    gaia = read_gaia_file(fname, wcs, bbox, gmax=30.0)
    assert gaia.size == 3


def test_read_gaia_fits(tmp_path):
    # the lsst-mdet-make-gaia layout: TAP column names and
    # proper motions, which move the pixel positions
    rustfits = pytest.importorskip('rustfits')

    wcs, bbox, ra, dec, gmag, _, _ = make_columns()
    fname = str(tmp_path / 'gaia.fits')

    pmra = np.array([100.0, 0.0, 0.0, 0.0])
    data = np.zeros(ra.size, dtype=[
        ('source_id', 'i8'), ('ra', 'f8'), ('dec', 'f8'),
        ('pmra', 'f8'), ('pmdec', 'f8'), ('phot_g_mean_mag', 'f8'),
    ])
    data['source_id'] = np.arange(ra.size)
    data['ra'] = ra
    data['dec'] = dec
    data['pmra'] = pmra
    data['phot_g_mean_mag'] = gmag
    rustfits.write(fname, data)

    gaia = read_gaia_file(fname, wcs, bbox)
    expected = gaia_from_columns(
        ra=ra, dec=dec, gmag=gmag, wcs=wcs, bbox=bbox,
        pmra=pmra, pmdec=np.zeros(ra.size),
    )
    assert np.array_equal(gaia, expected)
    assert gaia['pmra'][0] == 100.0

    # the center star moves off center by the proper motion:
    # 100 mas/yr over the 2016 to 2025 baseline is 0.9 arcsec,
    # 4.5 pixels at 0.2 arcsec/pixel, westward so -x (E left)
    x, y = gaia_pixel_positions(gaia, wcs, bbox)
    assert x[0] == pytest.approx(NPIX / 2 - 4.5, abs=0.01)
    assert y[0] == pytest.approx(NPIX / 2, abs=1e-3)


def test_default_gmax():
    # the file reader and the TAP query share the download
    # depth default
    import inspect
    sig = inspect.signature(read_gaia_file)
    assert sig.parameters['gmax'].default == GMAX
