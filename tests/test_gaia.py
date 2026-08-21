"""
the parquet gaia input must reproduce the fetch_gaia layout:
same patch circle and gmax cuts, neutral fills for the columns
the file does not carry
"""
import numpy as np
import pytest

from lsst_mdet.gaia import (
    GMAX, gaia_from_columns, gaia_pixel_positions,
    read_gaia_parquet,
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
    # ignored
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

    gaia = read_gaia_parquet(fname, wcs, bbox)
    expected = gaia_from_columns(
        ra=ra, dec=dec, gmag=gmag, wcs=wcs, bbox=bbox,
    )
    assert np.array_equal(gaia, expected)
    assert gaia.dtype == expected.dtype

    # the gmax passthrough
    gaia = read_gaia_parquet(fname, wcs, bbox, gmax=30.0)
    assert gaia.size == 3


def test_default_gmax():
    # the parquet reader and the TAP query share the download
    # depth default
    import inspect
    sig = inspect.signature(read_gaia_parquet)
    assert sig.parameters['gmax'].default == GMAX
