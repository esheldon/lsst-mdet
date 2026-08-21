"""
Gaia DR3 download and coordinate handling
"""
import numpy as np


# Gaia positions are queried at the catalog epoch and
# propagated by proper motion to the approximate observation
# epoch
GAIA_EPOCH = 2016.0
OBS_EPOCH = 2025.0
GMAX = 19.0         # download depth

# TAP sync endpoints serving gaiadr3.gaia_source with the same
# query dialect and csv output; tried in order.  ESA is the
# canonical archive, ARI Heidelberg a full mirror (ESA has been
# observed to reset connections during outages)
GAIA_TAP_URLS = [
    'https://gea.esac.esa.int/tap-server/tap/sync',
    'https://gaia.ari.uni-heidelberg.de/tap/sync',
]

GAIA_ADQL = (
    'SELECT source_id, ra, dec, pmra, pmdec, parallax, '
    'phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe '
    'FROM gaiadr3.gaia_source '
    "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
    "CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {rad:.4f})) "
    'AND phot_g_mean_mag < {gmax}'
)


def fetch_gaia(wcs, bbox, gmax=GMAX):
    """
    Gaia DR3 extract for this patch from the ESA TAP sync
    service (a few seconds), as a numpy structured array:
    circle centered on the patch, corner radius plus margin
    for off-patch intruders
    """
    import io
    import urllib.request
    import urllib.parse

    xmid = 0.5 * (bbox.x.start + bbox.x.stop)
    ymid = 0.5 * (bbox.y.start + bbox.y.stop)
    ctr = wcs.pixelToSky(xmid, ymid)
    corner = wcs.pixelToSky(
        float(bbox.x.start), float(bbox.y.start),
    )
    rad = ctr.separation(corner).asDegrees() + 0.02

    query = GAIA_ADQL.format(
        ra=ctr.getRa().asDegrees(),
        dec=ctr.getDec().asDegrees(),
        rad=rad,
        gmax=gmax,
    )
    print(query)
    data = urllib.parse.urlencode({
        'REQUEST': 'doQuery',
        'LANG': 'ADQL',
        'FORMAT': 'csv',
        'QUERY': query,
    }).encode()

    text = None
    errors = []
    for url in GAIA_TAP_URLS:
        try:
            with urllib.request.urlopen(
                url, data=data, timeout=120,
            ) as resp:
                text = resp.read().decode()
            if not text.startswith('source_id'):
                raise RuntimeError(
                    'unexpected TAP response: ' + text[:200],
                )
            break
        except (OSError, RuntimeError, UnicodeDecodeError) as err:
            # OSError covers the whole network family (URLError,
            # HTTPError, connection resets, timeouts, ssl);
            # RuntimeError is our own unexpected-response check,
            # so an error page from one mirror falls through to
            # the next
            print(f'    gaia query failed at {url}: {err}')
            errors.append(f'{url}: {err}')
            text = None
    if text is None:
        raise RuntimeError(
            'all gaia TAP services failed:\n    '
            + '\n    '.join(errors)
        )
    gaia = np.genfromtxt(
        io.StringIO(text), delimiter=',', names=True,
    )
    print(f'    gaia: {gaia.size} stars')
    return gaia


def read_gaia_parquet(fname, wcs, bbox, gmax=GMAX):
    """
    gaia stars for this patch from a parquet file with columns
    gaia_g_mag, ra, dec (degrees), converted to the fetch_gaia
    layout so everything downstream is unchanged.  The same
    circle as the TAP query is applied, plus the gmax cut.

    Columns the file does not carry are filled neutrally: zero
    proper motion (positions are used as given) and ruwe = 1
    (the template astrometric-quality guard passes everything).

    Requires pandas with a parquet engine, present in the
    rubin/stackvana environments
    """
    import pandas as pd

    df = pd.read_parquet(fname, columns=['ra', 'dec', 'gaia_g_mag'])
    gaia = gaia_from_columns(
        ra=df['ra'].to_numpy(dtype='f8'),
        dec=df['dec'].to_numpy(dtype='f8'),
        gmag=df['gaia_g_mag'].to_numpy(dtype='f8'),
        wcs=wcs,
        bbox=bbox,
        gmax=gmax,
    )
    print(f'    gaia from {fname}: {gaia.size} stars')
    return gaia


def gaia_from_columns(ra, dec, gmag, wcs, bbox, gmax=GMAX):
    """
    build the fetch_gaia structured layout from plain position
    and magnitude arrays, applying the same patch circle as the
    TAP query and the gmax cut
    """
    xmid = 0.5 * (bbox.x.start + bbox.x.stop)
    ymid = 0.5 * (bbox.y.start + bbox.y.stop)
    ctr = wcs.pixelToSky(xmid, ymid)
    corner = wcs.pixelToSky(
        float(bbox.x.start), float(bbox.y.start),
    )
    rad = ctr.separation(corner).asDegrees() + 0.02

    ra0 = np.deg2rad(ctr.getRa().asDegrees())
    dec0 = np.deg2rad(ctr.getDec().asDegrees())
    rar = np.deg2rad(np.asarray(ra, dtype='f8'))
    decr = np.deg2rad(np.asarray(dec, dtype='f8'))
    cossep = (
        np.sin(dec0) * np.sin(decr)
        + np.cos(dec0) * np.cos(decr) * np.cos(rar - ra0)
    )
    sep = np.rad2deg(np.arccos(np.clip(cossep, -1, 1)))

    w, = np.where((sep <= rad) & (np.asarray(gmag) < gmax))
    gaia = np.zeros(w.size, dtype=[
        ('ra', 'f8'), ('dec', 'f8'),
        ('pmra', 'f8'), ('pmdec', 'f8'),
        ('phot_g_mean_mag', 'f8'), ('ruwe', 'f8'),
    ])
    gaia['ra'] = np.asarray(ra)[w]
    gaia['dec'] = np.asarray(dec)[w]
    gaia['phot_g_mean_mag'] = np.asarray(gmag)[w]
    gaia['ruwe'] = 1.0
    return gaia


def gaia_pixel_positions(gaia, wcs, bbox):
    """
    patch-frame pixel positions with proper motions propagated
    to the observation epoch
    """
    dt = OBS_EPOCH - GAIA_EPOCH
    pmra = np.nan_to_num(gaia['pmra'])
    pmdec = np.nan_to_num(gaia['pmdec'])
    cosd = np.cos(np.deg2rad(gaia['dec']))
    ra = gaia['ra'] + dt * pmra / 3.6e6 / cosd
    dec = gaia['dec'] + dt * pmdec / 3.6e6

    x, y = wcs.skyToPixelArray(ra, dec, degrees=True)
    return x - bbox.x.start, y - bbox.y.start


def fetch_gaia_or_none(wcs, bbox, gmax=GMAX, require=False):
    """
    fetch_gaia with failure handling: when require is set a
    failure raises (never proceed silently without stars when
    the caller asked for star handling); otherwise the error
    is printed and None returned so the caller can degrade
    gracefully
    """
    try:
        return fetch_gaia(wcs, bbox, gmax=gmax)
    except RuntimeError as err:
        # the only expected failure: all mirrors exhausted
        # (per-mirror errors are consumed inside fetch_gaia)
        if require:
            raise RuntimeError(
                'gaia download failed and star subtraction '
                'was requested'
            ) from err
        print('    gaia download failed:', err)
        return None
