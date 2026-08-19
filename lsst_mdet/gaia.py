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
        except Exception as err:
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
