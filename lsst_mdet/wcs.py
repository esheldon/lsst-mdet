"""
WCS helpers: headers, jacobians, sky positions
"""


class ButlerWcs(object):
    """
    thin wrapper around the DM tract wcs adding the
    linearize_matrix method the jacobian helper uses;
    everything else passes through.  The stack import lives
    inside the method so this module imports without the LSST
    pipelines (patchfiles.FileWcs is the file-backed
    counterpart)
    """

    def __init__(self, dm_wcs):
        self._wcs = dm_wcs

    def __getattr__(self, name):
        return getattr(self._wcs, name)

    def linearize_matrix(self, x, y):
        import lsst.geom

        dm_jac = self._wcs.linearizePixelToSky(
            lsst.geom.Point2D(x, y),
            lsst.geom.arcseconds,
        )
        return dm_jac.getLinear().getMatrix()


def get_cell_jacobian(wcs, bbox, x, y):
    import ngmix

    matrix = wcs.linearize_matrix(x, y)

    # ESS reverse engineered this convention mismatch.  No documentation
    # was found.  Don't change this unless you know what you are doing.
    return ngmix.Jacobian(
        x=x - bbox.x.start,
        y=y - bbox.y.start,
        dudx=matrix[1, 1],
        dudy=-matrix[1, 0],
        dvdx=matrix[0, 1],
        dvdy=-matrix[0, 0],
    )


def calculate_positions(bbox, wcs, cat):
    cat['x'] = cat['xcell'] + bbox.x.start
    cat['y'] = cat['ycell'] + bbox.y.start
    cat['x_fit'] = cat['x_fit'] + bbox.x.start
    cat['y_fit'] = cat['y_fit'] + bbox.y.start

    cat['ra'], cat['dec'] = wcs.pixelToSkyArray(
        cat['x'].astype('f8'),
        cat['y'].astype('f8'),
        degrees=True,
    )


def get_wcs_header(wcs, bbox, tract, patch):
    """
    FITS WCS cards for the patch subimage: the tract wcs with
    CRPIX shifted by the patch bounding-box origin (a pure
    origin shift, so any SIP terms stay consistent)
    """
    x0 = bbox.x.start
    y0 = bbox.y.start
    md = wcs.getFitsMetadata()
    hdr = {}
    for key in md.names():
        val = md.getScalar(key)
        if key in ('SIMPLE', 'BITPIX', 'EXTEND') or \
                key.startswith('NAXIS'):
            continue
        hdr[key] = val
    hdr['CRPIX1'] = hdr['CRPIX1'] - x0
    hdr['CRPIX2'] = hdr['CRPIX2'] - y0
    # afw-style subimage origin, for going back to tract coords
    hdr['LTV1'] = -x0
    hdr['LTV2'] = -y0
    hdr['TRACT'] = tract
    hdr['PATCH'] = patch
    return hdr
