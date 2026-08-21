"""
file-backed patch loading: everything process_cells needs,
read from the getimages FITS output instead of the butler, so
a patch can be processed (and tested) without the LSST stack.

The objects duck-type the small slice of the DM API that
pull_mbobs and the wcs helpers consume.  Background
redetermination and star subtraction are getimages-time
operations and are refused here: one canonical place per
operation keeps provenance clean
"""
import numpy as np

from .defaults import CELL_SIZE


def load_coadds_files(patch_dir, tract, patch, bands):
    """
    load a patch from getimages FITS output: the file-backed
    counterpart of the butler loader.  Returns
    (coadds, wcs, starmask, star_table, apod, tract_bounds),
    with starmask the combined attenuation-zone mask when the
    files carry a starmask extension (getimages --starsub) and
    None otherwise, the star census and taper width for the
    footprint (None and 0 without starsub), and the tract
    inner sky bounds from the header
    """
    coadds = []
    starmask = None
    for band in bands:
        fname = get_patch_filename(
            tract=tract, patch=patch, band=band, patch_dir=patch_dir,
        )
        print(f'reading {fname}')
        coadd = FilePatchCoadd(fname, band)
        coadds.append(coadd)
        if coadd._starmask is not None:
            sm = coadd._starmask >= 1
            starmask = sm if starmask is None \
                else (starmask | sm)
    hdr = coadds[0].hdr
    wcs = FileWcs(hdr)
    tract_bounds = (
        hdr['TRAMIN'], hdr['TRAMAX'],
        hdr['TDECMIN'], hdr['TDECMAX'],
    )
    return (
        coadds, wcs, starmask,
        coadds[0]._star_table, coadds[0]._apod,
        tract_bounds,
    )


def get_patch_filename(tract, patch, band, patch_dir=None):
    """
    Get the standard patch file name

    Parameters
    ----------
    tract: int
        The tract id
    patch: int
        The patch id
    patch_dir: str, optional
        Optional directory

    Returns
    -------
    filename
    """
    import os

    fname = f'{tract:05d}-{patch:02d}-{band}.fits'

    if patch_dir is not None:
        fname = os.path.join(patch_dir, fname)

    return fname


class SimpleAxis(object):
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop


class SimpleBox(object):
    """bbox with the DM attribute layout: .x/.y start/stop"""
    def __init__(self, x0, x1, y0, y1):
        self.x = SimpleAxis(x0, x1)
        self.y = SimpleAxis(y0, y1)


class _PlaneView(object):
    """
    plane wrapper so that view[bbox].array mirrors the DM
    plane slicing used by pull_mbobs.  The stored array is
    patch frame; bboxes are tract frame
    """
    def __init__(self, arr, x0, y0):
        self.arr = arr
        self.x0 = x0
        self.y0 = y0

    def __getitem__(self, bbox):
        cut = self.arr[
            bbox.y.start - self.y0:bbox.y.stop - self.y0,
            bbox.x.start - self.x0:bbox.x.stop - self.x0,
        ]
        return _PlaneCut(cut)

    @property
    def array(self):
        return self.arr


class _PlaneCut(object):
    def __init__(self, arr):
        self.array = arr


class FilePatchCoadd(object):
    """
    one band of a patch, read from a getimages FITS file,
    presenting the deep_coadd interface pull_mbobs uses
    """

    def __init__(self, fname, band):
        import rustfits

        self.band = band
        with rustfits.FITS(fname) as f:
            extnames = [f[i].extname for i in range(len(f))]
            hdu = f['image']
            hdr = hdu.header
            self._image = np.ascontiguousarray(
                hdu.read(), dtype='f4',
            )
            self._var = f['var'].read()
            self._mask = f['mask'].read()
            self._noise = f['noise'].read()
            self._mfrac = f['mfrac'].read()
            self._psfs = f['psfs'].read().astype('f8')
            self._cells = f['psf_cells'].read()
            self._starmask = None
            self._apod = 0.0
            if 'starmask' in extnames:
                hdu = f['starmask']
                self._starmask = hdu.read()
                self._apod = float(hdu.header['APOD'])
            self._star_table = None
            if 'gaia_stars' in extnames:
                self._star_table = f['gaia_stars'].read()

        self.hdr = hdr
        # LTV records the afw subimage origin: patch = tract - x0
        x0 = -int(hdr['LTV1'])
        y0 = -int(hdr['LTV2'])
        ny, nx = self._image.shape
        self.bbox = SimpleBox(x0, x0 + nx, y0, y0 + ny)

        self.image = _PlaneView(self._image, x0, y0)
        self.variance = _PlaneView(self._var, x0, y0)
        self.mask = _PlaneView(self._mask, x0, y0)
        self.noise_realizations = [
            _PlaneView(self._noise, x0, y0),
        ]
        self.mask_fractions = {
            'rejected': _PlaneView(self._mfrac, x0, y0),
        }

    def cell_window(self, cell_i, cell_j, overlap):
        """
        the padded processing window of cell (i, j), tract
        frame, i along y and j along x matching the DM cell
        grid convention
        """
        y0 = self.bbox.y.start + cell_i * CELL_SIZE
        x0 = self.bbox.x.start + cell_j * CELL_SIZE
        return SimpleBox(
            x0 - overlap, x0 + CELL_SIZE + overlap,
            y0 - overlap, y0 + CELL_SIZE + overlap,
        )

    def psf_image(self, x, y):
        """
        psf at tract position (x, y): the nearest ok cell of
        the per-cell psf cube (the cube is evaluated at the
        cell centers, so interior windows match the butler
        evaluation exactly).  None if no cell is usable
        """
        ok = self._cells['ok'] == 1
        if not ok.any():
            return None
        px = x - self.bbox.x.start
        py = y - self.bbox.y.start
        d2 = (
            (self._cells['cellx'] - px) ** 2
            + (self._cells['celly'] - py) ** 2
        )
        d2[~ok] = np.inf
        return self._psfs[int(np.argmin(d2))]


class _Angle(object):
    def __init__(self, deg):
        self.deg = deg

    def asDegrees(self):
        return self.deg


class _SpherePoint(object):
    """the slice of the DM SpherePoint API fetch_gaia uses"""

    def __init__(self, ra, dec):
        self.ra = ra
        self.dec = dec

    def getRa(self):
        return _Angle(self.ra)

    def getDec(self):
        return _Angle(self.dec)

    def separation(self, other):
        import numpy as np

        r1, d1, r2, d2 = map(np.deg2rad, (
            self.ra, self.dec, other.ra, other.dec,
        ))
        cossep = (
            np.sin(d1) * np.sin(d2)
            + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
        )
        return _Angle(float(np.rad2deg(
            np.arccos(np.clip(cossep, -1, 1)),
        )))


class FileWcs(object):
    """
    astropy-backed tract wcs from the patch header, presenting
    the DM methods the pipeline uses.  All pixel arguments are
    tract frame, matching the DM tract wcs
    """

    def __init__(self, hdr):
        from astropy.wcs import WCS

        x0 = -int(hdr['LTV1'])
        y0 = -int(hdr['LTV2'])
        cards = {
            k: hdr[k] for k in (
                'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2',
                'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'RADESYS',
            )
        }
        # back to tract-frame CRPIX
        cards['CRPIX1'] = hdr['CRPIX1'] + x0
        cards['CRPIX2'] = hdr['CRPIX2'] + y0
        cards['NAXIS'] = 2
        self.wcs = WCS(cards)

    def pixelToSky(self, x, y):
        """scalar pixel -> sky, DM style: an object with
        getRa/getDec (asDegrees) and separation"""
        ra, dec = self.wcs.wcs_pix2world(float(x), float(y), 0)
        return _SpherePoint(float(ra), float(dec))

    def pixelToSkyArray(self, x, y, degrees=True):
        ra, dec = self.wcs.wcs_pix2world(x, y, 0)
        return ra, dec

    def skyToPixelArray(self, ra, dec, degrees=True):
        return self.wcs.wcs_world2pix(ra, dec, 0)

    def linearize_matrix(self, x, y):
        """
        local linearization of pixel -> sky in arcsec per
        pixel, rows (true ra offset, dec offset), by central
        finite differences: the file-backed analogue of the DM
        linearizePixelToSky matrix consumed by
        get_cell_jacobian
        """
        eps = 0.5
        ras, decs = self.wcs.wcs_pix2world(
            np.array([x + eps, x - eps, x, x]),
            np.array([y, y, y + eps, y - eps]),
            0,
        )
        cosd = np.cos(np.deg2rad(decs.mean()))
        dra_dx = (ras[0] - ras[1]) / (2 * eps) * cosd
        dra_dy = (ras[2] - ras[3]) / (2 * eps) * cosd
        ddec_dx = (decs[0] - decs[1]) / (2 * eps)
        ddec_dy = (decs[2] - decs[3]) / (2 * eps)
        return np.array([
            [dra_dx, dra_dy],
            [ddec_dx, ddec_dy],
        ]) * 3600.0
