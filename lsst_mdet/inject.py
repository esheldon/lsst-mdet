"""
injection tests, two kinds.

Residual halos: tabulated residual-halo light around the census
stars, for the amplified differential injection test: reprocess
patches with the field added at a known amplification, pair
object-by-object against the production catalogs (same seeds), and
measure the full-pipeline response -- detection, deblending,
measurement and selection -- to residual star light.  The profile
file is the prototype-seq output (run notes 2026-09-07): stacked
final-state mean blank-sky profiles per band and star G slice, in
sky-sigma units, versus the distance beyond the mask radius law in
pixels.

Objects: exponential galaxies and point sources from a per-patch
truth table, rendered with the coadd psf and added to the coadds as
loaded, before any star or sky processing, so they go through the
whole chain (the recovery of fluxes and colours under diffuse
emission such as cirrus).
"""
import numpy as np

INJECT_GMAX = 16.0
DEFAULT_SCALE = 5.0
SEQ = 'P'

# object stamps: odd, at least STAMP_MIN px on a side and at least
# STAMP_NHLR half-light radii to each side
STAMP_MIN = 49
STAMP_NHLR = 10.0

# the truth table columns an object injection reads: patch-frame
# pixel position, per-band flux in nJy, the exponential's
# half-light radius (arcsec) and shear, and whether it is a point
# source
TRUTH_COLUMNS = ('x', 'y', 'flux_r', 'flux_i', 'flux_z', 'hlr',
                 'g1', 'g2', 'is_star')

# the injection request, set by the lsst-mdet-inject-node CLI
# before processing starts (the target-psf METACAL_SETTINGS
# pattern: the node driver forks its children from the parent, so
# a parent-set value reaches every patch).  profiles None means
# no halo injection, objects None (a truth file pattern with
# {tract} and {patch}) no object injection; the values are
# recorded in the output provenance either way
INJECT_SETTINGS = {
    'profiles': None,
    'scale': DEFAULT_SCALE,
    'gmax': INJECT_GMAX,
    'objects': None,
}


def load_profiles(fname):
    """
    the tabulated profiles

    Returns
    -------
    profs: dict band -> list of ((glo, ghi), eta)
    dmcen: array, bin centers in pixels beyond the mask radius
    """
    import rustfits

    with rustfits.FITS(fname) as fits:
        prof = fits['profiles'].read()
        redges = fits['edges'].read()['redges'][0]
    dmcen = 0.5 * (redges[:-1] + redges[1:])

    profs = {}
    for p in prof:
        if str(p['seq']) != SEQ:
            continue
        eta = np.where(p['n'] > 0, p['mean'], 0.0)
        profs.setdefault(str(p['band']), []).append(
            ((float(p['glo']), float(p['ghi'])), eta),
        )
    return profs, dmcen


def inject_residual_halos(coadds, star_table, profile_file,
                          scale=DEFAULT_SCALE, gmax=INJECT_GMAX):
    """
    add scale times the tabulated residual profile around each
    census star brighter than gmax, per band, in that band's
    sky-sigma units.  The field is zero inside the mask radius
    and beyond the profile support.  The images are modified in
    place; call after all star handling, background work and
    tapering so the injected light rides on the final image
    state

    Parameters
    ----------
    coadds: list of ButlerCoadd or FilePatchCoadd
        The final processed bands
    star_table: array
        The census (x, y patch frame, G)
    profile_file: str
        The prototype-seq profiles fits file
    scale: float, optional
        The amplification of the injected field
    gmax: float, optional
        Inject around stars brighter than this

    Returns
    -------
    the number of (star, band) injections
    """
    from .starsub import circle_radius

    profs, dmcen = load_profiles(profile_file)
    rmax_dm = float(dmcen[-1]) + 8.0

    stars = star_table[star_table['G'] < gmax]
    ninj = 0
    for coadd in coadds:
        image = coadd.image.array
        var = coadd.variance.array
        good = np.isfinite(image) & np.isfinite(var) & (var > 0)
        sig = float(np.sqrt(np.nanmedian(var[good])))
        ny, nx = image.shape
        band_profs = profs[coadd.band]

        for st in stars:
            gval = float(st['G'])
            eta = None
            for (glo, ghi), e in band_profs:
                if glo <= gval < ghi:
                    eta = e
                    break
            if eta is None:
                continue
            rad = float(circle_radius(gval))
            m = int(rad + rmax_dm + 2)
            icx = int(round(float(st['x'])))
            icy = int(round(float(st['y'])))
            x0, x1 = max(0, icx - m), min(nx, icx + m + 1)
            y0, y1 = max(0, icy - m), min(ny, icy + m + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            gy, gx = np.mgrid[y0:y1, x0:x1]
            dm = np.hypot(
                gy - float(st['y']), gx - float(st['x']),
            ) - rad
            field = np.interp(dm, dmcen, eta, right=0.0)
            field[dm < 0] = 0.0
            image[y0:y1, x0:x1] += scale * sig * field
            ninj += 1

    print(f'    injected {ninj} star-band residual halos '
          f'at scale {scale:g} (G < {gmax:g})')
    return ninj


def read_truth(fname):
    """
    the object truth table of a patch

    Parameters
    ----------
    fname: str
        FITS file with the TRUTH_COLUMNS in extension 1

    Returns
    -------
    structured array
    """
    import rustfits

    with rustfits.FITS(fname) as fits:
        truth = fits[1].read()
    missing = [c for c in TRUTH_COLUMNS if c not in truth.dtype.names]
    if missing:
        raise ValueError(f'{fname}: the truth table lacks {missing}')
    return truth


def inject_objects(coadd, wcs, truth):
    """
    add the truth objects to one band of a patch coadd, in place.

    Each object is an exponential of the given half-light radius and
    shear (a point source where is_star), convolved with the coadd
    psf at its position and drawn on the local affine wcs there,
    with the band's flux in nJy.  The psf kernel image includes the
    pixel, so the stamps are drawn without a second pixel
    convolution.  Stamps are clipped at the image edge; the objects
    carry no noise

    Parameters
    ----------
    coadd: deep_coadd
        The butler coadd as loaded (the 'object' background
        applied), with band, bbox, psf and image
    wcs: ButlerWcs
        The tract wcs, for the local jacobian
    truth: structured array
        With the TRUTH_COLUMNS; x, y in the patch frame

    Returns
    -------
    the number of objects drawn
    """
    import galsim

    image = coadd.image.array
    ny, nx = image.shape
    bbox = coadd.bbox
    fcol = f'flux_{coadd.band}'
    ndrawn = 0
    for obj in truth:
        x, y = float(obj['x']), float(obj['y'])
        xt, yt = x + bbox.x.start, y + bbox.y.start
        mat = wcs.linearize_matrix(xt, yt)
        # the convention of wcs.get_cell_jacobian
        jac = galsim.JacobianWCS(
            mat[1, 1], -mat[1, 0], mat[0, 1], -mat[0, 0],
        )
        kim = coadd.psf.compute_kernel_image(x=xt, y=yt).array
        psf = galsim.InterpolatedImage(
            galsim.Image(kim.astype('f8'), wcs=jac), flux=1.0,
        )
        flux = float(obj[fcol])
        if bool(obj['is_star']):
            prof = galsim.DeltaFunction(flux=flux)
            half = STAMP_MIN // 2
        else:
            hlr = float(obj['hlr'])
            prof = galsim.Exponential(
                half_light_radius=hlr, flux=flux,
            ).shear(g1=float(obj['g1']), g2=float(obj['g2']))
            scale = np.sqrt(abs(np.linalg.det(mat)))
            half = max(STAMP_MIN // 2,
                       int(np.ceil(STAMP_NHLR * hlr / scale)))
        n = 2 * half + 1
        ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        stamp = galsim.Convolve(prof, psf).drawImage(
            nx=n, ny=n, wcs=jac, offset=(x - ix, y - iy),
            method='no_pixel',
        ).array
        x0, y0 = ix - half, iy - half
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1 = n - max(0, x0 + n - nx)
        sy1 = n - max(0, y0 + n - ny)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        image[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1] += \
            stamp[sy0:sy1, sx0:sx1]
        ndrawn += 1

    print(f'    band {coadd.band}: injected {ndrawn} objects')
    return ndrawn
