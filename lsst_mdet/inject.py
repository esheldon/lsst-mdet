"""
inject tabulated residual-halo light around the census stars, for
the amplified differential injection test: reprocess patches with
the field added at a known amplification, pair object-by-object
against the production catalogs (same seeds), and measure the
full-pipeline response -- detection, deblending, measurement and
selection -- to residual star light.

The profile file is the prototype-seq output (run notes
2026-09-07): stacked final-state mean blank-sky profiles per band
and star G slice, in sky-sigma units, versus the distance beyond
the mask radius law in pixels.
"""
import numpy as np

INJECT_GMAX = 16.0
DEFAULT_SCALE = 5.0
SEQ = 'P'

# the injection request, set by the lsst-mdet-inject-node CLI
# before processing starts (the target-psf METACAL_SETTINGS
# pattern: the node driver forks its children from the parent, so
# a parent-set value reaches every patch).  profiles None means
# no injection; the values are recorded in the output provenance
# either way
INJECT_SETTINGS = {
    'profiles': None,
    'scale': DEFAULT_SCALE,
    'gmax': INJECT_GMAX,
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
