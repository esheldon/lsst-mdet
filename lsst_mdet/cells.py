"""
mbobs construction from butler cell coadds
"""
import numpy as np
from .defaults import (
    DM_OUT, MIN_GOOD_FRAC, CELL_OVERLAP,
    CELL_OVERLAP_HIGH, CELL_OVERLAP_LOW, CELL_SIZE,
)
from .detect import get_detect_noise, make_kernel
from .structs import get_cell_meta
from .wcs import get_cell_jacobian


def pull_mbobs(deep_coadds, cell_i, cell_j, wcs, starmask=None,
               skyvars=None):
    """
    pull a MultiBandObsList from the input deep_coadds for the indicated cell.

    The bad pixels (non-finite variance, DM_OUT mask bits, and
    the starmask: the star attenuation zone, plus with the joint
    route the diffuse regions) are unioned across the bands, so
    every band is built on one shared footprint: a pixel with
    good data in only some bands would otherwise feed the joint
    fitting unbalanced information, and can produce zero-variance
    measurements in the bands lacking data.  The cell is kept
    when the shared good fraction exceeds MIN_GOOD_FRAC, which
    subsumes a per-band check: the shared fraction is at most
    the smallest per-band fraction

    Parameters
    ----------
    deep_coadds: list
        list of ButlerCoadd or patchfiles.FilePatchCoadd
    cell_i, cell_j: int
        The cell indices
    wcs: DM wcs object
        wcs used for jacobian
    starmask: bool array, optional
        Patch-frame mask of the pixels to exclude, from
        load_coadds_butler: the star attenuation zone, whose image
        is zeroed and tapered, and with the joint route the grown
        diffuse regions, whose pixels are left in the image.  Both
        get zero weight, bmask 1 and mfrac 1
    skyvars: list, optional
        per-band patch-frame sky-variance maps from
        redo_background, used for the pixel weights.  Entries
        of None fall back to the raw variance plane, which
        includes the objects' poisson noise (diagnostic runs
        only)

    Returns
    -------
    mbobs, cell_meta:
        The MultiBandObsList and a cell info struct (see
        structs.py).  cell_meta good_frac holds each band's own
        good fraction, showing which band drives any union loss;
        the shared fraction actually used goes to the obs meta
    """

    import ngmix

    nband = len(deep_coadds)
    cell_meta = get_cell_meta(nband)
    cell_meta['cell_i'] = cell_i
    cell_meta['cell_j'] = cell_j

    mbobs = ngmix.MultiBandObsList()

    # first pass: the per-band good sets
    bboxes = []
    goods = []
    weight_vars = []
    for iband, deep_coadd in enumerate(deep_coadds):
        bbox = deep_coadd.cell_window(cell_i, cell_j, CELL_OVERLAP)

        var = deep_coadd.variance[bbox].array.copy()
        mask = deep_coadd.mask[bbox].array[:, :, 0]

        good_band = np.isfinite(var) & (mask & DM_OUT == 0)

        pb = deep_coadd.bbox
        patch_cut = np.s_[
            bbox.y.start - pb.y.start:
            bbox.y.stop - pb.y.start,
            bbox.x.start - pb.x.start:
            bbox.x.stop - pb.x.start,
        ]

        if starmask is not None:
            # the star attenuation zone (patch-frame array)
            # carries no usable signal after subtraction and
            # apodization.  With the joint route the mask also
            # covers the diffuse regions (load_coadds_butler),
            # which keep their pixels and only lose their weight
            good_band &= ~starmask[patch_cut]

        # weights come from the sky-variance map when there is
        # one: the variance plane's object poisson term would
        # make them signal-dependent
        weight_var = var
        if skyvars is not None and skyvars[iband] is not None:
            weight_var = skyvars[iband][patch_cut]

        cell_meta['good_frac'][0, iband] = good_band.mean()

        bboxes.append(bbox)
        goods.append(good_band)
        weight_vars.append(weight_var)

    # one shared footprint for every band
    good = np.logical_and.reduce(goods)
    good_frac = float(good.mean())

    if good_frac <= MIN_GOOD_FRAC:
        return None, cell_meta

    for iband, deep_coadd in enumerate(deep_coadds):
        bbox = bboxes[iband]

        noise = deep_coadd.noise_realizations[0][bbox].array.copy()
        mfrac = deep_coadd.mask_fractions["rejected"][bbox].array.copy()
        # every pixel outside the shared footprint is fully
        # masked for selection purposes, matching the cell-edge
        # apodization convention; this includes the star
        # attenuation zone and the diffuse regions
        mfrac[~good] = 1.0
        image = deep_coadd.image[bbox].array.copy()

        xmid = 0.5 * (bbox.x.start + bbox.x.stop)
        ymid = 0.5 * (bbox.y.start + bbox.y.stop)

        psf_image = deep_coadd.psf_image(xmid, ymid)
        if psf_image is None:
            break

        cell_jacobian = get_cell_jacobian(
            wcs=wcs, bbox=bbox, x=xmid, y=ymid,
        )
        obs = _make_cell_obs(
            image=image,
            weight_var=weight_vars[iband],
            good=good,
            noise=noise,
            mfrac=mfrac,
            psf_image=psf_image,
            jacobian=cell_jacobian,
        )
        obs.meta['good_frac'] = good_frac
        obs.meta['bbox'] = bbox
        obs.meta['band'] = deep_coadd.band

        nbad = np.isnan(noise).sum()
        if nbad > 0:
            print(f'cell_i: {cell_i} cell_j: {cell_j} iband: {iband}')
            print(f'  found {nbad} / {noise.size} nan in noise')
            break

        sigma_band = get_detect_noise(
            noise=noise,
            kernel=make_kernel(),
            weight=obs.weight,
        )
        medwt = np.median(obs.weight[obs.weight > 0])
        obs.weight = obs.weight / (sigma_band ** 2 * medwt)

        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    if len(mbobs) < len(deep_coadds):
        return None, cell_meta
    else:
        cell_meta['kept'] = True
        return mbobs, cell_meta


def _make_cell_obs(image, weight_var, good, noise, mfrac,
                   psf_image, jacobian):
    import ngmix

    psf_jacobian = jacobian.copy()
    psf_cen = (np.array(psf_image.shape) - 1.0) / 2.0
    psf_jacobian.set_cen(row=psf_cen[0], col=psf_cen[1])

    psf_obs = ngmix.Observation(
        psf_image,
        weight=psf_image * 0 + 1.0 / 1.0e-6 ** 2,
        jacobian=psf_jacobian,
    )

    weight = np.zeros(image.shape)
    weight[good] = 1.0 / weight_var[good]

    bmask = np.zeros(image.shape, dtype='i4')
    bmask[~good] = 1

    return ngmix.Observation(
        image,
        weight=weight,
        bmask=bmask,
        noise=noise,
        mfrac=mfrac,
        jacobian=jacobian,
        psf=psf_obs,
    )


def get_cell_centers(deep_coadd):
    """
    tract-frame pixel centers of the coadd cells, as (xs, ys)
    1-d arrays whose outer product is the cell grid.

    Conventions per lsst.images._cell_grid.CellGrid: grid_size
    is a CellIJ with i along y and j along x, the grid is
    uniform with cell_shape pixels per cell, and cell (0, 0)
    has its corner at the grid bbox minimum
    """
    grid = deep_coadd.grid
    nyc = int(grid.grid_size.i)
    nxc = int(grid.grid_size.j)
    csy = int(grid.cell_shape.y)
    csx = int(grid.cell_shape.x)
    ys = grid.bbox.y.start + (np.arange(nyc) + 0.5) * csy
    xs = grid.bbox.x.start + (np.arange(nxc) + 0.5) * csx
    return xs, ys, (csx, csy), f'coadd grid {nyc} x {nxc}'


class ButlerCoadd(object):
    """
    thin wrapper around a butler deep_coadd presenting the
    interface pull_mbobs uses; everything not defined here
    passes through to the underlying object.  The stack
    imports live inside the methods so this module imports
    without the LSST pipelines
    """

    def __init__(self, deep_coadd):
        self._dc = deep_coadd

    def __getattr__(self, name):
        return getattr(self._dc, name)

    def cell_window(self, cell_i, cell_j, overlap):
        from lsst.images import Box
        from lsst.images._cell_grid import CellIJ

        b0 = self._dc.grid.bbox_of(CellIJ(cell_i, cell_j))
        return Box.factory[
            b0.y.start - overlap: b0.y.stop + overlap,
            b0.x.start - overlap: b0.x.stop + overlap,
        ]

    def psf_image(self, x, y):
        from lsst.images._geom import BoundsError

        try:
            return self._dc.psf.compute_kernel_image(
                x=x, y=y,
            ).array
        except BoundsError as err:
            print(err)
            return None


def diffuse_mask(starsub_fits, margin):
    """
    the union over the bands of the joint fit's diffuse regions (the
    large diffuse segments it did not mask as sources, the fit dicts'
    'diffuse'), grown by margin px; None when there are none
    """
    from scipy import ndimage

    masks = [fit['diffuse'] for fit in starsub_fits.values()
             if fit.get('diffuse') is not None]
    if not masks:
        return None
    union = np.logical_or.reduce(masks)
    if not union.any():
        return None
    if margin > 0:
        union = ndimage.distance_transform_edt(~union) <= margin
    return union


def load_coadds_butler(butler, tract, patch, bands,
                       redo_bg=False, starsub=False,
                       gaia_file=None, gsub=None,
                       apod_stars=True, starsub_method='template',
                       wing_pattern=None):
    """
    load the deep coadds for a patch from the butler, with the
    optional star subtraction and background redetermination
    applied in that order.  starsub_method 'template' is this
    package's handle_stars (the reference); 'joint' calls
    lsst_starsub.starsub.handle_stars_joint with the per-band
    wing file from wing_pattern ({band} placeholder).  The
    star-region taper uses the
    union of the per-band star masks, so the attenuation zones
    match across the bands.  Returns
    (coadds, wcs, starmask, star_table, apod, tract_bounds,
    skyvars, starsub_fits) with the coadds wrapped for pull_mbobs,
    the wcs wrapped for the jacobian helper, the star census and
    taper width for the footprint (None and 0 without starsub),
    the tract inner sky bounds for the primary cut and the
    footprint trim, the per-band sky-variance maps for the pixel
    weights (None entries without the background redo), and for
    the joint method the per-band fit dicts (band -> fit) for
    lsst_starsub.starsub.make_fit_tables, else None.  With the
    joint method, the returned starmask also includes the large
    diffuse segments (cirrus) that the sky fit did not mask as
    sources (lsst_starsub.joint SEG_DIFFUSE_MEDIAN), grown by
    DIFFUSE_MARGIN px (diffuse_mask).  Unlike the star zones, their
    pixels are left in the image, so no taper is needed: they get
    zero weight, bmask 1 and mfrac 1 in the cells (pull_mbobs), so
    detection skips them and objects there fail the mfrac cut, and
    they are cleared from the footprint
    """
    from .background import redo_background
    from .defaults import SKYMAP_VERS
    from .gaia import GMAX, fetch_gaia, read_gaia_file
    from .inject import INJECT_SETTINGS, inject_objects, read_truth
    from .starsub import (
        APOD_STARS, BG_GROW, DIFFUSE_MARGIN, GSUB, apply_star_taper,
        handle_stars,
    )
    from .wcs import ButlerWcs

    if gsub is None:
        gsub = GSUB

    if not redo_bg:
        print('WARNING: no background redo: diagnostic mode '
              'only; pixel weights will use the raw variance '
              'plane, which includes object poisson noise')

    skymap = butler.get("skyMap", skymap=SKYMAP_VERS)

    tract_info = skymap[tract]
    tract_bounds = get_tract_bounds(tract_info)
    wcs = ButlerWcs(tract_info.wcs)

    coadds = []
    gaia = None
    dstar_min = None
    star_table = None
    starsub_fits = None
    skyvars = []
    truth = None
    for band in bands:
        data_id = {
            "band": band,
            "skymap": SKYMAP_VERS,
            "tract": tract,
            "patch": patch,
        }
        print(data_id)
        deep_coadd = butler.get('deep_coadd', dataId=data_id)
        deep_coadd.apply_background('object')

        # the object injection test (lsst_mdet.inject, set up by
        # lsst-mdet-inject-node): the objects go in before any star
        # or sky processing, so they go through the whole chain.
        # apply_background adds and subtracts in place, so they
        # survive the joint route's return to the None state
        if INJECT_SETTINGS['objects'] is not None:
            if truth is None:
                truth = read_truth(INJECT_SETTINGS['objects'].format(
                    tract=tract, patch=patch,
                ))
            inject_objects(deep_coadd, wcs, truth)

        if starsub:
            if gaia is None:
                gmax = max(gsub, GMAX)
                if gaia_file is not None:
                    gaia = read_gaia_file(
                        gaia_file, wcs, deep_coadd.bbox,
                        gmax=gmax,
                    )
                else:
                    gaia = fetch_gaia(
                        wcs, deep_coadd.bbox, gmax=gmax,
                    )
            # the getimages sequence: subtract, background,
            # then the taper LAST, so the star holes stay
            # exactly zero (tapering first would leave -bkg
            # inside them after the background subtraction).
            # The taper is further deferred to after this loop:
            # its distance field must be shared across the
            # bands so the attenuation zones match
            if starsub_method == 'joint':
                from lsst_starsub.starsub import (
                    handle_stars_joint, load_wing,
                )
                wing = load_wing(wing_pattern.format(band=band))
                starmask_b, stable_b, dstar, fit = handle_stars_joint(
                    deep_coadd, wcs, gaia, wing, gsub=gsub,
                )
                if starsub_fits is None:
                    starsub_fits = {}
                starsub_fits[band] = fit
            else:
                starmask_b, stable_b, dstar = handle_stars(
                    deep_coadd, wcs, gaia, gsub=gsub,
                    subtract=True,
                )
            if star_table is None:
                # the census is the same in every band up to
                # per-band saturation details; keep the first
                star_table = stable_b
            skyvar = None
            if redo_bg:
                # the joint route has fit the sky already; the
                # redo then only calibrates the noise and makes
                # the sky-variance map
                skyvar = redo_background(
                    deep_coadd, starmask=dstar < BG_GROW,
                    subtract=(starsub_method != 'joint'),
                )
            # the distance to the union of the per-band star
            # masks is the minimum of the per-band distances
            if dstar_min is None:
                dstar_min = dstar
            else:
                np.minimum(dstar_min, dstar, out=dstar_min)
        elif redo_bg:
            skyvar = redo_background(deep_coadd, starmask=None)
        else:
            skyvar = None
        skyvars.append(skyvar)
        coadds.append(ButlerCoadd(deep_coadd))

    # one mask and one taper for all bands: consistent
    # footprints downstream, and matching attenuation zones in
    # the images and noise
    starmask = None
    apod = 0.0
    if starsub:
        if apod_stars:
            for coadd in coadds:
                apply_star_taper(
                    coadd, dstar_min, width=APOD_STARS,
                )
            apod = APOD_STARS
        starmask = dstar_min < APOD_STARS
        print(f'union star mask fraction {starmask.mean():.3f}')
        if starsub_fits is not None:
            diffuse = diffuse_mask(starsub_fits, DIFFUSE_MARGIN)
            if diffuse is not None:
                print(f'diffuse mask fraction {diffuse.mean():.3f} '
                      f'(grown {DIFFUSE_MARGIN} px)')
                starmask |= diffuse

    return (
        coadds, wcs, starmask, star_table, apod, tract_bounds,
        skyvars, starsub_fits,
    )


def make_psf_cube(deep_coadd, xs, ys):
    """
    evaluate the psf at every cell center for the getimages
    output.  The stamp shape comes from the first successful
    evaluation; failed cells (edge BoundsError) stay zero with
    ok=0 in the cell table.

    Parameters
    ----------
    deep_coadd: deep_coadd
        The butler coadd
    xs, ys: arrays
        Tract-frame cell center coordinates, whose outer
        product is the cell grid

    Returns
    -------
    psf_stack, cells:
        (ncell, ny, nx) f4 psf stamps and the row-matched cell
        table with patch-frame centers; (None, None) if no
        evaluation succeeded anywhere.  Stamp shape is constant
        across cells, guaranteed by the lsst.images psf classes
    """
    from lsst.images._geom import BoundsError

    psf = deep_coadd.psf
    bbox = deep_coadd.bbox

    psf_stack = None
    pshape = None
    cell_rows = []
    nfail = 0
    for cy in ys:
        for cx in xs:
            kim = None
            try:
                kim = psf.compute_kernel_image(x=cx, y=cy).array
            except BoundsError:
                nfail += 1
            if kim is not None and psf_stack is None:
                pshape = kim.shape
                psf_stack = np.zeros(
                    (ys.size * xs.size,) + pshape,
                    dtype='f4',
                )
            cell_rows.append([cx, cy, kim])
    if psf_stack is None:
        return None, None
    if nfail > 0:
        print(f'    {nfail} cells failed psf evaluation')

    cells = np.zeros(len(cell_rows), dtype=[
        ('cellx', 'f4'), ('celly', 'f4'), ('ok', 'i2'),
    ])
    for k, (cx, cy, kim) in enumerate(cell_rows):
        # patch-frame pixel coordinates
        cells['cellx'][k] = cx - bbox.x.start
        cells['celly'][k] = cy - bbox.y.start
        if kim is not None:
            psf_stack[k] = kim
            cells['ok'][k] = 1
    return psf_stack, cells


def get_cell_primary(x, y):
    """
    Returns True if x, y are in the primary region of the cell,
    False if in the overlap region
    """
    return (
        (x > CELL_OVERLAP_LOW)
        & (x < CELL_OVERLAP_HIGH)
        & (y > CELL_OVERLAP_LOW)
        & (y < CELL_OVERLAP_HIGH)
    )


def get_tract_bounds(tract_info):
    """
    the tract inner sky region as plain
    (ra_min, ra_max, dec_min, dec_max) degrees.  The ra range
    runs from ra_min to ra_max in the direction of increasing
    ra, wrapping through 360 when ra_max < ra_min (a sphgeom
    lon interval convention)

    Parameters
    ----------
    tract_info: TractInfo
        From skymap[tract]
    """
    ip = tract_info.inner_sky_region
    return (
        ip.getLon().getA().asDegrees(),
        ip.getLon().getB().asDegrees(),
        ip.getLat().getA().asDegrees(),
        ip.getLat().getB().asDegrees(),
    )


def get_tract_primary(tract_bounds, ra, dec):
    """
    True where (ra, dec) is within the tract inner region:
    a box bounded by lines of constant ra and constant dec.

    The RA range runs from ra_min to ra_max in the direction
    of *increasing* ra, wrapping through 360 if
    ra_max < ra_min.

    Parameters
    ----------
    tract_bounds: (ra_min, ra_max, dec_min, dec_max)
        From get_tract_bounds in butler mode or the patch file
        header in file mode
    ra, dec: arrays
        positions in degrees
    """
    ra_min, ra_max, dec_min, dec_max = tract_bounds

    width = (ra_max - ra_min) % 360.0
    if width == 0.0 and ra_max != ra_min:
        width = 360.0          # full wrap, e.g. 0 -> 360

    dra = (np.asarray(ra) - ra_min) % 360.0

    return (dra <= width) & (dec >= dec_min) & (dec <= dec_max)


def get_cell_healsparse_polygon(bbox, cell_i, cell_j, wcs):
    """
    Polygon over the inner footprint of cell (cell_i, cell_j).

    The corners come from the same arithmetic as the cell
    windows: patch outer bbox start plus index * CELL_SIZE,
    verified against skymap CellInfo.getInnerBBox

    Parameters
    ----------
    bbox: bbox
        The patch (outer) bounding box, tract frame
    cell_i, cell_j: int
        The cell indices, i along y and j along x
    wcs: ButlerWcs or FileWcs

    Returns
    -------
    healsparse.Polygon
    """
    from .hmaps import make_patch_polygon

    x0 = bbox.x.start + cell_j * CELL_SIZE
    y0 = bbox.y.start + cell_i * CELL_SIZE
    xv = np.array(
        [x0, x0 + CELL_SIZE, x0 + CELL_SIZE, x0], dtype='f8',
    )
    yv = np.array(
        [y0, y0, y0 + CELL_SIZE, y0 + CELL_SIZE], dtype='f8',
    )
    ra, dec = wcs.pixelToSkyArray(xv, yv, degrees=True)
    return make_patch_polygon(ra=ra, dec=dec)
