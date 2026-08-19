"""
mbobs construction from butler cell coadds
"""
import numpy as np
from .defaults import CELL_SIZE, DM_OUT, MIN_GOOD_FRAC, OVERLAP
from .detect import get_detect_noise, make_kernel
from .structs import get_cell_info
from .wcs import get_cell_jacobian


def pull_mbobs(deep_coadds, cell_i, cell_j, wcs, starmask=None):
    """
    pull a MultiBandObsList from the input deep_coadds for the indicated cell.

    Parameters
    ----------
    deep_coadds: list
        list of ButlerCoadd or patchfiles.FilePatchCoadd
    cell_i, cell_j: int
        The cell indices
    wcs: DM wcs object
        wcs used for jacobian
    starmask: bool, optional

    Returns
    -------
    mbobs, cell_info:
        The MultiBandObsList and a cell info struct (see structs.py)
    """

    import ngmix

    nband = len(deep_coadds)
    cell_info = get_cell_info(nband)
    cell_info['cell_i'] = cell_i
    cell_info['cell_j'] = cell_j

    mbobs = ngmix.MultiBandObsList()

    for iband, deep_coadd in enumerate(deep_coadds):
        bbox = deep_coadd.cell_window(cell_i, cell_j, OVERLAP)

        var = deep_coadd.variance[bbox].array.copy()
        mask = deep_coadd.mask[bbox].array[:, :, 0]

        good = np.isfinite(var) & (mask & DM_OUT == 0)

        smcut = None
        if starmask is not None:
            # the star attenuation zone (patch-frame array)
            # carries no usable signal after subtraction and
            # apodization
            pb = deep_coadd.bbox
            smcut = starmask[
                bbox.y.start - pb.y.start:
                bbox.y.stop - pb.y.start,
                bbox.x.start - pb.x.start:
                bbox.x.stop - pb.x.start,
            ]
            good &= ~smcut

        w = np.where(good)
        good_frac = w[0].size / var.size
        cell_info['good_frac'][0, iband] = good_frac

        if good_frac > MIN_GOOD_FRAC:

            noise = deep_coadd.noise_realizations[0][bbox].array.copy()
            mfrac = deep_coadd.mask_fractions["rejected"][bbox].array.copy()
            if smcut is not None:
                # the star zone is fully masked for selection
                # purposes, matching the cell-edge apodization
                # convention
                mfrac[smcut] = 1.0
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
                var=var,
                good=good,
                noise=noise,
                mfrac=mfrac,
                psf_image=psf_image,
                jacobian=cell_jacobian,
            )
            obs.meta['good_frac'] = good_frac
            obs.meta['bbox'] = bbox
            obs.meta['band'] = deep_coadd.band

            noise_bad = np.isnan(noise)
            wnoise_bad = np.where(noise_bad)
            assert np.all(np.isnan(noise[wnoise_bad]))
            # assert np.all(np.isnan(image[wnoise_bad]))
            if wnoise_bad[0].size > 0:
                print(f'cell_i: {cell_i} cell_j: {cell_j} iband: {iband}')
                print(
                    f'  found {wnoise_bad[0].size} / {noise.size} nan in noise'
                )
                if False:
                    png = 'bad.jpg'
                    import matplotlib.pyplot as mplt
                    with mplt.style.context('dark_background'):
                        fig, axs = mplt.subplots(
                            ncols=2, nrows=2, figsize=(10, 10)
                        )
                        axs[0, 0].set_title('image')
                        axs[0, 0].imshow(
                            np.log10(image.clip(min=0.001)), cmap='gray'
                        )
                        axs[0, 1].set_title('noise')
                        axs[0, 1].imshow(noise, cmap='gray')
                        axs[1, 0].set_title('mask')
                        axs[1, 0].imshow(mask, cmap='gray')
                        axs[1, 1].set_title('var')
                        axs[1, 1].imshow(var, cmap='gray')
                        fig.savefig(png)

                    import IPython
                    IPython.embed()
                break

            sigma_band = get_detect_noise(
                noise=noise,
                kernel=make_kernel(),
                weight=obs.weight,
            )
            medwt = np.median(obs.weight[obs.weight > 0])
            # print(f'    median weight: {medwt:g} sigma_band: {sigma_band:g}')
            # if not np.isfinite(sigma_band):
            #     import IPython; IPython.embed()
            obs.weight = obs.weight / (sigma_band ** 2 * medwt)

            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)
        else:
            # print(f'    good frac {good_frac} < {MIN_GOOD_FRAC}')
            break

    if len(mbobs) < len(deep_coadds):
        return None, cell_info
    else:
        cell_info['kept'] = True
        return mbobs, cell_info


def _make_cell_obs(image, var, good, noise, mfrac, psf_image, jacobian):
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
    weight[good] = 1.0 / var[good]

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
    1-d arrays whose outer product is the cell grid, from the
    coadd's cell grid via grid.bbox_of(CellIJ(i, j)) as in
    pull_mbobs.  Any API surprise raises: the psf evaluation
    positions must come from the real grid
    """
    from lsst.images._cell_grid import CellIJ

    grid = deep_coadd.grid

    def center(b, ax):
        a = getattr(b, ax)
        return 0.5 * (a.start + a.stop)

    # grid_size may be a plain pair or itself a CellIJ
    gs = grid.grid_size
    try:
        n0, n1 = (int(v) for v in gs)
    except TypeError:
        if hasattr(gs, 'i'):
            n0, n1 = int(gs.i), int(gs.j)
        else:
            n0, n1 = int(gs.x), int(gs.y)

    # probe which CellIJ axis is y rather than assuming the
    # convention
    b00 = grid.bbox_of(CellIJ(0, 0))
    i_is_y = True
    if n0 > 1:
        b10 = grid.bbox_of(CellIJ(1, 0))
        i_is_y = b10.y.start != b00.y.start
    if i_is_y:
        nyc, nxc = n0, n1
        ys = np.array([
            center(grid.bbox_of(CellIJ(i, 0)), 'y')
            for i in range(nyc)
        ])
        xs = np.array([
            center(grid.bbox_of(CellIJ(0, j)), 'x')
            for j in range(nxc)
        ])
    else:
        nxc, nyc = n0, n1
        xs = np.array([
            center(grid.bbox_of(CellIJ(i, 0)), 'x')
            for i in range(nxc)
        ])
        ys = np.array([
            center(grid.bbox_of(CellIJ(0, j)), 'y')
            for j in range(nyc)
        ])
    csx = int(round(xs[1] - xs[0])) if nxc > 1 else CELL_SIZE
    csy = int(round(ys[1] - ys[0])) if nyc > 1 else CELL_SIZE
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


def load_coadds_butler(butler, tract, patch, bands,
                       redo_bg=False, starsub=False):
    """
    load the deep coadds for a patch from the butler, with the
    optional star subtraction and background redetermination
    applied in that order.  Returns (coadds, wcs, starmask)
    with the coadds wrapped for pull_mbobs and the wcs wrapped
    for the jacobian helper
    """
    from .background import redo_background
    from .defaults import SKYMAP_VERS
    from .gaia import fetch_gaia
    from .starsub import subtract_and_mask_stars
    from .wcs import ButlerWcs

    skymap = butler.get("skyMap", skymap=SKYMAP_VERS)
    wcs = ButlerWcs(skymap[tract].wcs)

    coadds = []
    gaia = None
    starmasks = []
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

        smband = None
        if starsub:
            if gaia is None:
                gaia = fetch_gaia(wcs, deep_coadd.bbox)
            smband = subtract_and_mask_stars(
                deep_coadd, wcs, gaia,
            )
            starmasks.append(smband)
        if redo_bg:
            redo_background(deep_coadd, starmask=smband)
        coadds.append(ButlerCoadd(deep_coadd))

    # one mask for all bands: consistent footprints downstream
    starmask = None
    if starsub:
        starmask = np.logical_or.reduce(starmasks)
        print(f'union star mask fraction {starmask.mean():.3f}')

    return coadds, wcs, starmask


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
        evaluation succeeded anywhere
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
        if kim is not None and kim.shape == pshape:
            psf_stack[k] = kim
            cells['ok'][k] = 1
    return psf_stack, cells
