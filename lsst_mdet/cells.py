"""
mbobs construction from butler cell coadds
"""
from lsst.images import Box
from lsst.images._cell_grid import CellIJ
from lsst.images._geom import BoundsError
import numpy as np
from .defaults import CELL_SIZE, DM_OUT, MIN_GOOD_FRAC, OVERLAP
from .detect import get_detect_noise, make_kernel
from .structs import get_cell_info
from .wcs import get_cell_jacobian


def make_cell_obs(image, var, good, noise, mfrac, psf_image, jacobian):
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


def pull_mbobs(deep_coadds, cell_i, cell_j, wcs, starmask=None):
    import ngmix

    nband = len(deep_coadds)
    cell_info = get_cell_info(nband)
    cell_info['cell_i'] = cell_i
    cell_info['cell_j'] = cell_j

    mbobs = ngmix.MultiBandObsList()

    for iband, deep_coadd in enumerate(deep_coadds):
        bbox0 = deep_coadd.grid.bbox_of(CellIJ(cell_i, cell_j))

        bbox = Box.factory[
            bbox0.y.start - OVERLAP: bbox0.y.stop + OVERLAP,
            bbox0.x.start - OVERLAP: bbox0.x.stop + OVERLAP,
        ]

        var = deep_coadd.variance[bbox].array.copy()
        mask = deep_coadd.mask[bbox].array[:, :, 0]

        good = np.isfinite(var) & (mask & DM_OUT == 0)

        if starmask is not None:
            # the star attenuation zone (patch-frame array)
            # carries no usable signal after subtraction and
            # apodization
            pb = deep_coadd.bbox
            good &= ~starmask[
                bbox.y.start - pb.y.start:
                bbox.y.stop - pb.y.start,
                bbox.x.start - pb.x.start:
                bbox.x.stop - pb.x.start,
            ]

        w = np.where(good)
        good_frac = w[0].size / var.size
        cell_info['good_frac'][0, iband] = good_frac

        if good_frac > MIN_GOOD_FRAC:

            noise = deep_coadd.noise_realizations[0][bbox].array.copy()
            mfrac = deep_coadd.mask_fractions["rejected"][bbox].array.copy()
            image = deep_coadd.image[bbox].array.copy()

            psf = deep_coadd.psf

            xmid = 0.5 * (bbox.x.start + bbox.x.stop)
            ymid = 0.5 * (bbox.y.start + bbox.y.stop)

            try:
                psf_image = psf.compute_kernel_image(
                    x=xmid,
                    y=ymid,
                ).array
            except BoundsError as err:
                print(err)
                break

            cell_jacobian = get_cell_jacobian(
                wcs=wcs, bbox=bbox, x=xmid, y=ymid,
            )
            obs = make_cell_obs(
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


def get_cell_centers(deep_coadd):
    """
    tract-frame pixel centers of the coadd cells, as (xs, ys)
    1-d arrays whose outer product is the cell grid.  Tries the
    coadd's cell grid attributes; falls back to the
    lsst_cells_v2 150 px inner-cell grid over the patch bbox
    """
    bbox = deep_coadd.bbox
    try:
        from lsst.images._cell_grid import CellIJ

        grid = deep_coadd.grid

        def center(b, ax):
            a = getattr(b, ax)
            return 0.5 * (a.start + a.stop)

        # cell access as in process_cells.py pull_mbobs:
        # grid.bbox_of(CellIJ(i, j)), centers at the bbox
        # midpoint.  Probe which CellIJ axis is y rather than
        # assuming the convention
        gs = grid.grid_size
        try:
            n0, n1 = (int(v) for v in gs)
        except TypeError:
            # grid_size is itself a CellIJ
            if hasattr(gs, 'i'):
                n0, n1 = int(gs.i), int(gs.j)
            else:
                n0, n1 = int(gs.x), int(gs.y)
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
        return (
            xs, ys, (csx, csy), f'coadd grid {nyc} x {nxc}',
        )
    except Exception as err:
        print('cell grid introspection failed:', err)
    csx = csy = CELL_SIZE
    xs = np.arange(bbox.x.start + csx / 2 - 0.5, bbox.x.stop, csx)
    ys = np.arange(bbox.y.start + csy / 2 - 0.5, bbox.y.stop, csy)
    return xs, ys, (csx, csy), f'fallback {CELL_SIZE}px grid'
