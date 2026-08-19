"""
cell-edge and star-mask apodization
"""
from numba import njit
import numpy as np


AP_RAD = 1.5


def apodize_mbobs(mbobs):
    """
    Apply an apodization mask around the edge of the images in an mbobs to
    prevent FFT artifacts.

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to mask
    """

    ap_mask = np.ones_like(mbobs[0][0].image)

    _build_square_apodization_mask(AP_RAD, ap_mask)

    msk = np.where(ap_mask < 1)

    if msk[0].size > 0:
        for obslist in mbobs:
            for obs in obslist:

                # the pixels list will be reset upon exiting
                with obs.writeable():

                    obs.image[msk] *= ap_mask[msk]
                    obs.noise[msk] *= ap_mask[msk]

                    obs.bmask[msk] |= 1

                    if hasattr(obs, "mfrac"):
                        obs.mfrac[msk] = 1.0

                    if msk[0].size == obs.image.size:
                        obs.ignore_zero_weight = False

                    obs.weight[msk] = 0.0

                    if np.all(obs.weight == 0):
                        obs.ignore_zero_weight = False


@njit
def _build_square_apodization_mask(ap_rad, ap_mask):
    ap_range = get_ap_range()

    ny, nx = ap_mask.shape
    for y in range(min(ap_range + 1, ny)):
        for x in range(nx):
            ap_mask[y, x] *= _ap_kern_kern(y, ap_range, ap_rad)
            ap_mask[ny - 1 - y, x] *= _ap_kern_kern(y, ap_range, ap_rad)

    for y in range(ny):
        for x in range(min(ap_range + 1, nx)):
            ap_mask[y, x] *= _ap_kern_kern(x, ap_range, ap_rad)
            ap_mask[y, nx - 1 - x] *= _ap_kern_kern(x, ap_range, ap_rad)


@njit
def get_ap_range():
    """
    Get the range over which the the apodization kernel drops to zero

        int(6*AP_RAD + 0.5)

    Returns
    -------
    ap range as an integer
    """
    return int(6 * AP_RAD + 0.5)


def taper_from_distance(dist, width):
    """
    smooth taper as a function of distance into valid territory:
    0 at dist=0, 1 at dist >= width, using the same cumulative
    triweight kernel as the cell-edge apodization so star masks
    and cell edges share one smoothness class
    """
    y = (np.asarray(dist, dtype=float) - width) * (6.0 / width) + 3
    out = np.where(y > 3, 1.0, 0.0)
    w = (y >= -3) & (y <= 3)
    yy = y[w]
    out[w] = (
        -5 * yy ** 7 / 69984
        + 7 * yy ** 5 / 2592
        - 35 * yy ** 3 / 864
        + 35 * yy / 96
        + 1 / 2
    )
    return out


@njit
def _ap_kern_kern(x, m, h):
    # cumulative triweight kernel
    y = (x - m) / h + 3
    if y < -3:
        return 0
    elif y > 3:
        return 1
    else:
        val = (
            -5 * y ** 7 / 69984
            + 7 * y ** 5 / 2592
            - 35 * y ** 3 / 864
            + 35 * y / 96
            + 1 / 2
        )
        return val
