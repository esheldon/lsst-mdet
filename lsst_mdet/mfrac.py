"""
masked-fraction measurement
"""
import numpy as np


MFRAC_FWHM = 1.2


def calculate_mfrac(mbobs, mfrac_weight):

    mfrac = mbobs[0][0].mfrac.copy()

    for iband, obslist in enumerate(mbobs):
        if iband == 0:
            mfrac = obslist[0].mfrac.copy()
        else:
            mfrac[:, :] = np.maximum(mfrac, obslist[0].mfrac)

    mfrac_obs = mbobs[0][0].copy()

    with mfrac_obs.writeable():
        mfrac_obs.image[:, :] = mfrac
        mfrac_obs.weight[:, :] = 1.0

    stats = mfrac_weight.get_weighted_sums(mfrac_obs, MFRAC_FWHM * 2)
    return stats["sums"][5] / stats["wsum"]


def smooth_mfrac_map(mbobs, fwhm=MFRAC_FWHM):
    """
    the band-maximum mfrac plane smoothed by the unit gaussian
    of the given fwhm, edge-normalized: sampling this map at a
    position equals the gaussian-weighted mean the per-stamp
    calculate_mfrac computes, but works for every row (deblend
    branch and extra detections included) at one convolution
    per cell

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations; the mfrac planes are combined by max
    fwhm: float, optional
        The gaussian weight fwhm in arcsec

    Returns
    -------
    2d array, same shape as the images
    """
    from scipy.ndimage import gaussian_filter
    from ngmix.moments import fwhm_to_sigma

    mfrac = mbobs[0][0].mfrac.copy()
    for obslist in mbobs[1:]:
        mfrac[:, :] = np.maximum(mfrac, obslist[0].mfrac)

    scale = mbobs[0][0].jacobian.scale
    sigma_px = fwhm_to_sigma(fwhm) / scale
    # truncate to match the per-stamp maxrad of 2 * fwhm
    trunc = 2.0 * fwhm / fwhm_to_sigma(fwhm)
    num = gaussian_filter(
        mfrac, sigma_px, mode='constant', cval=0.0,
        truncate=trunc,
    )
    den = gaussian_filter(
        np.ones_like(mfrac), sigma_px, mode='constant',
        cval=0.0, truncate=trunc,
    )
    return num / den


def sample_mfrac(mfrac_map, x, y):
    """
    bilinear sample of the smoothed mfrac map at (x, y) pixel
    positions, clipped to the map bounds
    """
    ny, nx = mfrac_map.shape
    x = np.clip(np.asarray(x, dtype=float), 0, nx - 1)
    y = np.clip(np.asarray(y, dtype=float), 0, ny - 1)
    x0 = np.clip(np.floor(x).astype(int), 0, nx - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, ny - 2)
    fx = x - x0
    fy = y - y0
    return (
        mfrac_map[y0, x0] * (1 - fx) * (1 - fy)
        + mfrac_map[y0, x0 + 1] * fx * (1 - fy)
        + mfrac_map[y0 + 1, x0] * (1 - fx) * fy
        + mfrac_map[y0 + 1, x0 + 1] * fx * fy
    )
