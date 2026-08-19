"""
masked-fraction measurement

TODO: extend to the deblend branch
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
