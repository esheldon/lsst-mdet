"""
sky background determination
"""
import numpy as np
from .defaults import DM_OUT


def redo_background(deep_coadd, starmask=None):
    """
    redo the background determination on the image, in place,
    and calibrate the noise realization to the measured sky
    rms.

    Returns the sky-variance map: the squared per-box rms of
    the final object-and-star-masked fit.  Measured from the
    image fluctuations, it carries the depth structure but no
    object poisson term, and is the plane the pixel weights
    should be built from (the variance plane includes the
    objects' poisson noise, making weights signal-dependent)
    """
    import sep

    image = deep_coadd.image.array
    var = deep_coadd.variance.array
    mask = deep_coadd.mask.array[:, :, 0]
    noise = deep_coadd.noise_realizations[0].array

    good = (
        np.isfinite(var)
        & np.isfinite(noise)
        & (mask & DM_OUT == 0)
    )
    if starmask is not None:
        # star vicinities are subtracted/apodized; keep them
        # out of the background boxes and noise calibration
        good &= ~starmask
    bad = ~good

    bkg = sep.Background(image, mask=bad)
    # image -= bkg.back()

    objects, seg = sep.extract(
        image - bkg.back(),
        1.5,
        mask=bad,
        err=bkg.globalrms,
        segmentation_map=True,
    )
    new_good = good & (seg == 0)
    bad = ~new_good
    bkg = sep.Background(image, mask=bad)

    for i in range(0):
        objects, seg = sep.extract(
            image,
            1.0,
            mask=bad,
            err=bkg.globalrms,
            segmentation_map=True,
        )

        new_good = good & (seg == 0)
        bad = ~new_good
        bkg = sep.Background(image, mask=bad)

        image -= bkg.back()

    image[:, :] -= bkg.back()

    w = np.where(new_good)
    medvar = np.median(var[w])

    noise_factor = bkg.globalrms / np.sqrt(medvar)
    # noise_factor = bkg.globalrms / noise.std()
    print(f'    band: {deep_coadd.band} noise_factor: {noise_factor:g}')

    noise[:, :] *= noise_factor
    var[:, :] *= noise_factor ** 2

    return bkg.rms().astype('f4') ** 2
