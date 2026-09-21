"""
sky background determination
"""
import numpy as np
from .defaults import DM_OUT


# sep's limit on the active object pixels in an extraction: a galaxy
# filling a good part of the patch overflows the default (NGC 1527 in
# 02395-00072, D25 4.5 arcmin: "internal pixel buffer full").  On that
# error the extraction is retried with the stack grown by
# PIXSTACK_GROWTH, up to PIXSTACK_MAX.  The stack is capacity only:
# an extraction that fits gives the same result at any size
PIXSTACK_GROWTH = 4
PIXSTACK_MAX = int(3.2e7)


def extract_with_retry(*args, **kwargs):
    """
    sep.extract, retried with a larger pixel stack when it overflows.

    The first attempt runs with the stack as it is set (sep's default
    unless the caller changed it); on "internal pixel buffer full" the
    stack grows by PIXSTACK_GROWTH per retry up to PIXSTACK_MAX, past
    which the error is raised.  The previous setting is restored.

    Parameters
    ----------
    *args, **kwargs:
        Passed to sep.extract

    Returns
    -------
    What sep.extract returns
    """
    import sep

    old_stack = sep.get_extract_pixstack()
    stack = old_stack
    try:
        while True:
            try:
                return sep.extract(*args, **kwargs)
            except Exception as error:
                if ('pixel buffer full' not in str(error)
                        or stack >= PIXSTACK_MAX):
                    raise
                stack = min(stack * PIXSTACK_GROWTH, PIXSTACK_MAX)
                print(f'    background: sep pixel stack full; retrying '
                      f'with {stack}')
                sep.set_extract_pixstack(stack)
    finally:
        sep.set_extract_pixstack(old_stack)


def redo_background(deep_coadd, starmask=None, subtract=True):
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

    With subtract=False the background is measured but not
    subtracted: the noise calibration and the sky-variance map
    are still made.  For an image whose sky is already fit,
    e.g. by the lsst_starsub joint fit of the stars and the sky,
    whose deep source mask this 64 px background would undo
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

    objects, seg = extract_with_retry(
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

    if subtract:
        image[:, :] -= bkg.back()
    else:
        print(f'    band: {deep_coadd.band} background measured, '
              f'not subtracted (globalback {bkg.globalback:+.3f})')

    w = np.where(new_good)
    medvar = np.median(var[w])

    noise_factor = bkg.globalrms / np.sqrt(medvar)
    # noise_factor = bkg.globalrms / noise.std()
    print(f'    band: {deep_coadd.band} noise_factor: {noise_factor:g}')

    noise[:, :] *= noise_factor
    var[:, :] *= noise_factor ** 2

    return bkg.rms().astype('f4') ** 2
