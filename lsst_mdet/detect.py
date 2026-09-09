"""
sep detection: kernel, config, noise calibration
"""

# the detection settings, recorded in the output meta table (see
# provenance.py): the sep threshold in units of the kernel-scale
# noise, the gaussian detection kernel fwhm in arcsec at the pixel
# scale, and the sep deblending and area parameters.  deblend_cont
# 1e-5 recovers peak-resolved objects in crowded regions that the
# DES default 1e-3 merges; measured to add no false detections at
# any tested density (see docs/detection-color-filter)
DETECT_SETTINGS = dict(
    thresh=0.8,
    kernel_fwhm=0.8,
    pixel_scale=0.2,
    deblend_cont=1.0e-5,
    deblend_nthresh=64,
    minarea=4,
    filter_type='conv',
)


def run_sep(obs):
    """
    Run sep (python wrapper for sextractor) on the input ngmix Observation

    The noise for the relative threshold is taken from the weight map.
    For metacal'd observations the weight map is calibrated in
    mdet.do_metacal so that 1/sqrt(median(weight)) is the measured noise
    on the detection-kernel scale, the correct calibration when the
    noise is correlated

    Parameters
    ----------
    obs: ngmix.Observation
        The observation holding image and weight map needed for detection

    Returns
    -------
    cat, seg

    cat: array with fields
        The object data.
    seg: array
        The segmentation map, with values matching the catalog
        'number' field
    """
    import numpy as np
    import sxdes

    # median over valid pixels only: with star masking a cell
    # can be 20-50 percent zero weight and still pass the
    # good-frac gate, and the all-pixel median is then zero,
    # sending the detection threshold to infinity
    medwt = np.median(obs.weight[obs.weight > 0])
    noise = 1 / np.sqrt(medwt)

    with obs.writeable():
        objs, seg = sxdes.run_sep(
            image=obs.image,
            noise=noise,
            config=get_sx_config(),
            mask=obs.bmask,
            thresh=DETECT_SETTINGS['thresh'],
        )
    return objs, seg


def get_detect_noise(noise, kernel, weight=None):
    """
    Get the noise value on the detection-kernel scale, used by
    mdet.do_metacal to calibrate the weight maps for correlated noise.

    sep with filter_type 'conv' normalizes the kernel to unit sum and
    thresholds the convolved image at thresh * err.  The std of the
    convolved noise field, divided by sqrt(sum(khat^2)), gives the err
    for which thresh keeps its white-noise meaning; for uncorrelated
    noise this reduces to the pixel sigma

    Parameters
    ----------
    noise: array
        A realization of the noise in the image, e.g. the metacal'd
        noise field for a metacal'd image
    kernel: array
        The detection kernel
    weight: array, optional
        The weight map.  When given, the std is measured only
        over valid pixels eroded by the kernel width: with the
        star-masked (apodized to zero) noise plane, a plain
        std over all pixels underestimates sigma by about
        sqrt(1 - fmask) and inflates the weights and S/N

    Returns
    -------
    noise value for sxdes.run_sep
    """
    import numpy as np
    from scipy.ndimage import convolve, binary_erosion

    khat = kernel / kernel.sum()
    conv = convolve(noise, khat, mode='reflect')
    if weight is not None:
        valid = binary_erosion(weight > 0, iterations=4)
        if valid.sum() < 1000:
            valid = weight > 0
        return conv[valid].std() / np.sqrt((khat**2).sum())
    return conv.std() / np.sqrt((khat**2).sum())


def get_sx_config():
    """
    The configuration for sextractor.  This is currently hard wired
    to use a detection kernel of 0.8 arcseconds assuming pixel scale
    of 0.2 (sensible for LSST)
    """
    s = DETECT_SETTINGS
    return {
        'deblend_cont': s['deblend_cont'],
        'deblend_nthresh': s['deblend_nthresh'],
        'minarea': s['minarea'],
        'filter_type': s['filter_type'],
        'filter_kernel': make_kernel(),
    }


def make_kernel():
    """
    Make the detection kernel, a 7x7 gaussian of DETECT_SETTINGS kernel_fwhm.
    """
    import ngmix

    # 7x7 convolution mask of a gaussian PSF with FWHM = 4 pixels
    # for the default 0.8 arcseconds at 0.2 arcseconds per pixel
    s = DETECT_SETTINGS
    fwhm = s['kernel_fwhm'] / s['pixel_scale']  # pixels
    T = ngmix.moments.fwhm_to_T(fwhm)
    kernel_gm = ngmix.GMixModel(
        pars=[0.0, 0.0, 0.0, 0.0, T, 1.0],
        model='gauss',
    )
    return kernel_gm.make_image([7, 7])
