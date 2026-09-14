"""
the settings lsst_starsub keeps its own copies of must match
lsst_mdet's: the DM mask bits, and the detection settings of the
joint fit's source segmentation (lsst_mdet passes its own, the copy
is the default for lsst_starsub's own tools)
"""
import numpy as np
import pytest


def test_mask_bits_match():
    from lsst_mdet import defaults
    from lsst_starsub import maskbits

    for name in ('DM_NO_DATA', 'DM_SAT', 'DM_INTRP', 'DM_DETECTION_EDGE',
                 'DM_OUT'):
        assert getattr(maskbits, name) == getattr(defaults, name), name


def test_detect_settings_match():
    from lsst_mdet import detect
    from lsst_starsub import joint

    for name, value in joint.DETECT_SETTINGS.items():
        assert detect.DETECT_SETTINGS[name] == value, name


def test_detection_kernel_matches():
    pytest.importorskip('ngmix')
    from lsst_mdet import detect
    from lsst_starsub import joint

    assert np.array_equal(joint.make_kernel(), detect.make_kernel())
    assert np.array_equal(joint.make_kernel(detect.DETECT_SETTINGS),
                          detect.make_kernel())
