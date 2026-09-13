"""
the mask of the joint fit's diffuse regions: the union over the
bands, grown by the margin
"""
import numpy as np

from lsst_mdet.cells import diffuse_mask


def test_diffuse_mask_union_and_margin():
    shape = (200, 300)
    r = np.zeros(shape, dtype=bool)
    r[50:60, 50:60] = True
    i = np.zeros(shape, dtype=bool)
    i[150:155, 250:255] = True
    fits = {
        'r': {'diffuse': r},
        'i': {'diffuse': i},
        'z': {'diffuse': np.zeros(shape, dtype=bool)},
    }
    m = diffuse_mask(fits, 20)
    assert m[r].all() and m[i].all()
    # grown by 20 px and no further
    assert m[55, 79] and not m[55, 80]
    assert m[30, 55] and not m[29, 55]
    assert np.array_equal(diffuse_mask(fits, 0), r | i)


def test_diffuse_mask_none():
    empty = {'r': {'diffuse': np.zeros((50, 50), dtype=bool)}}
    assert diffuse_mask(empty, 20) is None
    # fit dicts from an lsst_starsub without the diffuse region
    assert diffuse_mask({'r': {}}, 20) is None
