"""
the catalog struct initialization. flags set to the no-attempt
value, the bool columns False (a nan fill would cast to True, which
is what the catalogs through v0.3.3 carry in extra_det), the
measurements nan
"""
import numpy as np

from lsst_mdet.structs import get_struct, NO_ATTEMPT


def test_struct_init():
    st = get_struct(['r', 'i'], n=3)
    assert not st['extra_det'].any()
    assert not st['is_primary'].any()
    assert (st['flags'] == NO_ATTEMPT).all()
    assert (st['psf_flags_r'] == NO_ATTEMPT).all()
    assert np.isnan(st['flux_r']).all()
    assert np.isnan(st['xcell']).all()
    assert (st['group_size'] == 0).all()
    assert (st['obj_id'] == np.arange(3)).all()
