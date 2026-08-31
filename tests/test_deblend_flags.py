"""
the deblend_flags column stores the kdeblend flag word as is, and
g_flags stores the kdeblend e_flags word (zero iff the shape and
its errors are usable), with stars kept at NO_ATTEMPT.  The
columns initialize to NO_ATTEMPT like every flags column, and the
kdeblend bits are defined in the same ngmix convention, clear of it
"""
import numpy as np
import pytest

from lsst_mdet.defaults import NO_ATTEMPT
from lsst_mdet.structs import get_struct


def test_column_initializes_to_no_attempt():
    st = get_struct(['r', 'i'], 3)
    assert np.all(st['deblend_flags'] == NO_ATTEMPT)
    assert np.all(st['flags'] == NO_ATTEMPT)
    assert np.all(st['g_flags'] == NO_ATTEMPT)


def test_kdeblend_convention():
    kdeblend = pytest.importorskip('kdeblend')
    assert kdeblend.NO_ATTEMPT is NO_ATTEMPT
    for bit in (
        kdeblend.DEBLENDED_AS_PSF, kdeblend.RESTARTED,
        kdeblend.EXTERNALS_SUBTRACTED, kdeblend.WEIGHT_BOUNDED,
    ):
        assert bit & NO_ATTEMPT == 0


def make_obj_res(otype, e_flags, nband=2):
    """a minimal kdeblend per-object result for the packer"""
    usable = otype != 'star' and e_flags == 0
    return {
        'type': otype,
        'deblend_flags': 0,
        'e_flags': e_flags,
        'e1': 0.1 if usable else np.nan,
        'e2': -0.05 if usable else np.nan,
        'e1_err': 0.02 if usable else np.nan,
        'e2_err': 0.02 if usable else np.nan,
        'T': 0.0 if otype == 'star' else 0.4,
        'T_err': 0.05,
        'flux': np.full(nband, 100.0),
        'flux_err': np.full(nband, 3.0),
        'flux_cov': np.diag(np.full(nband, 9.0)),
        's2n': 30.0,
        'cen': np.array([0.0, 0.0]),
    }


@pytest.mark.parametrize('otype,e_flags,expected', [
    ('exp', 0, 0),
    ('star', 0, NO_ATTEMPT),
])
def test_g_flags_from_e_flags(otype, e_flags, expected):
    import ngmix

    from lsst_mdet.deblend import pack_deblend_object

    bands = ['r', 'i']
    jacobian = ngmix.DiagonalJacobian(scale=0.2, row=0, col=0)

    st = get_struct(bands, 1)
    pack_deblend_object(
        st=st[0],
        obj_res=make_obj_res(otype, e_flags),
        bands=bands,
        jacobian=jacobian,
    )
    assert st['g_flags'][0] == expected

    # a degenerate galaxy shape carries the e_flags bits as is
    st = get_struct(bands, 1)
    pack_deblend_object(
        st=st[0],
        obj_res=make_obj_res('exp', ngmix.flags.NONPOS_SHAPE_VAR),
        bands=bands,
        jacobian=jacobian,
    )
    assert st['g_flags'][0] == ngmix.flags.NONPOS_SHAPE_VAR
    assert st['g_flags'][0] & NO_ATTEMPT == 0
