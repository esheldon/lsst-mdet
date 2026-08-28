"""
the deblend_flags column stores the kdeblend flag word as is: the
column initializes to NO_ATTEMPT like every flags column, and the
kdeblend bits are defined in the same ngmix convention, clear of it
"""
import numpy as np
import pytest

from lsst_mdet.defaults import NO_ATTEMPT
from lsst_mdet.structs import get_struct, get_kdeblend_struct


@pytest.mark.parametrize('maker', [get_struct, get_kdeblend_struct])
def test_column_initializes_to_no_attempt(maker):
    st = maker(['r', 'i'], 3)
    assert np.all(st['deblend_flags'] == NO_ATTEMPT)
    assert np.all(st['flags'] == NO_ATTEMPT)


def test_kdeblend_convention():
    kdeblend = pytest.importorskip('kdeblend')
    assert kdeblend.NO_ATTEMPT is NO_ATTEMPT
    for bit in (
        kdeblend.DEBLENDED_AS_PSF, kdeblend.RESTARTED,
        kdeblend.EXTERNALS_SUBTRACTED, kdeblend.WEIGHT_BOUNDED,
    ):
        assert bit & NO_ATTEMPT == 0
