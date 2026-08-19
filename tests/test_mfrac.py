"""
the smoothed-map mfrac must reproduce the per-stamp
gaussian-weighted mean of the retired ml-branch implementation
"""
import numpy as np
import ngmix
import pytest

from lsst_mdet.mfrac import (
    MFRAC_FWHM, calculate_mfrac, sample_mfrac, smooth_mfrac_map,
)
from lsst_mdet.maxlike import extract_stamp_mbobs


def make_mbobs(rng, dim=250, scale=0.2):
    from scipy.ndimage import gaussian_filter

    mbobs = ngmix.MultiBandObsList()
    for band in range(3):
        # smooth random mfrac pattern in [0, 1] plus a fully
        # masked block
        mfrac = gaussian_filter(
            rng.uniform(size=(dim, dim)), 8.0,
        )
        mfrac -= mfrac.min()
        mfrac /= mfrac.max()
        mfrac[40:90, 150:220] = 1.0

        im = rng.normal(size=(dim, dim))
        jac = ngmix.DiagonalJacobian(
            scale=scale, row=(dim - 1) / 2, col=(dim - 1) / 2,
        )
        pdim = 35
        pc = (pdim - 1) / 2
        gy, gx = np.mgrid[0:pdim, 0:pdim]
        psf_im = np.exp(
            -((gy - pc) ** 2 + (gx - pc) ** 2) / (2 * 2.0 ** 2)
        )
        psf_obs = ngmix.Observation(
            psf_im,
            weight=psf_im * 0 + 1e12,
            jacobian=ngmix.DiagonalJacobian(
                scale=scale, row=pc, col=pc,
            ),
        )
        obs = ngmix.Observation(
            im,
            weight=np.ones((dim, dim)),
            mfrac=mfrac,
            noise=rng.normal(size=(dim, dim)),
            bmask=np.zeros((dim, dim), dtype='i4'),
            jacobian=jac,
            psf=psf_obs,
        )
        ol = ngmix.ObsList()
        ol.append(obs)
        mbobs.append(ol)
    return mbobs


def test_map_matches_per_stamp():
    rng = np.random.RandomState(7)
    mbobs = make_mbobs(rng)

    mfrac_weight = ngmix.GMixModel(
        [0, 0, 0, 0, ngmix.moments.fwhm_to_T(MFRAC_FWHM), 1],
        'gauss',
    )
    mmap = smooth_mfrac_map(mbobs)

    # interior positions away from the cell edge
    xs = rng.uniform(40, 210, size=25)
    ys = rng.uniform(40, 210, size=25)
    icat = np.zeros(1, dtype=[('x', 'f8'), ('y', 'f8')])

    stamp_vals = []
    for x, y in zip(xs, ys):
        icat['x'] = x
        icat['y'] = y
        stamp_mbobs = extract_stamp_mbobs(
            mbobs=mbobs, icat=icat[0],
        )
        stamp_vals.append(calculate_mfrac(
            mbobs=stamp_mbobs, mfrac_weight=mfrac_weight,
        ))
    stamp_vals = np.array(stamp_vals)
    map_vals = sample_mfrac(mmap, xs, ys)

    assert np.all(np.abs(map_vals - stamp_vals) < 2.0e-3)


def test_map_bounds():
    rng = np.random.RandomState(11)
    mbobs = make_mbobs(rng)
    mmap = smooth_mfrac_map(mbobs)
    assert np.all((mmap >= 0) & (mmap <= 1 + 1e-9))
    # fully masked block reads ~1 at its center
    assert sample_mfrac(mmap, 185.0, 65.0) == pytest.approx(
        1.0, abs=1e-3,
    )
