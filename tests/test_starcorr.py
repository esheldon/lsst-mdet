"""
the source counts around the stars in lsst-mdet-starcorr: the
Landy-Szalay star-source cross-correlation is -1 where sources are
missing from the usable area and flat at zero where the area itself
is masked (the source randoms then exclude it too)
"""
import numpy as np
import pytest
from scipy.spatial import cKDTree

pytest.importorskip('treecorr')

from lsst_mdet.cli.starcorr import measure_starcorr  # noqa: E402

HOLE = 1.0  # arcmin


def _sims():
    """stars, sources and randoms uniform in a 2 x 2 deg box on the
    equator, the sources removed within HOLE of every star"""
    rng = np.random.RandomState(9)

    def uniform(n):
        return rng.uniform(0, 2, n), rng.uniform(-1, 1, n)

    stars = np.zeros(300, dtype=[('ra', 'f8'), ('dec', 'f8'), ('gmag', 'f8')])
    stars['ra'], stars['dec'] = uniform(stars.size)
    stars['gmag'] = 17.0
    tree = cKDTree(np.column_stack([stars['ra'], stars['dec']]))

    def near_stars(ra, dec):
        return tree.query(np.column_stack([ra, dec]))[0] < HOLE / 60.0

    ra, dec = uniform(40000)
    keep = ~near_stars(ra, dec)
    gals = np.zeros(keep.sum(), dtype=[
        ('ra', 'f8'), ('dec', 'f8'), ('g1', 'f8'), ('g2', 'f8'), ('w', 'f8'),
    ])
    gals['ra'], gals['dec'] = ra[keep], dec[keep]
    gals['g1'], gals['g2'] = rng.normal(scale=0.2, size=(2, gals.size))
    gals['w'] = rng.uniform(0.5, 1.5, gals.size)
    rra, rdec = uniform(100000)
    return stars, gals, rra, rdec, near_stars(rra, rdec)


def _counts(rand_in_footprint):
    stars, gals, rra, rdec, _ = _sims()
    (lo, hi, data, cov, ncov), = measure_starcorr(
        stars=stars, gals=gals, resp=1.0, gmag_edges=[16, 18],
        theta_min=0.2, theta_max=6.0, nbins=6,
        rand_ra=rra, rand_dec=rdec, npatch=4,
        rand_in_footprint=rand_in_footprint(rra, rdec),
    )
    assert set(ncov) == {'nsrc', 'wsrc'}
    return data


def test_counts_missing_sources():
    # sources missing, the randoms cover the holes: a deficit
    stars, gals, rra, rdec, near = _sims()
    data = _counts(lambda ra, dec: np.ones(ra.size, dtype=bool))
    inner = data['theta'] < 0.7 * HOLE
    outer = data['theta'] > 2 * HOLE
    for key in ('nsrc', 'wsrc'):
        assert np.allclose(data[f'{key}_xi'][inner], -1, atol=0.05)
        assert np.all(np.abs(data[f'{key}_xi'][outer]) < 0.1)


def test_counts_masked_area():
    # the holes are masked: the source randoms exclude them, flat
    stars, gals, rra, rdec, near = _sims()
    data = _counts(lambda ra, dec: ~near)
    for key in ('nsrc', 'wsrc'):
        assert np.all(np.abs(data[f'{key}_xi']) < 0.1)
