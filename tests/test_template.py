"""
the adaptive template faint limit and the mask-only fallback:
dense fields keep the preferred bright window, sparse fields
extend to the cap, and a patch too barren for any template
masks without subtracting instead of crashing
"""
import numpy as np

import lsst_mdet.starsub as ss
from lsst_mdet.starsub import (
    TMPL_GMAX,
    TMPL_GMAX_CAP,
    TMPL_HALF,
    TMPL_MIN_CAND,
    select_template_stars,
)

DIM = 600
SHAPE = (DIM, DIM)


def make_gaia(gmags, rng):
    """
    synthetic census at interior positions, well away from the
    stamp edge cut
    """
    gmags = np.asarray(gmags, dtype='f8')
    n = gmags.size
    gaia = np.zeros(n, dtype=[
        ('ra', 'f8'), ('dec', 'f8'),
        ('pmra', 'f8'), ('pmdec', 'f8'),
        ('phot_g_mean_mag', 'f8'), ('ruwe', 'f8'),
    ])
    gaia['phot_g_mean_mag'] = gmags
    gaia['ruwe'] = 1.0
    lo = TMPL_HALF + 10
    hi = DIM - TMPL_HALF - 10
    x = rng.uniform(lo, hi, size=n)
    y = rng.uniform(lo, hi, size=n)
    return gaia, x, y


def test_dense_keeps_bright_window():
    rng = np.random.RandomState(3)
    # plenty of bright candidates plus faint ones on offer
    gmags = np.concatenate([
        np.linspace(15.6, 17.4, TMPL_MIN_CAND + 10),
        np.full(20, 18.5),
    ])
    gaia, x, y = make_gaia(gmags, rng)
    sel = select_template_stars(gaia, x, y, SHAPE)
    assert sel.size >= TMPL_MIN_CAND
    assert np.all(gaia['phot_g_mean_mag'][sel] < TMPL_GMAX)


def test_sparse_extends_faint_limit():
    rng = np.random.RandomState(5)
    # 5 bright candidates, the rest fainter: must extend
    gmags = np.concatenate([
        np.linspace(15.6, 17.4, 5),
        np.linspace(18.6, 18.9, 25),
    ])
    gaia, x, y = make_gaia(gmags, rng)
    sel = select_template_stars(gaia, x, y, SHAPE)
    assert sel.size >= TMPL_MIN_CAND
    assert np.any(gaia['phot_g_mean_mag'][sel] > TMPL_GMAX)
    # brightest first: the bright candidates all selected
    assert np.all(np.isin(np.arange(5), sel))


def test_extension_respects_cap():
    rng = np.random.RandomState(7)
    # only stars beyond the cap: never selected, even though
    # the count stays below TMPL_MIN_CAND
    gmags = np.concatenate([
        np.linspace(15.6, 17.4, 3),
        np.full(30, TMPL_GMAX_CAP + 0.5),
    ])
    gaia, x, y = make_gaia(gmags, rng)
    sel = select_template_stars(gaia, x, y, SHAPE)
    assert sel.size == 3
    assert np.all(gaia['phot_g_mean_mag'][sel] < TMPL_GMAX)


def test_edge_cut_applies_to_extended_stars():
    rng = np.random.RandomState(9)
    gmags = np.linspace(18.6, 18.9, 25)
    gaia, x, y = make_gaia(gmags, rng)
    # push one candidate onto the edge: it must drop out
    x[0] = 1.0
    sel = select_template_stars(gaia, x, y, SHAPE)
    assert 0 not in sel
    assert sel.size == 24


def test_mask_only_fallback():
    rng = np.random.RandomState(11)
    # a barren field: 3 census stars, no possible template
    gaia, x, y = make_gaia([16.0, 16.5, 17.0], rng)

    image = rng.normal(size=SHAPE)
    var = np.ones(SHAPE)
    mask0 = np.zeros(SHAPE, dtype='i4')

    stars = ss.select_stars(gaia, x, y, mask0)
    assert stars.size == 3
    starmask, comps = ss.build_star_mask(stars, mask0)
    assert starmask.any()

    im0 = image.copy()
    slist = ss.subtract_stars(
        image, var, mask0, gaia, x, y, stars, comps,
    )
    # no template: nothing subtracted, nothing crashed
    assert slist == []
    assert np.array_equal(image, im0)

    # the census table still comes out, with zero amplitudes,
    # so the mask and footprint machinery are unaffected
    star_table = ss.make_star_table(stars, slist)
    assert star_table.size == 3
    assert np.all(star_table['A'] == 0)
