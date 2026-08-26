"""
the adaptive template faint limit and the mask-only fallback:
dense fields keep the preferred bright window, sparse fields
extend to the cap, and a patch too barren for any template
masks without subtracting instead of crashing
"""
import numpy as np
import pytest

import lsst_mdet.starsub as ss
from lsst_mdet.starsub import (
    AUR_BREAK,
    AUR_MIN_STARS,
    AUR_SLOPE,
    HALO_SLOPE,
    TMPL_GMAX,
    TMPL_GMAX_CAP,
    TMPL_HALF,
    TMPL_MIN_CAND,
    fit_aureole,
    fit_halo_slope,
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


def make_profile(slope=-4.0, amp=3.0e-3, ped=5.0e-7, n=60):
    r = np.arange(n).astype(float)
    prof = amp * np.maximum(r, 1.0) ** slope + ped
    return prof


def test_fit_halo_slope_joint():
    prof = make_profile()
    slope, ln_a, ped = fit_halo_slope(prof)
    assert slope == pytest.approx(-4.0, abs=0.05)
    assert np.exp(ln_a) == pytest.approx(3.0e-3, rel=0.1)
    assert ped == pytest.approx(5.0e-7, rel=0.2)


def test_fit_halo_slope_pedestal_robust():
    # a large pedestal must not bias the slope: the failure
    # mode of the legacy log-log fit
    prof = make_profile(ped=5.0e-6)
    slope, ln_a, ped = fit_halo_slope(prof)
    assert slope == pytest.approx(-4.0, abs=0.1)
    assert ped == pytest.approx(5.0e-6, rel=0.1)


def test_fit_halo_slope_guard():
    # garbage profile: falls back to HALO_SLOPE
    prof = np.full(60, 1.0e-6)
    slope, ln_a, ped = fit_halo_slope(prof)
    assert slope == HALO_SLOPE


def aureole_cloud(slope, ln_a, s_aur, b, n=11):
    rmid = np.logspace(np.log10(45), np.log10(245), n)
    med = (
        np.exp(ln_a) * rmid ** slope + b * rmid ** s_aur
    )
    count = np.full(n, 12)
    return rmid, med, count


def test_fit_aureole_tier1():
    slope, ln_a = -4.0, np.log(3.0e-3)
    b_true = 1.0e-5
    rmid, med, count = aureole_cloud(slope, ln_a, -2.3, b_true)
    s_aur, b, tier = fit_aureole(
        rmid, med, count, nstars=20, slope=slope, ln_a=ln_a,
    )
    assert tier == 1
    assert s_aur == pytest.approx(-2.3, abs=0.05)
    assert b == pytest.approx(b_true, rel=0.05)


def test_fit_aureole_tier2():
    # too few stars for a slope: fixed AUR_SLOPE, amplitude
    # recovered when the data follow it
    slope, ln_a = -4.0, np.log(3.0e-3)
    # within the continuity amplitude guard
    b_true = 1.0e-6
    rmid, med, count = aureole_cloud(
        slope, ln_a, AUR_SLOPE, b_true,
    )
    s_aur, b, tier = fit_aureole(
        rmid, med, count,
        nstars=AUR_MIN_STARS - 1, slope=slope, ln_a=ln_a,
    )
    assert tier == 2
    assert s_aur == AUR_SLOPE
    assert b == pytest.approx(b_true, rel=0.05)


def test_fit_aureole_tier3():
    # nothing measurable: continuity with the inner law at
    # AUR_BREAK
    slope, ln_a = -4.0, np.log(3.0e-3)
    rmid = np.logspace(np.log10(45), np.log10(245), 11)
    med = np.full(11, np.nan)
    count = np.zeros(11, dtype=int)
    s_aur, b, tier = fit_aureole(
        rmid, med, count, nstars=0, slope=slope, ln_a=ln_a,
    )
    assert tier == 3
    assert s_aur == AUR_SLOPE
    inner_at_break = np.exp(ln_a) * AUR_BREAK ** slope
    assert b * AUR_BREAK ** s_aur == pytest.approx(
        inner_at_break, rel=1e-10,
    )


def test_fit_aureole_amp_guard_clips():
    # a wild fitted amplitude is clipped to the guard bound,
    # never replaced by the continuity prior: the prior
    # over-subtracts on a base whose wings were partly
    # absorbed (and under-subtracts in the opposite case)
    slope, ln_a = -4.0, np.log(3.0e-3)
    b_cont = np.exp(ln_a) * AUR_BREAK ** (slope - AUR_SLOPE)
    for factor, bound in [(1000.0, 10.0), (1e-3, 0.1)]:
        rmid, med, count = aureole_cloud(
            slope, ln_a, AUR_SLOPE, factor * b_cont,
        )
        s_aur, b, tier = fit_aureole(
            rmid, med, count, nstars=AUR_MIN_STARS - 1,
            slope=slope, ln_a=ln_a,
        )
        assert tier == 2
        assert b == pytest.approx(bound * b_cont, rel=1e-6)


def test_extend_template_two_scale():
    slope, ln_a = -4.0, np.log(3.0e-3)
    s_aur, b = -2.0, 1.0e-5
    tmpl = np.zeros((2 * TMPL_HALF + 1, 2 * TMPL_HALF + 1))
    big = ss.extend_template_halo(tmpl, slope, ln_a, s_aur, b)
    half = ss.TMPL_OUT_HALF
    for r in (60, 120, 200):
        expected = (
            np.exp(ln_a) * float(r) ** slope
            + b * float(r) ** s_aur
        )
        assert big[half, half + r] == pytest.approx(
            expected, rel=1e-6,
        )


class FakeRendered:
    def __init__(self, arr):
        class Q:
            value = arr
        self.quantity = Q()


class FakeBackground:
    def __init__(self, arr):
        outer = self

        class F:
            def render(self, bbox, dtype=None):
                return FakeRendered(outer.arr)
        self.arr = arr
        self.field = F()


class FakePlane:
    def __init__(self, arr):
        self.array = arr


class FakeCoadd:
    def __init__(self, image, model=None):
        self.image = FakePlane(image)
        self.bbox = None
        if model is not None:
            self.backgrounds = {'object': FakeBackground(model)}


def test_restore_object_background():
    dim = 200
    image = np.zeros((dim, dim))
    model = np.full((dim, dim), 2.0)
    # bright star zone in one corner: distances from it
    gy, gx = np.mgrid[0:dim, 0:dim]
    dbright = np.hypot(gy - 30.0, gx - 30.0)

    coadd = FakeCoadd(image, model=model)
    ss.restore_object_background(coadd, dbright)

    # full restoration inside RESTORE_RAD, zero beyond the
    # taper
    assert image[30, 30] == pytest.approx(2.0)
    far = dbright > ss.RESTORE_RAD + ss.RESTORE_TAPER
    if far.any():
        assert np.all(image[far] == 0)


def test_restore_no_model_is_noop():
    image = np.ones((50, 50))
    coadd = FakeCoadd(image, model=None)
    dbright = np.zeros((50, 50))
    ss.restore_object_background(coadd, dbright)
    assert np.all(image == 1.0)


def test_template_out_half():
    from lsst_mdet.starsub import (
        TMPL_EXT_FACTOR, TMPL_OUT_HALF, TMPL_OUT_MAX,
        circle_radius, template_out_half,
    )
    # faint stars keep the floor
    assert template_out_half(17.0) == TMPL_OUT_HALF
    # bright stars scale with the mask radius
    g = 10.0
    expect = int(TMPL_EXT_FACTOR * circle_radius(g))
    assert template_out_half(g) == expect
    assert expect > TMPL_OUT_HALF
    # the very brightest are capped
    assert template_out_half(2.0) == TMPL_OUT_MAX


def test_extend_template_halo_sized():
    slope, ln_a = -4.0, np.log(3.0e-3)
    s_aur, b = -2.0, 1.0e-5
    tmpl = np.zeros((2 * TMPL_HALF + 1, 2 * TMPL_HALF + 1))
    out_half = 500
    big = ss.extend_template_halo(
        tmpl, slope, ln_a, s_aur, b, out_half=out_half,
    )
    assert big.shape == (2 * out_half + 1, 2 * out_half + 1)
    # the halo continues beyond the old fixed edge
    r = 400
    expected = (
        np.exp(ln_a) * float(r) ** slope
        + b * float(r) ** s_aur
    )
    assert big[out_half, out_half + r] == pytest.approx(
        expected, rel=1e-6,
    )
    # tapered to zero at the new edge
    assert big[out_half, -1] == 0
