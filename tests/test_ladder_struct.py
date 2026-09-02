"""
the ladder catalog: the flux columns hold the tau-completed total,
the colors come from the gauss fluxes with their covariance, the
gauss fluxes and the color gradient are kept, and demoted stars
fall back to the psf flux; the struct only carries the ladder
columns when asked for them, and the meta table records the deblend
settings
"""
import numpy as np
import pytest

from lsst_mdet.defaults import NO_ATTEMPT
from lsst_mdet.structs import get_struct

BANDS = ['g', 'r', 'i']
COLOR_FAC = 2.5 / np.log(10)


def make_ladder_res(nband=3):
    """a minimal kdeblend ladder result for the packer"""
    gflux = np.array([50.0, 100.0, 150.0])[:nband]
    gcov = np.diag(np.full(nband, 4.0))
    gcov[0, 1] = gcov[1, 0] = 1.0
    return {
        'type': 'ladder',
        'deblend_flags': 0,
        'e_flags': 0,
        'e1': 0.1,
        'e2': -0.05,
        'e1_err': 0.02,
        'e2_err': 0.02,
        'T': 0.4,
        'T_err': 0.05,
        'flux': gflux.copy(),
        'flux_err': np.sqrt(np.diag(gcov)),
        'flux_cov': gcov.copy(),
        's2n': 40.0,
        'cen': np.array([0.0, 0.0]),
        'gauss_flux': gflux.copy(),
        'gauss_flux_err': np.sqrt(np.diag(gcov)),
        'gauss_flux_cov': gcov.copy(),
        'total_flux': gflux * 1.1,
        'total_flux_err': np.full(nband, 3.0),
        'fixed_flux': gflux * 0.8,
        'fixed_flux_err': np.full(nband, 2.5),
        'gradient': np.array([0.03, -0.02])[:nband - 1],
        'gradient_err': np.array([0.01, 0.015])[:nband - 1],
    }


def make_star_res(nband=3):
    """a demoted star in a ladder run: psf flux only"""
    flux = np.full(nband, 20.0)
    return {
        'type': 'star',
        'deblend_flags': 2,
        'e_flags': 0,
        'e1': np.nan,
        'e2': np.nan,
        'e1_err': np.nan,
        'e2_err': np.nan,
        'T': 0.0,
        'T_err': np.nan,
        'flux': flux,
        'flux_err': np.full(nband, 1.0),
        'flux_cov': np.diag(np.full(nband, 1.0)),
        's2n': 20.0,
        'cen': np.array([0.0, 0.0]),
    }


def test_struct_columns_by_model():
    st = get_struct(BANDS, 2, model='ladder')
    for band in BANDS:
        assert f'gauss_flux_{band}' in st.dtype.names
        assert f'gauss_flux_err_{band}' in st.dtype.names
    assert 'gradient_gmr' in st.dtype.names
    assert 'gradient_rmi_err' in st.dtype.names
    assert 'fwhm_smooth' in st.dtype.names
    assert np.all(np.isnan(st['fwhm_smooth']))

    st = get_struct(BANDS, 2, model='exp')
    assert 'gauss_flux_g' not in st.dtype.names
    assert 'gradient_gmr' not in st.dtype.names
    assert 'fwhm_smooth' in st.dtype.names


def test_pack_ladder_row():
    import ngmix
    from lsst_mdet.deblend import pack_deblend_object

    jacobian = ngmix.DiagonalJacobian(scale=0.2, row=0, col=0)
    res = make_ladder_res()
    st = get_struct(BANDS, 1, model='ladder')
    pack_deblend_object(st=st[0], obj_res=res, bands=BANDS,
                        jacobian=jacobian)
    row = st[0]

    assert row['g_flags'] == 0
    for i, band in enumerate(BANDS):
        assert row[f'flux_{band}'] == pytest.approx(res['total_flux'][i])
        assert row[f'flux_err_{band}'] == pytest.approx(
            res['total_flux_err'][i]
        )
        assert row[f'gauss_flux_{band}'] == pytest.approx(
            res['gauss_flux'][i]
        )
        assert row[f'gauss_flux_err_{band}'] == pytest.approx(
            res['gauss_flux_err'][i]
        )

    # colors from the gauss fluxes, with the covariance-aware error
    g = res['gauss_flux']
    c = res['gauss_flux_cov']
    assert row['gmr'] == pytest.approx(-2.5 * np.log10(g[0] / g[1]))
    var = COLOR_FAC ** 2 * (
        c[0, 0] / g[0] ** 2 + c[1, 1] / g[1] ** 2
        - 2 * c[0, 1] / (g[0] * g[1])
    )
    assert row['gmr_err'] == pytest.approx(np.sqrt(var))

    assert row['gradient_gmr'] == pytest.approx(res['gradient'][0])
    assert row['gradient_rmi_err'] == pytest.approx(res['gradient_err'][1])
    assert row['s2n'] == pytest.approx(res['s2n'])


def test_pack_star_in_ladder_run():
    import ngmix
    from lsst_mdet.deblend import pack_deblend_object

    jacobian = ngmix.DiagonalJacobian(scale=0.2, row=0, col=0)
    res = make_star_res()
    st = get_struct(BANDS, 1, model='ladder')
    pack_deblend_object(st=st[0], obj_res=res, bands=BANDS,
                        jacobian=jacobian)
    row = st[0]

    # the demotion is the DEBLENDED_AS_PSF bit of deblend_flags
    assert row['deblend_flags'] == 2
    assert row['g_flags'] == NO_ATTEMPT
    assert row['flux_i'] == pytest.approx(res['flux'][2])
    assert row['gmr'] == pytest.approx(0.0)
    assert np.isnan(row['gauss_flux_i'])
    assert np.isnan(row['gradient_gmr'])


def test_meta_records_provenance(tmp_path):
    """
    the meta table carries the run options, the stage settings from
    the module constants, the package versions and the run identity,
    as plain columns
    """
    import rustfits
    from lsst_mdet import defaults, detect, metacal, starsub
    from lsst_mdet.deblend import DEBLEND_SETTINGS
    from lsst_mdet.io import write_output
    from lsst_mdet.structs import get_cell_meta

    fname = str(tmp_path / 'out.fits')
    st = get_struct(BANDS, 1, model='ladder')
    write_output(
        fname=fname, st=st, cell_meta=get_cell_meta(len(BANDS)),
        tract=1, patch=2, model='ladder', seed=3, with_mdet=True,
        redo_bg=False, starsub=False, deblend=True, s2_detect=False,
        run_options=dict(
            repo='dp2', collections=['a', 'b'], patch_dir=None,
            gaia_file=None, gsub=19.0, apod_stars=True,
            cells=[(10, 10), (11, 12)],
        ),
    )
    meta = rustfits.read(fname, ext='meta')
    assert meta.size == 1
    m = meta[0]

    assert m['model'] == 'ladder'
    assert m['tract'] == 1 and m['patch'] == 2 and m['seed'] == 3
    assert m['with_mdet'] and m['deblend'] and not m['starsub']
    assert m['repo'] == 'dp2'
    assert m['collections'] == 'a,b'
    assert m['patch_dir'] == ''
    assert m['gsub'] == 19.0
    assert m['apod_stars']
    assert m['cells'] == '(10, 10),(11, 12)'

    for name, value in DEBLEND_SETTINGS.items():
        assert m[f'deblend_{name}'] == value
    for name, value in detect.DETECT_SETTINGS.items():
        assert m[f'detect_{name}'] == value
    for name, value in metacal.METACAL_SETTINGS.items():
        assert m[f'mcal_{name}'] == value
    assert m['starsub_gsat'] == starsub.GSAT
    assert m['starsub_mask_rmax'] == starsub.MASK_RMAX
    assert m['cell_size'] == defaults.CELL_SIZE
    assert m['skymap'] == defaults.SKYMAP_VERS
    assert m['min_good_frac'] == defaults.MIN_GOOD_FRAC

    for name in ('lsst_mdet', 'kdeblend', 'ngmix', 'numpy'):
        assert m[f'version_{name}'] not in ('', 'unknown')
    assert m['date'].endswith('Z')
    assert m['hostname'] != ''
    assert m['command'] != ''
