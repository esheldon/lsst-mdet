"""
The provenance of an output file: the meta table.

One row of plain columns, so a reader gets every setting a
catalog was measured with straight from the FITS extension: the
run options, the settings of every processing stage taken from
the module constants the code itself runs with, the versions of
the packages, and the run identity (date, host, command line).
"""
import numpy as np

# string column widths: paths and lists get room, short names less
STRING_WIDTHS = {
    'model': 8,
    'repo': 256,
    'collections': 256,
    'patch_dir': 256,
    'gaia_file': 256,
    'cells': 256,
    'inject_profiles': 256,
    'inject_objects': 256,
    'command': 1024,
    'hostname': 64,
    'date': 32,
}
DEFAULT_STRING_WIDTH = 32

# the packages whose versions are recorded: the measurement engine
# and its k-space fitter, the metacal operator, the detector, the
# grouper, this package and numpy
VERSION_PACKAGES = (
    'lsst_mdet', 'kdeblend', 'ngmix', 'metacal', 'sep', 'sxdes',
    'fofx', 'numpy',
)


def make_meta(run_options):
    """
    The one-row meta table for an output file.

    Parameters
    ----------
    run_options: dict
        The run identity and options as invoked: tract, patch,
        seed, model, with_mdet, redo_bg, starsub, deblend,
        s2_detect and the other process_cells options (repo,
        collections, patch_dir, gaia_file, gsub, apod_stars,
        cells).  Lists become comma-joined strings, None an
        empty string

    Returns
    -------
    structured array with one row
    """
    entries = list(run_options.items())
    entries += settings_entries()
    entries += version_entries()
    entries += identity_entries()

    dtype = [(name, _column_dtype(name, value)) for name, value in entries]
    meta = np.zeros(1, dtype=dtype)
    for name, value in entries:
        meta[name] = _column_value(value)
    return meta


def settings_entries():
    """
    The settings of every processing stage, prefixed by stage.

    Taken from the module constants at call time, so the record
    cannot drift from what ran.
    """
    from . import defaults, detect, metacal, deblend, extra_detect
    from . import starsub, mfrac, apodize, inject

    entries = []

    entries += [
        ('cell_size', defaults.CELL_SIZE),
        ('cell_overlap', defaults.CELL_OVERLAP),
        ('cell_overlap_low', defaults.CELL_OVERLAP_LOW),
        ('cell_overlap_high', defaults.CELL_OVERLAP_HIGH),
        ('skymap', defaults.SKYMAP_VERS),
        ('min_good_frac', defaults.MIN_GOOD_FRAC),
    ]
    entries += _prefixed('detect', detect.DETECT_SETTINGS)
    entries += _prefixed('mcal', metacal.METACAL_SETTINGS)
    entries += _prefixed('deblend', deblend.DEBLEND_SETTINGS)
    entries += _prefixed('inject', inject.INJECT_SETTINGS)
    entries += [
        ('deblend_r_dup_fit', deblend.R_DUP_FIT),
        ('deblend_group_box_pad', deblend.GROUP_BOX_PAD),
        ('deblend_maxiter_size_ref', deblend.MAXITER_SIZE_REF),
        ('s2_jscale', extra_detect.S2_JSCALE),
        ('s2_extra_min_sep', extra_detect.S2_EXTRA_MIN_SEP),
        ('s2_extra_dup', extra_detect.S2_EXTRA_DUP),
        ('s2_nreal', extra_detect.S2_NREAL),
        ('mfrac_fwhm', mfrac.MFRAC_FWHM),
        ('apod_rad', apodize.AP_RAD),
    ]
    entries += [
        ('starsub_' + name.lower(), getattr(starsub, name))
        for name in STARSUB_NAMES
    ]
    return entries


# the star subtraction and masking constants (see starsub.py)
STARSUB_NAMES = (
    'MASK_SLOPE', 'MASK_RMAX', 'MINRAD', 'STAR_MARGIN', 'GSAT', 'GSUB',
    'RUWE_MAX', 'BG_GROW', 'APOD_STARS',
    'TMPL_HALF', 'TMPL_OUT_HALF', 'TMPL_EXT_FACTOR', 'TMPL_OUT_MAX',
    'TMPL_NSTAR', 'TMPL_GMIN', 'TMPL_GMAX', 'TMPL_GMAX_CAP',
    'TMPL_MIN_CAND', 'TMPL_MIN_STAMPS', 'HALO_SLOPE',
    'AUR_GMIN', 'AUR_GMAX', 'AUR_RMAX', 'AUR_SLOPE', 'AUR_SLOPE_MIN',
    'AUR_SLOPE_MAX', 'AUR_MIN_STARS', 'AUR_SLOPE_SEP', 'AUR_BREAK',
    'AUR_AMP_GUARD',
)


def version_entries():
    """
    The version string of each package in VERSION_PACKAGES.

    The package's __version__ when it has one, else the installed
    distribution version, else 'unknown'.
    """
    return [
        ('version_' + name, package_version(name))
        for name in VERSION_PACKAGES
    ]


def package_version(name):
    import importlib
    from importlib.metadata import version, PackageNotFoundError

    try:
        mod = importlib.import_module(name)
        v = getattr(mod, '__version__', None)
        if v is not None:
            return str(v)
    except ImportError:
        pass
    try:
        return version(name)
    except PackageNotFoundError:
        return 'unknown'


def identity_entries():
    """
    When, where and how the file was made.
    """
    import sys
    import socket
    from datetime import datetime, timezone

    return [
        ('date', datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')),
        ('hostname', socket.gethostname()),
        ('command', ' '.join(sys.argv)),
    ]


def _prefixed(prefix, settings):
    return [(f'{prefix}_{name}', value) for name, value in settings.items()]


def _column_dtype(name, value):
    if isinstance(value, (bool, np.bool_)):
        return bool
    if isinstance(value, (int, np.integer)):
        return 'i8'
    if isinstance(value, (float, np.floating)):
        return 'f8'
    width = STRING_WIDTHS.get(name, DEFAULT_STRING_WIDTH)
    return f'U{width}'


def _column_value(value):
    if value is None:
        return ''
    if isinstance(value, (list, tuple)):
        return ','.join(str(v) for v in value)
    return value
