"""
sums and stats for the shear null tests, driven by a yaml config

    select:            # the selection stages, applied in order
      basic:           # a usable measurement; also the hist2d sample
        is_primary: {equal: true}
        flags:      {equal: 0}
        mfrac:      {maxval: 0.1}
      shape:           # a usable shape
        g_flags:    {equal: 0}
        rmi:        {minval: -2, maxval: 3}
        psfrec_gmax: {maxval: 0.05}
      galaxy:          # the galaxy sample for the binned stats
        s2n:        {minval: 10}
        Tratio:     {minval: 0.5}
    bins:              # the binned stats: g/R and R vs the value
      s2n: {minval: 10, maxval: 350, nbin: 20, use_log: true}
    hist2d:            # optional 2d diagnostics over the basic stage
      Tratio_vs_s2n:
        x: {name: s2n, minval: 5, maxval: 5000, nbin: 200, use_log: true}
        y: {name: Tratio, minval: -0.6, maxval: 30, nbin: 200,
            use_symlog: true, linthresh: 0.3}

A selection entry names a column or a derived value (see
get_named_value) with the tests minval, maxval (both inclusive),
equal and absmax (|value| <= absmax); a stage is the AND of its
entries, and a non-finite value fails every test.  A stage may also
carry an exclude list, each item a set of entries whose tests all
passing drops the object, e.g. to leave out a sky region

      galaxy:
        s2n: {minval: 10}
        exclude:
          - {ra: {minval: 288}, dec: {minval: -8}}

All three stages
must be present, {} for one with no cuts: there are no selection
defaults in the code.  The binned stats use basic + shape + galaxy,
the hist2d diagnostics basic only.  The config text is stored in
the sums and stats files in a 'config' extension, see
read_config_text
"""
from numba import njit

# the per-component shape noise in raw (pre-response) units, for
# the weights: the S/N 240-950 plateau of the noise-subtracted
# shear scatter, 0.255 after response correction, times the mean
# response of that range, 0.855 (measured on run-dp2-v00, see its
# notes.txt, shape noise measurement)
SN = 0.219

# the selection stages, in order
STAGES = ('basic', 'shape', 'galaxy')
SELECT_TESTS = ('minval', 'maxval', 'equal', 'absmax')
EXCLUDE_KEY = 'exclude'
TOP_LEVEL_KEYS = ('select', 'bins', 'hist2d')
DERIVED_VALUES = (
    'Tratio', 'T_times_T_err', 'T_div_T_err', 'gmag', 'psfrec_gmax',
    'cell_edge_dist', 'psf_fwhm',
)
CONFIG_EXTNAME = 'config'


class ConfigError(ValueError):
    pass


def load_config(fname):
    """
    read and check the config.  The text is kept as config['_text']
    for the provenance extension of the outputs
    """
    import yaml

    with open(fname) as fobj:
        text = fobj.read()
    config = yaml.safe_load(text)
    validate_config(config, fname)
    config['_text'] = text
    return config


def validate_config(config, fname='config'):
    """
    the structure: the three select stages, a non-empty bins
    section, known test names and no unknown top level keys
    """
    if not isinstance(config, dict):
        raise ConfigError(f'{fname}: not a mapping')

    extra = set(config) - set(TOP_LEVEL_KEYS) - {'_text'}
    if extra:
        raise ConfigError(
            f'{fname}: unknown top level keys {sorted(extra)}; the '
            f'sections are {TOP_LEVEL_KEYS}, binned entries go '
            'under bins'
        )

    sel = config.get('select')
    if not isinstance(sel, dict) or any(s not in sel for s in STAGES):
        raise ConfigError(
            f'{fname}: select must have the stages {STAGES}; use '
            '{} for a stage with no cuts'
        )

    def check_entries(entries, where):
        if not isinstance(entries, dict):
            raise ConfigError(f'{fname}: {where} is not a mapping')
        for name, tests in entries.items():
            if not isinstance(tests, dict) or len(tests) == 0:
                raise ConfigError(
                    f'{fname}: {where}.{name} needs tests from '
                    f'{SELECT_TESTS}'
                )
            bad = set(tests) - set(SELECT_TESTS)
            if bad:
                raise ConfigError(
                    f'{fname}: {where}.{name}: unknown tests '
                    f'{sorted(bad)}; use {SELECT_TESTS}'
                )

    for stage in STAGES:
        entries = sel[stage] or {}
        if not isinstance(entries, dict):
            raise ConfigError(f'{fname}: select.{stage} is not a mapping')
        # the exclude list: each item is a set of tests that, all
        # passing, drops the object (a region to leave out, say)
        exclude = entries.get(EXCLUDE_KEY, [])
        if not isinstance(exclude, list):
            raise ConfigError(
                f'{fname}: select.{stage}.{EXCLUDE_KEY} must be a list'
            )
        for i, item in enumerate(exclude):
            check_entries(item, f'select.{stage}.{EXCLUDE_KEY}[{i}]')
        check_entries(
            {k: v for k, v in entries.items() if k != EXCLUDE_KEY},
            f'select.{stage}',
        )
        sel[stage] = entries

    bins = config.get('bins')
    if not isinstance(bins, dict) or len(bins) == 0:
        raise ConfigError(f'{fname}: bins section missing or empty')


def validate_config_columns(config, st):
    """
    check every value name in the config against a catalog (its
    columns plus the derived names) before the run starts
    """
    names = set()
    for stage in STAGES:
        entries = config['select'][stage]
        names |= set(entries) - {EXCLUDE_KEY}
        for item in entries.get(EXCLUDE_KEY, []):
            names |= set(item)
    names |= set(config['bins'])
    for hconfig in (config.get('hist2d') or {}).values():
        names |= {hconfig['x']['name'], hconfig['y']['name']}

    known = set(st.dtype.names) | set(DERIVED_VALUES)
    bad = sorted(names - known)
    if bad:
        raise ConfigError(
            f'unknown value names in the config: {bad}; columns and '
            f'the derived values {DERIVED_VALUES} are allowed'
        )


def _tests_mask(st, entries):
    """
    the AND of the tests of a set of entries (name -> tests)
    """
    import numpy as np

    logic = np.ones(st.size, dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        for name, tests in entries.items():
            vals = np.asarray(get_named_value(st, name))
            for test, lim in tests.items():
                if test == 'minval':
                    logic &= vals >= lim
                elif test == 'maxval':
                    logic &= vals <= lim
                elif test == 'equal':
                    logic &= vals == lim
                elif test == 'absmax':
                    logic &= np.abs(vals) <= lim
    return logic


def select_stage(st, config, stage):
    """
    apply one selection stage: the AND of its entries, each a
    named value with minval/maxval (inclusive), equal or absmax
    tests, minus the objects matching any item of its exclude
    list.  Non-finite values fail every test
    """
    import numpy as np

    entries = config['select'][stage]
    logic = _tests_mask(
        st, {k: v for k, v in entries.items() if k != EXCLUDE_KEY},
    )
    for item in entries.get(EXCLUDE_KEY, []):
        logic &= ~_tests_mask(st, item)

    w, = np.where(logic)
    return st[w]


def apply_selection(st, config, stages=STAGES):
    """
    apply the selection stages in order; all three by default, the
    sample of the binned stats
    """
    for stage in stages:
        st = select_stage(st, config, stage)
    return st


def describe_selection(config):
    """
    one line per stage, for the log
    """
    lines = []
    for stage in STAGES:
        entries = config['select'][stage]
        if len(entries) == 0:
            lines.append(f'    {stage}: no cuts')
            continue

        def fmt(ent):
            return '; '.join(
                f'{name} ' + ' '.join(f'{t} {v}' for t, v in tests.items())
                for name, tests in ent.items()
            )

        parts = fmt({k: v for k, v in entries.items() if k != EXCLUDE_KEY})
        lines.append(f'    {stage}: {parts}')
        for item in entries.get(EXCLUDE_KEY, []):
            lines.append(f'        excluding: {fmt(item)}')
    return '\n'.join(lines)


def _write_config(fits, config):
    """
    the config text as a one row table in the config extension
    """
    import numpy as np

    text = config['_text'].encode()
    st = np.zeros(1, dtype=[('config', f'S{max(len(text), 1)}')])
    st['config'][0] = text
    fits.write_table(st, extname=CONFIG_EXTNAME)


def read_config_text(fname):
    """
    the config text stored in a sums or stats file
    """
    import rustfits

    # raw bytes: the default read rejects non-ascii text, e.g. an
    # accent in a comment
    with rustfits.FITS(fname) as fits:
        raw = fits[CONFIG_EXTNAME].read_column('config', as_bytes=True)
    return bytes(raw[0]).decode('utf-8', errors='replace')

#
# dosums
#


def _get_dosums_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--flist', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def get_weights(st):
    # the trace is the two-component measurement variance, so it
    # pairs with the two-component intrinsic variance 2 SN^2
    cov_trace = st['g1_err'] ** 2 + st['g2_err'] ** 2
    return 1.0 / (2 * SN ** 2 + cov_trace)


def get_named_value(st, name):
    """
    a column or a derived value by name (DERIVED_VALUES): Tratio
    (T/psf_T), T_times_T_err, T_div_T_err, gmag (|g|), psfrec_gmax
    (the largest |psfrec g1|, |psfrec g2| over the bands),
    cell_edge_dist (pixels from the primary region boundary of the
    cell, positive inside, negative in the overlap band), psf_fwhm
    (arcsec, from psfrec_T with the gaussian relation T = 2 sigma^2,
    fwhm = 2 sqrt(2 ln 2) sigma), else the column itself
    """
    import numpy as np

    if name == 'cell_edge_dist':
        from ..defaults import CELL_OVERLAP_HIGH, CELL_OVERLAP_LOW
        x, y = st['xcell'], st['ycell']
        return np.min([
            x - CELL_OVERLAP_LOW, CELL_OVERLAP_HIGH - x,
            y - CELL_OVERLAP_LOW, CELL_OVERLAP_HIGH - y,
        ], axis=0)
    elif name == 'Tratio':
        return st['T'] / st['psf_T']
    elif name == 'T_times_T_err':
        return st['T'] * st['T_err']
    elif name == 'T_div_T_err':
        return st['T'] / st['T_err']
    elif name == 'gmag':
        return np.hypot(st['g1'], st['g2'])
    elif name == 'psf_fwhm':
        sigma = np.sqrt(st['psfrec_T'] / 2)
        return 2 * np.sqrt(2 * np.log(2)) * sigma
    elif name == 'psfrec_gmax':
        cols = [
            c for c in st.dtype.names
            if c.startswith('psfrec_g1_') or c.startswith('psfrec_g2_')
        ]
        return np.max([np.abs(st[c]) for c in cols], axis=0)
    return st[name]


#
# the optional hist2d section: 2d diagnostic histograms over the ns
# step, accumulated in dosums, summed over chunks in dostats and
# plotted by plotstats, e.g.
#
#     hist2d:
#       Tratio_vs_s2n:
#         x: {name: s2n, minval: 10, maxval: 2000, nbin: 200,
#             use_log: true}
#         y: {name: Tratio, minval: -1, maxval: 30, nbin: 200,
#             use_symlog: true, linthresh: 0.3}
#
# an axis is linear by default, use_log for logarithmic binning
# (positive values only), or use_symlog with linthresh for binning
# uniform in sign(v) * log10(1 + |v|/linthresh): linear through
# zero so negative values (stars scattering below T = 0) are kept,
# logarithmic well above linthresh
#

def _symlog(v, linthresh):
    import numpy as np
    return np.sign(v) * np.log10(1.0 + np.abs(v) / linthresh)


def _symlog_inv(t, linthresh):
    import numpy as np
    return np.sign(t) * linthresh * (10.0 ** np.abs(t) - 1.0)


def _hist2d_axis_edges(aconfig):
    """
    the bin edges in true axis values: uniform for a linear axis,
    uniform in log10 (returned as log10 values) for use_log, and
    uniform in the symlog transform (returned as true values, so
    non-uniform) for use_symlog
    """
    import numpy as np

    if aconfig.get('use_log') and aconfig.get('use_symlog'):
        raise ValueError('use_log and use_symlog are exclusive')

    minval = aconfig['minval']
    maxval = aconfig['maxval']
    nbin = aconfig['nbin']

    if aconfig.get('use_symlog'):
        lt = aconfig['linthresh']
        tedges = np.linspace(
            _symlog(minval, lt), _symlog(maxval, lt), nbin + 1,
        )
        return _symlog_inv(tedges, lt)

    if aconfig.get('use_log'):
        minval = np.log10(minval)
        maxval = np.log10(maxval)
    return np.linspace(minval, maxval, nbin + 1)


def _init_hist2d(config):
    import numpy as np

    h2 = config.get('hist2d')
    if h2 is None:
        return {}
    return {
        key: np.zeros(
            (hconfig['x']['nbin'], hconfig['y']['nbin']),
            dtype='i8',
        )
        for key, hconfig in h2.items()
    }


def _do_hist2d(allhist, st, config):
    """
    accumulate the diagnostic histograms from the ns rows
    """
    import numpy as np

    h2 = config.get('hist2d')
    if h2 is None:
        return

    wns, = np.where(st['mcal_step'] == 'ns')
    stns = st[wns]

    for key, hconfig in h2.items():
        vals = {}
        with np.errstate(divide='ignore', invalid='ignore'):
            for axis in ('x', 'y'):
                v = np.asarray(
                    get_named_value(stns, hconfig[axis]['name']),
                    dtype='f8',
                )
                # symlog and linear axes bin the raw values (the
                # symlog edges are non-uniform true values)
                if hconfig[axis].get('use_log'):
                    v = np.log10(v)
                vals[axis] = v

        ok = np.isfinite(vals['x']) & np.isfinite(vals['y'])
        counts, _, _ = np.histogram2d(
            vals['x'][ok], vals['y'][ok],
            bins=[_hist2d_axis_edges(hconfig['x']),
                  _hist2d_axis_edges(hconfig['y'])],
        )
        allhist[key] += counts.astype('i8')


def _write_hist2d(fits, allhist):
    import numpy as np

    for key, counts in allhist.items():
        st = np.zeros(1, dtype=[('counts', 'i8', counts.shape)])
        st['counts'][0] = counts
        fits.write_table(st, extname=f'hist2d_{key}')


def _read_and_sum_hist2d(flist, config):
    import rustfits

    allhist = _init_hist2d(config)
    if not allhist:
        return allhist

    for fname in flist:
        with rustfits.FITS(fname) as fits:
            for key in allhist:
                allhist[key] += (
                    fits[f'hist2d_{key}'].read()['counts'][0]
                )
    return allhist


def _do_sums_by_binval_name(
    allsums,
    allstats,
    binval_name,
    st,
    weights,
    config,
):
    import numpy as np

    sums_dict = allsums[binval_name]
    stats = allstats[binval_name]
    bconfig = config[binval_name]

    minval = bconfig['minval']
    maxval = bconfig['maxval']

    if bconfig['use_log']:
        minval = np.log10(minval)
        maxval = np.log10(maxval)

    for mcal_step, mcal_step_sums in sums_dict.items():
        wtype, = np.where(st['mcal_step'] == mcal_step)
        sums = _get_sum_struct(bconfig['nbin'])

        binval = get_named_value(st[wtype], binval_name)

        if bconfig['use_log']:
            binval = np.log10(binval)

        _do_sums_by_field(
            binval=binval,
            minval=minval,
            maxval=maxval,
            nbin=bconfig['nbin'],
            g1=st['g1'][wtype],
            g2=st['g2'][wtype],
            T=st['T'][wtype],
            psf_T=st['psf_T'][wtype],
            weights=weights[wtype],
            sums=sums,
            stats=stats,
        )
        mcal_step_sums.append(sums)


def _do_sums(allsums, allstats, st, config):

    weights = get_weights(st)

    for binval_name in allsums:
        _do_sums_by_binval_name(
            allsums=allsums,
            allstats=allstats,
            binval_name=binval_name,
            st=st,
            weights=weights,
            config=config,
        )


@njit
def _do_sums_by_field(
    binval,
    minval,
    maxval,
    nbin,
    g1,
    g2,
    T,
    psf_T,
    weights,
    sums,
    stats,
):
    nobj = g1.size

    binsize = (maxval - minval) / nbin

    for iobj in range(nobj):

        this_binval = binval[iobj]

        binnum = int((this_binval - minval) / binsize)

        if binnum < 0 or binnum > (nbin - 1):
            continue

        wt = weights[iobj]

        sums['n'][0, binnum] += 1
        sums['wsum'][0, binnum] += wt

        sums['binval'][0, binnum] += wt * this_binval
        sums['g1'][0, binnum] += wt * g1[iobj]
        sums['g2'][0, binnum] += wt * g2[iobj]

        stats['binval_min'][0] = min(stats['binval_min'][0], this_binval)
        stats['binval_max'][0] = max(stats['binval_max'][0], this_binval)

        stats['weight_max'][0] = max(stats['weight_max'][0], wt)


def _get_stat_struct():
    import numpy as np

    dtype = [
        ('binval_min', 'f8'),
        ('binval_max', 'f8'),
        ('weight_max', 'f8'),
    ]

    st = np.zeros(1, dtype=dtype)
    return st


def _get_sum_struct(nbin):
    import numpy as np

    dtype = [
        ('n', 'i8', nbin),
        ('wsum', 'f8', nbin),
        ('binval', 'f8', nbin),
        ('g1', 'f8', nbin),
        ('g2', 'f8', nbin),
    ]

    st = np.zeros(1, dtype=dtype)
    return st


def _dict2array(d):
    """
    helper to convert a dict to an array with fields
    """
    import numpy as np

    def make_dtype(d):
        fields = []
        for k, v in d.items():
            a = np.asarray(v)
            if a.ndim == 0:
                fields.append((k, a.dtype))  # scalar field
            else:
                fields.append((k, a.dtype, a.shape))  # subarray field
        return fields

    return np.array([tuple(d.values())], dtype=make_dtype(d))


def _read_flist(fname):
    with open(fname) as fobj:
        flist = [f.strip() for f in fobj]

    return flist


def _write_sums_output(fname, allsums, allstats, allhist, config):
    import numpy as np
    import rustfits

    print('writing to:', fname)

    with rustfits.FITS(fname, 'w+') as fits:
        for binval_name, binval_sums in allsums.items():

            stats = allstats[binval_name]
            fits.write_table(stats, extname=f'{binval_name}_stats')

            for mcal_step, mcal_step_sums in binval_sums.items():
                extname = f'{binval_name}_{mcal_step}'

                sum_st = np.concatenate(mcal_step_sums)
                fits.write_table(sum_st, extname=extname)

        _write_hist2d(fits, allhist)
        _write_config(fits, config)


def _dosums_main(config_file, flist_file, outfile):
    import rustfits
    from tqdm import tqdm

    config = load_config(config_file)
    print('selection:')
    print(describe_selection(config))

    flist = _read_flist(flist_file)
    flist = [f for f in flist if f != '' and not f.startswith('#')]
    if len(flist) == 0:
        raise ValueError(f'no files in {flist_file}')

    allsums = {}
    allstats = {}
    for key in config['bins']:
        allsums[key] = {'ns': [], '1p': [], '1m': []}
        allstats[key] = _get_stat_struct()

    allhist = _init_hist2d(config)

    for i, fname in enumerate(tqdm(flist, ascii=True, ncols=70)):
        orig = rustfits.read(fname)
        if i == 0:
            validate_config_columns(config, orig)

        gals = apply_selection(orig, config)

        _do_sums(
            allsums=allsums,
            allstats=allstats,
            st=gals,
            config=config['bins'],
        )
        # the diagnostic histograms come from the basic stage only:
        # no shape cuts (g_flags would drop the negative T stars)
        # and no galaxy cuts, so they show the population the
        # later cuts act on
        _do_hist2d(allhist, select_stage(orig, config, 'basic'), config)

    _write_sums_output(outfile, allsums, allstats, allhist, config)


def dosums_cli():
    args = _get_dosums_args()
    _dosums_main(
        config_file=args.config,
        flist_file=args.flist,
        outfile=args.output,
    )


#
# dostats
#

def _get_dostats_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--flist', nargs='+', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--nrand', type=int, required=True)
    parser.add_argument('--nproc', type=int, default=1,
                        help='processes for the bootstrap; the '
                             'realizations are split over them, so '
                             'the exact bootstrap errors depend on '
                             'nproc for a given seed')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def _read_all_sums(flist, config):
    import numpy as np
    import rustfits

    allsums = {}
    allstats = {}
    for key in config:
        allsums[key] = {'ns': [], '1p': [], '1m': []}
        allstats[key] = []

    for fname in flist:
        with rustfits.FITS(fname) as fits:
            for key in config:
                for mcal_type in allsums[key]:
                    mcal_type_key = f'{key}_{mcal_type}'
                    allsums[key][mcal_type].append(
                        fits[mcal_type_key].read()
                    )
                allstats[key].append(
                    fits[f'{key}_stats'].read()
                )

    for key in allsums:
        for mcal_type in allsums[key]:
            allsums[key][mcal_type] = np.concatenate(
                allsums[key][mcal_type],
            )

        allstats[key] = np.concatenate(allstats[key])

    return allsums, allstats


def _get_mean_struct(n, nbin):
    import numpy as np

    dtype = [
        ('hist', 'i8', nbin),
        ('wsum', 'f8', nbin),
        ('binval', 'f8', nbin),
        ('binval_err', 'f8', nbin),
        ('g1', 'f8', nbin),
        ('g1_err', 'f8', nbin),
        ('g2', 'f8', nbin),
        ('g2_err', 'f8', nbin),
        ('R', 'f8', nbin),
        ('R_err', 'f8', nbin),
    ]
    st = np.zeros(n, dtype=dtype)
    for n in st.dtype.names:

        if n == 'hist':
            continue

        st[n] = np.nan

    return st


def _get_means(sums, ind):
    import numpy as np
    nbin = sums['wsum'].shape[1]
    means = _get_mean_struct(n=1, nbin=nbin)

    means['hist'][0] = sums['n'][ind].sum(axis=0)

    wsum = sums['wsum'][ind].sum(axis=0)
    means['wsum'][0] = wsum

    w, = np.where(wsum > 0)

    if w.size > 0:
        means['binval'][0, w] = (
            sums['binval'][ind][:, w].sum(axis=0) / wsum[w]
        )
        means['g1'][0, w] = sums['g1'][ind][:, w].sum(axis=0) / wsum[w]
        means['g2'][0, w] = sums['g2'][ind][:, w].sum(axis=0) / wsum[w]

    return means


def _get_corrected_means(sums, ind):
    means_ns = _get_means(sums=sums['ns'], ind=ind)
    means_1p = _get_means(sums=sums['1p'], ind=ind)
    means_1m = _get_means(sums=sums['1m'], ind=ind)

    R11 = (means_1p['g1'][0] - means_1m['g1'][0]) / 0.02
    means_ns['g1'][0] *= 1.0 / R11
    means_ns['g2'][0] *= 1.0 / R11
    means_ns['R'][0] = R11
    return means_ns


def _make_boot_means(sums, nrand, rng):
    """
    nrand bootstrap realizations of the corrected means
    """
    ntot = sums['ns'].size
    nbin = sums['ns']['wsum'].shape[1]
    boot_means = _get_mean_struct(n=nrand, nbin=nbin)

    for i in range(nrand):
        ind = rng.choice(ntot, size=ntot)
        boot_means[i] = _get_corrected_means(sums, ind=ind)

    return boot_means


# the sums shared with the fork-started bootstrap workers, set in
# _do_all_bootstraps before the pool is created so the children
# inherit them rather than receiving a pickled copy per task
_BOOT_ALLSUMS = None


def _boot_worker(task):
    import numpy as np

    key, seed, nrand = task
    rng = np.random.RandomState(seed)
    return key, _make_boot_means(_BOOT_ALLSUMS[key], nrand, rng)


def _do_bootstrap(sums, nrand, rng, boot_means=None):
    import numpy as np

    ntot = sums['ns'].size
    means = _get_corrected_means(sums, ind=np.arange(ntot))

    if boot_means is None:
        boot_means = _make_boot_means(sums, nrand, rng)

    means['binval_err'] = np.nanstd(
        boot_means['binval'],
        axis=0,
    )
    means['g1_err'] = np.nanstd(
        boot_means['g1'],
        axis=0,
    )
    means['g2_err'] = np.nanstd(
        boot_means['g2'],
        axis=0,
    )
    means['R_err'] = np.nanstd(
        boot_means['R'],
        axis=0,
    )

    return means


def _do_all_bootstraps(allsums, nrand, rng, nproc=1):
    import numpy as np

    if nproc <= 1:
        return {
            key: _do_bootstrap(
                sums=allsums[key], nrand=nrand, rng=rng,
            )
            for key in allsums
        }

    import multiprocessing as mp

    # split the realizations of every key over the workers; the
    # sums reach the children by fork, see _BOOT_ALLSUMS
    global _BOOT_ALLSUMS
    _BOOT_ALLSUMS = allsums

    keys = list(allsums)
    ntask_per_key = max(1, nproc // len(keys))
    tasks = []
    for key in keys:
        lo = 0
        for i in range(ntask_per_key):
            n = (nrand - lo) // (ntask_per_key - i)
            if n > 0:
                seed = int(rng.choice(2 ** 31))
                tasks.append((key, seed, n))
            lo += n

    try:
        with mp.get_context('fork').Pool(nproc) as pool:
            results = pool.map(_boot_worker, tasks)
    finally:
        _BOOT_ALLSUMS = None

    allmeans = {}
    for key in keys:
        boot_means = np.concatenate(
            [b for k, b in results if k == key],
        )
        allmeans[key] = _do_bootstrap(
            sums=allsums[key], nrand=nrand, rng=rng,
            boot_means=boot_means,
        )

    return allmeans


def _write_means_output(fname, allmeans, allhist, config):
    import rustfits

    print('writing to:', fname)

    with rustfits.FITS(fname, 'w+') as fits:
        for binval_name, binval_means in allmeans.items():

            fits.write_table(binval_means, extname=binval_name)

        _write_hist2d(fits, allhist)
        _write_config(fits, config)


def _dostats_main(config_file, flist, nrand, seed, outfile, nproc=1):
    import numpy as np

    rng = np.random.RandomState(seed)

    config = load_config(config_file)

    allsums, allstats = _read_all_sums(config=config['bins'], flist=flist)
    allhist = _read_and_sum_hist2d(flist=flist, config=config)

    allmeans = _do_all_bootstraps(
        allsums=allsums,
        nrand=nrand,
        rng=rng,
        nproc=nproc,
    )
    _write_means_output(outfile, allmeans, allhist, config)


def dostats_cli():
    args = _get_dostats_args()
    _dostats_main(
        config_file=args.config,
        flist=args.flist,
        seed=args.seed,
        nrand=args.nrand,
        outfile=args.output,
        nproc=args.nproc,
    )


#
# plot stats
#

def _get_plotstats_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fname', required=True)
    parser.add_argument('--outfront', required=True)
    parser.add_argument('--ymin', type=float, default=DEFAULT_PLOT_YMIN,
                        help='lower y limit for the g/R trend plots')
    parser.add_argument('--ymax', type=float, default=DEFAULT_PLOT_YMAX,
                        help='upper y limit for the g/R trend plots')
    return parser.parse_args()


def _read_all_means(fname):
    import rustfits

    allmeans = {}

    with rustfits.FITS(fname) as fits:
        for hdu in fits:
            if not hdu.has_data:
                continue
            key = hdu.extname
            if key.startswith('hist2d_') or key == CONFIG_EXTNAME:
                continue
            print(key)
            allmeans[key] = hdu.read()

    return allmeans


HIST_PEAK_FRAC = 0.9


def _add_scaled_hist(ax, hist, bconfig):
    """
    draw the bin counts as a filled gray histogram behind the
    points, scaled to span the y range from the bottom to
    HIST_PEAK_FRAC of the way up at the peak
    """
    import numpy as np

    nbin = bconfig['nbin']
    assert hist.size == nbin, f'hist has {hist.size} bins, config {nbin}'

    hmax = hist.max()
    if hmax <= 0:
        return

    minval = bconfig['minval']
    maxval = bconfig['maxval']

    if bconfig['use_log']:
        minval = np.log10(minval)
        maxval = np.log10(maxval)

    edges = np.linspace(minval, maxval, nbin + 1)

    ylo, yhi = ax.get_ylim()
    scaled = ylo + HIST_PEAK_FRAC * (yhi - ylo) * hist / hmax

    ax.stairs(
        scaled,
        edges,
        baseline=ylo,
        fill=True,
        color='gray',
        alpha=0.3,
        zorder=0,
    )


def _get_xlabel(binval_name, bconfig):
    if binval_name == 's2n':
        xlabel = 'S/N'
    elif binval_name == 'Tratio':
        xlabel = r'T / T$_{\mathrm{PSF}}$'
    elif binval_name == 'psfrec_g1':
        xlabel = r'PSF $g_1$'
    elif binval_name == 'psfrec_g2':
        xlabel = r'PSF $g_2$'
    elif binval_name == 'psf_fwhm':
        xlabel = 'PSF FWHM [arcsec]'
    elif binval_name == 'mfrac':
        xlabel = 'masked fraction'
    elif binval_name == 'T':
        xlabel = r'T [arcsec$^2$]'
    else:
        xlabel = binval_name

    if bconfig['use_log']:
        xlabel = r'log$_{10}$(' + xlabel + ')'

    return xlabel


DEFAULT_PLOT_YMIN = -0.002
DEFAULT_PLOT_YMAX = 0.002


def _doplot_g1g2_vs_binval(binval_name, means, bconfig, outfront,
                           ymin=DEFAULT_PLOT_YMIN,
                           ymax=DEFAULT_PLOT_YMAX):
    """
    the response corrected mean shear g/R in bins of the value
    """
    import matplotlib.pyplot as mplt

    # sized so the ~10pt fonts stay readable at half a text width
    # in the paper
    fig, ax = mplt.subplots(figsize=(6.25, 6.25 / 1.62))

    ax.set(
        xlabel=_get_xlabel(binval_name, bconfig),
        ylabel=r'$g / R$',
        ylim=[ymin, ymax],
    )

    _add_scaled_hist(ax=ax, hist=means['hist'][0], bconfig=bconfig)

    # the psf ellipticity trends get weighted linear fit overlays,
    # points without connecting lines, and the legend outside
    is_psf_e = binval_name.startswith('psfrec_g')

    markersize = 4.5
    ax.errorbar(
        means['binval'][0],
        means['g1'][0],
        means['g1_err'][0],
        marker='o',
        markersize=markersize,
        linestyle='none' if is_psf_e else '-',
        label=r'$g_1 / R$',
    )
    ax.errorbar(
        means['binval'][0],
        means['g2'][0],
        means['g2_err'][0],
        marker='o',
        markersize=markersize,
        linestyle='none' if is_psf_e else '-',
        label=r'$g_2 / R$',
    )

    # the slope against the matching psf component is the leakage
    # alpha; the offset c is the value at zero psf ellipticity
    if is_psf_e:
        import numpy as np
        x = means['binval'][0]
        for comp, color in (('g1', 'C0'), ('g2', 'C1')):
            y = means[comp][0]
            e = means[f'{comp}_err'][0]
            g = (means['hist'][0] > 100) & (e > 0) & np.isfinite(y)
            w = 1.0 / e[g] ** 2
            xm = np.sum(w * x[g]) / w.sum()
            ym = np.sum(w * y[g]) / w.sum()
            slope = (np.sum(w * (x[g] - xm) * (y[g] - ym))
                     / np.sum(w * (x[g] - xm) ** 2))
            serr = np.sqrt(1.0 / np.sum(w * (x[g] - xm) ** 2))
            c0 = ym - slope * xm
            c0err = np.sqrt(1.0 / w.sum()
                            + xm ** 2 * serr ** 2)
            xx = np.array([x[g].min(), x[g].max()])
            sub = comp[1]
            ax.plot(
                xx, ym + slope * (xx - xm), color=color,
                linestyle='dashed', linewidth=1,
                label=(rf'$\alpha(g_{sub}) = {slope:+.3f} '
                       rf'\pm {serr:.3f}$''\n'
                       rf'$c(g_{sub}) = ({c0 * 1e3:+.2f} '
                       rf'\pm {c0err * 1e3:.2f}) '
                       r'\times 10^{-3}$'),
            )

    ax.axhline(0, color='black')
    if is_psf_e:
        # flat above the axes so the plot keeps its full width
        ax.legend(loc='lower left',
                  bbox_to_anchor=(0.0, 1.02, 1.0, 0.3),
                  mode='expand', ncol=2, fontsize=8)
    else:
        ax.legend()

    outfile = outfront + f'{binval_name}.pdf'
    print('writing:', outfile)
    if is_psf_e:
        fig.savefig(outfile, bbox_inches='tight')
    else:
        fig.savefig(outfile)
    mplt.close(fig)


def _doplot_R_vs_binval(binval_name, means, bconfig, outfront):
    """
    the response R = R11 in bins of the value, with its bootstrap
    error
    """
    import matplotlib.pyplot as mplt

    # sized so the ~10pt fonts stay readable at half a text width
    # in the paper
    fig, ax = mplt.subplots(figsize=(5.5, 5.5 / 1.62))

    ax.set(
        xlabel=_get_xlabel(binval_name, bconfig),
        ylabel=r'$R$',
    )

    ax.errorbar(
        means['binval'][0],
        means['R'][0],
        means['R_err'][0],
        marker='o',
        color='black',
    )
    # after the points, so the histogram scales to their y range
    _add_scaled_hist(ax=ax, hist=means['hist'][0], bconfig=bconfig)

    outfile = outfront + f'R-{binval_name}.pdf'
    print('writing:', outfile)
    fig.savefig(outfile)
    mplt.close(fig)


def _doplot_hist2d(key, counts, hconfig, outfront):
    import numpy as np
    import matplotlib.pyplot as mplt
    from matplotlib.colors import LogNorm

    fig, ax = mplt.subplots(figsize=(8, 7))

    xedges = _hist2d_axis_edges(hconfig['x'])
    yedges = _hist2d_axis_edges(hconfig['y'])

    # rows are y for pcolormesh; empty bins masked rather than
    # drawn as the lowest color.  Rasterized: as vector art the
    # nbin^2 rectangles make a slow, megabyte pdf
    masked = np.ma.masked_equal(counts.T, 0)
    pc = ax.pcolormesh(
        xedges, yedges, masked, norm=LogNorm(), cmap='inferno',
        rasterized=True,
    )
    fig.colorbar(pc, ax=ax, label='count')

    def axis_label(aconfig):
        label = aconfig['name']
        if aconfig.get('use_log'):
            label = r'log$_{10}$(' + label + ')'
        return label

    # a symlog axis holds true values with non-uniform edges; the
    # matplotlib symlog scale renders it linear through zero
    for axis, setscale in (('x', ax.set_xscale), ('y', ax.set_yscale)):
        if hconfig[axis].get('use_symlog'):
            setscale('symlog', linthresh=hconfig[axis]['linthresh'])

    ax.set(
        xlabel=axis_label(hconfig['x']),
        ylabel=axis_label(hconfig['y']),
    )

    outfile = outfront + f'hist2d-{key}.pdf'
    print('writing:', outfile)
    fig.savefig(outfile)
    mplt.close(fig)


def _plotstats_main(config_file, fname, outfront,
                    ymin=DEFAULT_PLOT_YMIN, ymax=DEFAULT_PLOT_YMAX):
    import matplotlib

    # files only: never let matplotlib probe for a display, which
    # is slow (minutes) on a login node with X forwarding
    matplotlib.use('Agg')

    config = load_config(config_file)

    print('reading:', fname)
    allmeans = _read_all_means(fname)

    for binval_name in allmeans:
        _doplot_g1g2_vs_binval(
            binval_name=binval_name,
            means=allmeans[binval_name],
            bconfig=config['bins'][binval_name],
            outfront=outfront,
            ymin=ymin,
            ymax=ymax,
        )
        _doplot_R_vs_binval(
            binval_name=binval_name,
            means=allmeans[binval_name],
            bconfig=config['bins'][binval_name],
            outfront=outfront,
        )

    allhist = _read_and_sum_hist2d(flist=[fname], config=config)
    for key, counts in allhist.items():
        _doplot_hist2d(
            key=key,
            counts=counts,
            hconfig=config['hist2d'][key],
            outfront=outfront,
        )


def plotstats_cli():
    args = _get_plotstats_args()
    _plotstats_main(
        config_file=args.config,
        fname=args.fname,
        outfront=args.outfront,
        ymin=args.ymin,
        ymax=args.ymax,
    )
