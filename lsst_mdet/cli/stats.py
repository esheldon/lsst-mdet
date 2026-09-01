"""
get sums, stats for null tests etc.
"""
from numba import njit

SN = 0.27

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


# the mfrac threshold of the basic selection; a config can override
# it with a basic section, e.g. to study the shear vs mfrac
#
#     basic:
#       max_mfrac: 1.0
#
MAX_MFRAC = 0.1


def basic_select(st, max_mfrac=MAX_MFRAC):
    import numpy as np

    # only primary objects.  The copy matters: without it the
    # in-place &= below writes the accumulated logic back into
    # the caller's is_primary column
    logic = st['is_primary'].copy()

    # objects that were successfully processed
    logic &= (st['flags'] == 0)

    # objects with usable shapes.  Removes objects
    # that were DEBLENDED_AS_PSF and objects with bad
    # shape errors
    logic &= (st['g_flags'] == 0)

    # mfrac is the gaussian weighted fraction of zero weight
    # pixels
    logic &= (st['mfrac'] < max_mfrac)

    # sanity color checks
    logic &= (st['rmi'] > -2)
    logic &= (st['rmi'] < 3)
    logic &= (st['imz'] > -2)
    logic &= (st['imz'] < 3)

    # skip large objects. 20 for exp, 4 for gauss (future ladder may
    # effectively use gauss?)
    Tratio = np.zeros(st.size)
    w, = np.where(st['psf_T'] > 0)
    if w.size > 0:
        Tratio[w] = st['T'][w] / st['psf_T'][w]

    # these are arbitrary at this point
    logic &= (Tratio < 20)
    logic &= (st['T'] < 20)

    # don't include very high PSF ellipticity
    logic &= (np.abs(st['psfrec_g1_r']) < 0.05)
    logic &= (np.abs(st['psfrec_g1_i']) < 0.05)
    logic &= (np.abs(st['psfrec_g1_z']) < 0.05)

    logic &= (np.abs(st['psfrec_g2_r']) < 0.05)
    logic &= (np.abs(st['psfrec_g2_i']) < 0.05)
    logic &= (np.abs(st['psfrec_g2_z']) < 0.05)

    w, = np.where(logic)

    return st[w]


def galaxy_select(st):
    import numpy as np

    Tratio = st['T'] / st['psf_T']
    logic = (st['s2n'] > 10) & (Tratio > 0.5)

    w, = np.where(logic)

    return st[w]


def diagnostic_select(st):
    """
    the broad selection for the hist2d diagnostics: primary,
    successfully processed, low mfrac.  Deliberately no g_flags
    cut (it removes the negative T measurements, NONPOS_SIZE, so
    half the stellar locus), and no color or size cuts
    """
    import numpy as np

    logic = (
        st['is_primary']
        & (st['flags'] == 0)
        & (st['mfrac'] < 0.1)
    )
    w, = np.where(logic)
    return st[w]


def get_weights(st):
    cov_trace = st['g1_err'] ** 2 + st['g2_err'] ** 2
    return 1.0 / (SN ** 2 + cov_trace)


# config keys that are sections, not binning entries
RESERVED_CONFIG_KEYS = ('basic', 'select', 'hist2d')


def get_max_mfrac(config):
    """
    the basic selection mfrac threshold, from the optional basic
    section of the config
    """
    return config.get('basic', {}).get('max_mfrac', MAX_MFRAC)


def get_bin_config(config):
    """
    the binning entries of the config, leaving out the reserved
    select and hist2d sections
    """
    return {
        key: config[key] for key in config
        if key not in RESERVED_CONFIG_KEYS
    }


def get_named_value(st, name):
    """
    a column or a derived value by name: Tratio (T/psf_T),
    T_times_T_err, T_div_T_err, gmag (|g|), else the column itself
    """
    import numpy as np

    if name == 'Tratio':
        return st['T'] / st['psf_T']
    elif name == 'T_times_T_err':
        return st['T'] * st['T_err']
    elif name == 'T_div_T_err':
        return st['T'] / st['T_err']
    elif name == 'gmag':
        return np.hypot(st['g1'], st['g2'])
    return st[name]


def config_select(st, config):
    """
    apply the optional select section of the config: named values
    (see get_named_value) with optional minval and maxval, e.g.

        select:
          Tratio: {maxval: 5}
          T_times_T_err: {maxval: 1}
    """
    import numpy as np

    sel = config.get('select')
    if sel is None:
        return st

    logic = np.ones(st.size, dtype=bool)
    for name, lims in sel.items():
        vals = get_named_value(st, name)
        if 'minval' in lims:
            logic &= vals >= lims['minval']
        if 'maxval' in lims:
            logic &= vals <= lims['maxval']

    w, = np.where(logic)
    return st[w]


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


def _write_sums_output(fname, allsums, allstats, allhist):
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


def _dosums_main(config_file, flist_file, outfile):
    import rustfits
    from tqdm import tqdm
    import yaml

    with open(config_file) as fobj:
        config = yaml.safe_load(fobj)

    bin_config = get_bin_config(config)
    max_mfrac = get_max_mfrac(config)
    if max_mfrac != MAX_MFRAC:
        print(f'basic selection mfrac < {max_mfrac} from the config')

    flist = _read_flist(flist_file)
    # flist = flist[:100]

    allsums = {}
    allstats = {}
    for key in bin_config:
        allsums[key] = {'ns': [], '1p': [], '1m': []}
        allstats[key] = _get_stat_struct()

    allhist = _init_hist2d(config)

    for fname in tqdm(flist, ascii=True, ncols=70):
        if fname == '' or fname == '#':
            continue

        orig = rustfits.read(fname)

        basic = basic_select(orig, max_mfrac=max_mfrac)
        gals = galaxy_select(basic)
        gals = config_select(gals, config)

        _do_sums(
            allsums=allsums,
            allstats=allstats,
            st=gals,
            config=bin_config,
        )
        # the diagnostic histograms are made from the broad
        # diagnostic selection: no g_flags cut (negative T kept),
        # no galaxy s2n/Tratio cuts, no config select cuts, so
        # they show the population the cuts act on
        _do_hist2d(allhist, diagnostic_select(orig), config)

    _write_sums_output(outfile, allsums, allstats, allhist)


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


def _write_means_output(fname, allmeans, allhist):
    import rustfits

    print('writing to:', fname)

    with rustfits.FITS(fname, 'w+') as fits:
        for binval_name, binval_means in allmeans.items():

            fits.write_table(binval_means, extname=binval_name)

        _write_hist2d(fits, allhist)


def _dostats_main(config_file, flist, nrand, seed, outfile, nproc=1):
    import yaml
    import numpy as np

    rng = np.random.RandomState(seed)

    with open(config_file) as fobj:
        config = yaml.safe_load(fobj)

    bin_config = get_bin_config(config)

    allsums, allstats = _read_all_sums(config=bin_config, flist=flist)
    allhist = _read_and_sum_hist2d(flist=flist, config=config)

    allmeans = _do_all_bootstraps(
        allsums=allsums,
        nrand=nrand,
        rng=rng,
        nproc=nproc,
    )
    _write_means_output(outfile, allmeans, allhist)


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
    return parser.parse_args()


def _read_all_means(fname):
    import rustfits

    allmeans = {}

    with rustfits.FITS(fname) as fits:
        for hdu in fits:
            if not hdu.has_data:
                continue
            key = hdu.extname
            if key.startswith('hist2d_'):
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
    else:
        xlabel = binval_name

    if bconfig['use_log']:
        xlabel = r'log$_{10}$(' + xlabel + ')'

    return xlabel


def _doplot_g1g2_vs_binval(binval_name, means, bconfig, outfront):
    """
    the response corrected mean shear g/R in bins of the value
    """
    import matplotlib.pyplot as mplt

    fig, ax = mplt.subplots(figsize=(10, 10 / 1.62))

    ax.set(
        xlabel=_get_xlabel(binval_name, bconfig),
        ylabel=r'$g / R$',
        ylim=[-0.002, 0.002],
    )

    _add_scaled_hist(ax=ax, hist=means['hist'][0], bconfig=bconfig)

    ax.errorbar(
        means['binval'][0],
        means['g1'][0],
        means['g1_err'][0],
        marker='o',
        label=r'$g_1 / R$',
    )
    ax.errorbar(
        means['binval'][0],
        means['g2'][0],
        means['g2_err'][0],
        marker='o',
        label=r'$g_2 / R$',
    )
    ax.axhline(0, color='black')
    ax.legend()

    outfile = outfront + f'{binval_name}.pdf'
    print('writing:', outfile)
    fig.savefig(outfile)
    mplt.close(fig)


def _doplot_R_vs_binval(binval_name, means, bconfig, outfront):
    """
    the response R = R11 in bins of the value, with its bootstrap
    error
    """
    import matplotlib.pyplot as mplt

    fig, ax = mplt.subplots(figsize=(10, 10 / 1.62))

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


def _plotstats_main(config_file, fname, outfront):
    import yaml
    import matplotlib

    # files only: never let matplotlib probe for a display, which
    # is slow (minutes) on a login node with X forwarding
    matplotlib.use('Agg')

    with open(config_file) as fobj:
        config = yaml.safe_load(fobj)

    print('reading:', fname)
    allmeans = _read_all_means(fname)

    for binval_name in allmeans:
        _doplot_g1g2_vs_binval(
            binval_name=binval_name,
            means=allmeans[binval_name],
            bconfig=config[binval_name],
            outfront=outfront,
        )
        _doplot_R_vs_binval(
            binval_name=binval_name,
            means=allmeans[binval_name],
            bconfig=config[binval_name],
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
    )
