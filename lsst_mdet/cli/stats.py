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


def basic_select(st):
    import numpy as np

    # objects that were successfully processed
    logic = (st['flags'] == 0)

    # objects with usable shapes.  Removes objects
    # that were DEBLENDED_AS_PSF and objects with bad
    # shape errors
    logic &= (st['g_flags'] == 0)

    w, = np.where(logic)

    return st[w]


def galaxy_select(st):
    import numpy as np

    Tratio = st['T'] / st['psf_T']
    logic = (st['s2n'] > 10) & (Tratio > 0.5)

    w, = np.where(logic)

    return st[w]


def get_weights(st):
    cov_trace = st['g1_err'] ** 2 + st['g2_err'] ** 2
    return 1.0 / (SN ** 2 + cov_trace)


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

    for mcal_step, mcal_step_sums in sums_dict.items():
        wtype, = np.where(st['mcal_step'] == mcal_step)
        sums = _get_sum_struct(bconfig['nbin'])

        _do_sums_by_field(
            binval=st[binval_name][wtype],
            minval=bconfig['minval'],
            maxval=bconfig['maxval'],
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

        sums['bin_field'][0, binnum] += wt * this_binval
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
        ('bin_field', 'f8', nbin),
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


def _write_sums_output(fname, allsums, allstats):
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


def _dosums_main(config_file, flist_file, outfile):
    import rustfits
    from tqdm import tqdm
    import yaml

    with open(config_file) as fobj:
        config = yaml.safe_load(fobj)

    flist = _read_flist(flist_file)
    # flist = flist[:100]

    allsums = {}
    allstats = {}
    for key in config:
        allsums[key] = {'ns': [], '1p': [], '1m': []}
        allstats[key] = _get_stat_struct()

    for fname in tqdm(flist, ascii=True, ncols=70):
        if fname == '' or fname == '#':
            continue

        orig = rustfits.read(fname)

        basic = basic_select(orig)
        gals = galaxy_select(basic)

        _do_sums(
            allsums=allsums,
            allstats=allstats,
            st=gals,
            config=config,
        )

    _write_sums_output(outfile, allsums, allstats)


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
        ('bin_field', 'f8', nbin),
        ('bin_field_err', 'f8', nbin),
        ('g1', 'f8', nbin),
        ('g1_err', 'f8', nbin),
        ('g2', 'f8', nbin),
        ('g2_err', 'f8', nbin),
        ('R', 'f8', nbin),
        ('R_err', 'f8', nbin),
    ]
    st = np.zeros(n, dtype=dtype)
    for n in st.dtype.names:
        st[n] = np.nan

    return st


def _get_means(sums, ind):
    import numpy as np
    nbin = sums['wsum'].shape[1]
    means = _get_mean_struct(n=1, nbin=nbin)

    wsum = sums['wsum'][ind].sum(axis=0)
    w, = np.where(wsum > 0)

    if w.size > 0:
        means['bin_field'][0, w] = (
            sums['bin_field'][ind][:, w].sum(axis=0) / wsum[w]
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


def _do_bootstrap(sums, nrand, rng):
    import numpy as np

    ntot = sums['ns'].size
    means = _get_corrected_means(sums, ind=np.arange(ntot))

    nbin = sums['ns']['wsum'].shape[1]
    boot_means = _get_mean_struct(n=nrand, nbin=nbin)

    for i in range(nrand):
        ind = rng.choice(ntot, size=ntot)
        boot_means[i] = _get_corrected_means(sums, ind=ind)

    means['bin_field_err'] = np.nanstd(
        boot_means['bin_field'],
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


def _do_all_bootstraps(allsums, nrand, rng):
    allmeans = {}

    for key in allsums:
        allmeans[key] = _do_bootstrap(
            sums=allsums[key],
            nrand=nrand,
            rng=rng,
        )

    return allmeans


def _write_means_output(fname, allmeans):
    import rustfits

    print('writing to:', fname)

    with rustfits.FITS(fname, 'w+') as fits:
        for binval_name, binval_means in allmeans.items():

            fits.write_table(binval_means, extname=binval_name)


def _dostats_main(config_file, flist, nrand, seed, outfile):
    import yaml
    import numpy as np

    rng = np.random.RandomState(seed)

    with open(config_file) as fobj:
        config = yaml.safe_load(fobj)

    allsums, allstats = _read_all_sums(config=config, flist=flist)

    allmeans = _do_all_bootstraps(
        allsums=allsums,
        nrand=nrand,
        rng=rng,
    )
    _write_means_output(outfile, allmeans)


def dostats_cli():
    args = _get_dostats_args()
    _dostats_main(
        config_file=args.config,
        flist=args.flist,
        seed=args.seed,
        nrand=args.nrand,
        outfile=args.output,
    )


#
# plot stats
#

def _get_plotstats_args():
    import argparse
    parser = argparse.ArgumentParser()
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
            print(key)
            allmeans[key] = hdu.read()

    return allmeans


def _doplot_g1g2_vs_bin_field(binval_name, means, outfront):
    import matplotlib.pyplot as mplt

    fig, ax = mplt.subplots(figsize=(10, 10 / 1.62))

    ax.set(
        xlabel=binval_name,
        ylabel=r'$g$',
        ylim=[-0.01, 0.01],
    )

    ax.errorbar(
        means['bin_field'][0],
        means['g1'][0],
        means['g1_err'][0],
        label=r'$g_1$',
    )
    ax.errorbar(
        means['bin_field'][0],
        means['g2'][0],
        means['g2_err'][0],
        label=r'$g_2$',
    )
    ax.axhline(0, color='black')
    ax.legend()

    outfile = outfront + f'{binval_name}.pdf'
    print('writing:', outfile)
    fig.savefig(outfile)
    mplt.close(fig)


def _plotstats_main(fname, outfront):
    print('reading:', fname)
    allmeans = _read_all_means(fname)

    for binval_name in allmeans:
        _doplot_g1g2_vs_bin_field(
            binval_name=binval_name,
            means=allmeans[binval_name],
            outfront=outfront,
        )


def plotstats_cli():
    args = _get_plotstats_args()
    _plotstats_main(fname=args.fname, outfront=args.outfront)
