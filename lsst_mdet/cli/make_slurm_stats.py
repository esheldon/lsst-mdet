"""
slurm job generation for the stats stage of a run: split the run's
catalog file list into per-core chunks, run lsst-mdet-dosums on the
chunks in parallel on one node, then lsst-mdet-dostats (with its
parallel bootstrap) on the chunk sums.

Everything lives under <run-dir>/stats, with paths relative to the
run directory; submit from there:

    stats/sums_config.yaml    the binning config (must exist)
    stats/flist.txt           every catalog file of the run,
                              regenerated at maker time
    stats/flists/flist-NNNN.txt   the per-process chunks
    stats/sums/sums-NNNN.fits     dosums output per chunk (log
                                  beside it)
    stats/stats.fits          the dostats output
    stats/run-stats.sh        runs the two stages
    stats/stats-mdet.slurm    the slurm script; debug QOS default

    cd <run-dir> && sbatch stats/stats-mdet.slurm
"""
import glob
import os

from .make_slurm_nersc import (
    DEFAULT_ACCOUNT,
    DEFAULT_CONSTRAINT,
    DEFAULT_NPROC,
    MAX_SEED,
)

DEFAULT_QOS = 'debug'
DEFAULT_WALLTIME = '00:30:00'
DEFAULT_NRAND = 1000

SCRIPT = r"""#!/usr/bin/bash
# run the stats stage: parallel dosums over the flist chunks, then
# dostats on the chunk sums
if [ $# -lt 1 ]; then
    echo "./run-stats.sh nproc"
    exit 1
fi

nproc=$1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p stats/sums

# one dosums per chunk, all at once (the chunk count is the nproc
# the maker was given)
for chunk in stats/flists/flist-*.txt; do
    num=$(basename ${chunk} .txt)
    num=${num#flist-}
    lsst-mdet-dosums \
        --config stats/sums_config.yaml \
        --flist ${chunk} \
        --output stats/sums/sums-${num}.fits \
        > stats/sums/sums-${num}.log 2>&1 &
done
wait

# a chunk writes its output only on success
fail=0
for chunk in stats/flists/flist-*.txt; do
    num=$(basename ${chunk} .txt)
    num=${num#flist-}
    if [ ! -e stats/sums/sums-${num}.fits ]; then
        echo "FAILED chunk ${num}: see stats/sums/sums-${num}.log"
        fail=1
    fi
done
if [ ${fail} -ne 0 ]; then
    exit 1
fi

lsst-mdet-dostats \
    --config stats/sums_config.yaml \
    --flist stats/sums/sums-*.fits \
    --seed %(stats_seed)d \
    --nrand %(nrand)d \
    --nproc ${nproc} \
    --output stats/stats.fits
"""

SLURM_TEMPLATE = r'''#!/bin/bash
#SBATCH --job-name=stats-mdet
#SBATCH --output stats/stats-mdet.log

#SBATCH --account=%(account)s
#SBATCH --qos=%(qos)s
#SBATCH --constraint=%(constraint)s

#SBATCH --nodes=1
#SBATCH --exclusive

#SBATCH --time=%(time)s

./stats/run-stats.sh %(nproc)d
'''


def get_flist(run_dir):
    """
    every per-patch catalog of the run, relative to the run
    directory, sorted
    """
    pattern = os.path.join(run_dir, '[0-9]*', '*-mdet.fits')
    flist = [os.path.relpath(f, run_dir) for f in glob.glob(pattern)]
    return sorted(flist)


def write_flists(stats_dir, flist, nchunk):
    """
    the full flist.txt and the nchunk contiguous chunks under
    flists/; stale chunks from an earlier, larger split are removed
    """
    fname = os.path.join(stats_dir, 'flist.txt')
    print(f'writing {len(flist)} files to {fname}')
    with open(fname, 'w') as fobj:
        for f in flist:
            fobj.write(f'{f}\n')

    chunk_dir = os.path.join(stats_dir, 'flists')
    os.makedirs(chunk_dir, exist_ok=True)
    for old in glob.glob(os.path.join(chunk_dir, 'flist-*.txt')):
        os.remove(old)

    # contiguous near-even chunks, no empties
    nchunk = min(nchunk, len(flist))
    lo = 0
    for i in range(nchunk):
        hi = lo + (len(flist) - lo) // (nchunk - i)
        with open(
            os.path.join(chunk_dir, f'flist-{i:04d}.txt'), 'w',
        ) as fobj:
            for f in flist[lo:hi]:
                fobj.write(f'{f}\n')
        lo = hi

    print(f'wrote {nchunk} chunks of about {len(flist) // nchunk} '
          f'files in {chunk_dir}/')
    return nchunk


def go(args):
    import numpy as np

    rng = np.random.RandomState(args.seed)

    stats_dir = os.path.join(args.run_dir, 'stats')
    config = os.path.join(stats_dir, 'sums_config.yaml')
    if not os.path.exists(config):
        raise RuntimeError(f'no config found: {config}')

    flist = get_flist(args.run_dir)
    if len(flist) == 0:
        raise RuntimeError(f'no catalogs found under {args.run_dir}')

    write_flists(stats_dir, flist, args.nproc)

    script = os.path.join(stats_dir, 'run-stats.sh')
    print('writing:', script)
    with open(script, 'w') as fobj:
        fobj.write(SCRIPT % {
            'stats_seed': rng.choice(MAX_SEED),
            'nrand': args.nrand,
        })
    os.chmod(script, 0o755)

    slurm_file = os.path.join(stats_dir, 'stats-mdet.slurm')
    print('writing:', slurm_file)
    with open(slurm_file, 'w') as fobj:
        fobj.write(SLURM_TEMPLATE % {
            'account': args.account,
            'qos': args.qos,
            'constraint': args.constraint,
            'time': args.walltime,
            'nproc': args.nproc,
        })

    print(f'submit from {args.run_dir or "."}: '
          f'sbatch stats/stats-mdet.slurm')


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--run-dir', default='.',
                        help='the run directory holding the '
                             '{tract}/{tract}-{patch}-mdet.fits '
                             'catalogs; stats go in its stats/ '
                             'subdirectory')
    parser.add_argument('--seed', type=int, required=True,
                        help='seed for the dostats bootstrap seed')
    parser.add_argument('--nproc', type=int, default=DEFAULT_NPROC,
                        help='dosums processes to run at once, and '
                             'the flist chunk count and the dostats '
                             'bootstrap workers')
    parser.add_argument('--nrand', type=int, default=DEFAULT_NRAND,
                        help='bootstrap realizations for dostats')
    parser.add_argument('--walltime', default=DEFAULT_WALLTIME)
    parser.add_argument('--qos', default=DEFAULT_QOS)
    parser.add_argument('--account', default=DEFAULT_ACCOUNT)
    parser.add_argument('--constraint', default=DEFAULT_CONSTRAINT)

    args = parser.parse_args()
    if args.nproc < 1:
        parser.error('--nproc must be >= 1')
    return args


def main():
    go(get_args())


if __name__ == '__main__':
    main()
