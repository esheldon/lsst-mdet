"""
slurm job generation for perlmutter at NERSC

The patches are packed into whole-node jobs, each running
lsst-mdet-process-node on a job list of patches.  That driver imports
the stack once per node and forks the per-patch workers, so a node
full of patches does not storm the file system with imports.

The layout matches lsst-mdet-make-slurm for the per-patch outputs

    seed.txt                          the seed used to generate the jobs
    run.sh                            runs the node driver on a job list
    nodejobs/nodejob-0000-mdet.txt    job list: seed tract patch outfile
    nodejobs/nodejob-0000-mdet.slurm  the slurm script for that node
    nodejobs/nodejob-0000-mdet.log    slurm output for that node
    {tract}/{tract}-{patch}-mdet.*    per-patch outputs and logs

The gaia stars come from per-tract files made by lsst-mdet-make-gaia,
see --gaia-pattern.  Patches with no gaia file are left out and listed
in missing-gaia.txt
"""
import os

from .make_gaia import GAIA_PATTERN
from .make_slurm import (
    MAX_SEED,
    get_outfile,
    write_seed,
)

# perlmutter cpu nodes have 128 physical cores (256 hyperthreads) and
# about 500 GB of memory; a patch needs about 1.5 GB
DEFAULT_NPROC = 128
DEFAULT_WALLTIME = '03:00:00'
DEFAULT_QOS = 'regular'
DEFAULT_CONSTRAINT = 'cpu'

SCRIPT = r"""#!/usr/bin/bash
if [ $# -lt 2 ]; then
    echo "./run.sh joblist nproc"
    exit 1
fi

joblist=$1
nproc=$2

export OMP_NUM_THREADS=1

lsst-mdet-process-node \
    --joblist ${joblist} \
    --nproc ${nproc} \
    --gaia-pattern '%(gaia_pattern)s' \
    --redo-bg \
    --model exp \
    --deblend \
    --starsub \
    --mdet
"""

SLURM_TEMPLATE = r'''#!/bin/bash
#SBATCH --job-name=%(job_name)s
#SBATCH --output %(logfile)s

#SBATCH --account=%(account)s
#SBATCH --qos=%(qos)s
#SBATCH --constraint=%(constraint)s

# one whole node; the driver forks the per-patch workers itself
#SBATCH --nodes=1
#SBATCH --exclusive

#SBATCH --time=%(time)s

./run.sh %(joblist)s %(nproc)d
'''


def write_script(gaia_pattern):
    fname = 'run.sh'

    print('writing:', fname)
    with open(fname, 'w') as fobj:
        fobj.write(SCRIPT % {'gaia_pattern': gaia_pattern})

    os.chmod(fname, 0o755)


def get_node_job_dir():
    return 'nodejobs'


def get_node_job_name(index):
    return f'nodejob-{index:04d}-mdet'


def get_node_job_file(index, ext):
    job_name = get_node_job_name(index)
    return os.path.join(get_node_job_dir(), f'{job_name}.{ext}')


def get_gaia_file(gaia_pattern, tract, patch):
    return gaia_pattern.format(tract=tract, patch=patch)


def select_with_gaia(patches, gaia_pattern):
    """
    keep the patches with a gaia file; the rest are written to
    missing-gaia.txt
    """
    import numpy as np

    has_gaia = np.array([
        os.path.exists(get_gaia_file(gaia_pattern, p['tract'], p['patch']))
        for p in patches
    ], dtype=bool)

    nmissing = (~has_gaia).sum()
    if nmissing == patches.size:
        raise RuntimeError(
            f'no gaia files found for any patch with pattern {gaia_pattern}'
        )

    if nmissing > 0:
        fname = 'missing-gaia.txt'
        print(f'WARNING: {nmissing} patches have no gaia file, '
              f'leaving them out; see {fname}')
        with open(fname, 'w') as fobj:
            for p in patches[~has_gaia]:
                fobj.write(f'{p["tract"]} {p["patch"]}\n')

    return patches[has_gaia]


def write_node_jobs(args, rng, patches):
    """
    write the job lists and slurm scripts, args.patches_per_node
    patches per node
    """
    node_dir = get_node_job_dir()
    os.makedirs(node_dir, exist_ok=True)

    per_node = args.patches_per_node

    nnodes = 0
    for start in range(0, patches.size, per_node):
        index = nnodes
        nnodes += 1

        joblist = get_node_job_file(index, 'txt')
        slurm_file = get_node_job_file(index, 'slurm')

        with open(joblist, 'w') as fobj:
            for p in patches[start:start + per_node]:
                tract = p['tract']
                patch = p['patch']
                seed = rng.choice(MAX_SEED)
                outfile = get_outfile(tract=tract, patch=patch)
                fobj.write(f'{seed} {tract} {patch} {outfile}\n')

        job_text = SLURM_TEMPLATE % {
            'job_name': get_node_job_name(index),
            'logfile': get_node_job_file(index, 'log'),
            'account': args.account,
            'qos': args.qos,
            'constraint': args.constraint,
            'time': args.walltime,
            'joblist': joblist,
            'nproc': args.nproc,
        }
        with open(slurm_file, 'w') as fobj:
            fobj.write(job_text)

    print(f'wrote {nnodes} node jobs for {patches.size} patches in '
          f'{node_dir}/ ({per_node} patches per node, {args.nproc} at a '
          f'time)')


def get_patches(good_cells):
    """
    the unique (tract, patch) pairs in the good cells list, which has
    one row per cell
    """
    import numpy as np

    patches = np.unique(good_cells[['tract', 'patch']])
    print(f'{good_cells.size} good cells in {patches.size} patches')
    return patches


def go(args):
    import numpy as np
    import rustfits

    write_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # one row per cell; only the patch ids are needed
    with rustfits.FITS(args.good_cells) as fits:
        good_cells = fits[1].read(columns=['tract', 'patch'])
    patches = get_patches(good_cells)

    if args.njobs is not None:
        ri = rng.choice(patches.size, size=args.njobs, replace=False)
        patches = patches[ri]

    patches = select_with_gaia(patches, args.gaia_pattern)

    write_script(args.gaia_pattern)
    write_node_jobs(args=args, rng=rng, patches=patches)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--good-cells', default='good-cells.fits')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--njobs', type=int,
                        help='only generate jobs for this many patches, '
                             'chosen at random')
    parser.add_argument('--account', required=True,
                        help='allocation to charge, e.g. des or m1727')
    parser.add_argument('--qos', default=DEFAULT_QOS)
    parser.add_argument('--constraint', default=DEFAULT_CONSTRAINT)
    parser.add_argument('--walltime', default=DEFAULT_WALLTIME,
                        help='walltime for each node job, e.g. 03:00:00. '
                             'This must cover the warmup and all the '
                             'waves of patches on the node')
    parser.add_argument('--nproc', type=int, default=DEFAULT_NPROC,
                        help='patches to run concurrently on each node')
    parser.add_argument('--patches-per-node', type=int,
                        help='total patches in each node job; default '
                             'is --nproc, a single wave')
    parser.add_argument('--gaia-pattern', default=GAIA_PATTERN,
                        help='gaia file pattern with {tract} and '
                             'optionally {patch} placeholders; default '
                             'is the lsst-mdet-make-gaia output')

    args = parser.parse_args()

    if args.nproc < 1:
        parser.error('--nproc must be >= 1')
    if args.patches_per_node is None:
        args.patches_per_node = args.nproc
    if args.patches_per_node < 1:
        parser.error('--patches-per-node must be >= 1')

    return args


def main():
    args = get_args()
    go(args)


if __name__ == '__main__':
    main()
