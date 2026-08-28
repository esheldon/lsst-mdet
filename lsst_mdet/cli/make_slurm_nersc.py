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
                                      [cells...], the trailing i,j
                                      cells restricting a partial
                                      patch to its good cells
    nodejobs/nodejob-0000-mdet.slurm  the slurm script for that node
    nodejobs/nodejob-0000-mdet.log    slurm output for that node
    {tract}/{tract}-{patch}-mdet.*    per-patch outputs and logs

The gaia stars come from per-tract files made by lsst-mdet-make-gaia,
see --gaia-pattern.  Tracts at low galactic latitude are left out as
in lsst-mdet-make-gaia (--min-abs-b) and listed in low-latitude.txt;
patches with no gaia file are left out and listed in missing-gaia.txt
"""
import os

from ..defaults import BUTLER_COLLECTIONS, BUTLER_REPO, SKYMAP_VERS
from .make_gaia import DEFAULT_MIN_ABS_B, GAIA_PATTERN, select_high_latitude
from .make_slurm import (
    MAX_SEED,
    format_cells,
    get_outfile,
    group_cells_by_patch,
    write_seed,
)

# perlmutter cpu nodes have 128 physical cores (256 hyperthreads) and
# about 500 GB of memory; a patch needs about 1.5 GB
DEFAULT_NPROC = 128
DEFAULT_WALLTIME = '03:00:00'
DEFAULT_QOS = 'regular'
DEFAULT_CONSTRAINT = 'cpu'
DEFAULT_ACCOUNT = 'm1727'

SCRIPT = r"""#!/usr/bin/bash
if [ $# -lt 2 ]; then
    echo "./run.sh joblist nproc"
    exit 1
fi

joblist=$1
nproc=$2

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

lsst-mdet-process-node \
    --joblist ${joblist} \
    --nproc ${nproc} \
    --gaia-pattern '%(gaia_pattern)s' \
    --redo-bg \
    --model exp \
    --deblend \
    --starsub%(mdet)s
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


def write_script(gaia_pattern, mdet=True):
    fname = 'run.sh'

    print('writing:', fname)
    with open(fname, 'w') as fobj:
        fobj.write(SCRIPT % {
            'gaia_pattern': gaia_pattern,
            'mdet': ' \\\n    --mdet' if mdet else '',
        })

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


def select_high_latitude_patches(args, patches):
    """
    drop the patches in tracts below the galactic latitude cut; those
    tracts are written to low-latitude.txt
    """
    import numpy as np

    if args.min_abs_b <= 0:
        return patches

    from lsst.daf.butler import Butler

    butler = Butler(args.repo, collections=args.collections)
    skymap = butler.get('skyMap', skymap=SKYMAP_VERS)

    tracts = np.unique(patches['tract']).tolist()
    _, dropped = select_high_latitude(skymap, tracts, args.min_abs_b)

    if len(dropped) == 0:
        return patches

    keep = ~np.isin(patches['tract'], dropped)
    fname = 'low-latitude.txt'
    print(f'leaving out {(~keep).sum()} patches in {len(dropped)} tracts '
          f'with galactic latitude |b| < {args.min_abs_b:g}; see {fname}')
    with open(fname, 'w') as fobj:
        for tract in dropped:
            fobj.write(f'{tract}\n')

    return patches[keep]


def select_in_box(args, good_cells, patch_jobs):
    """
    keep the patches with a good cell centered in the --ra-range
    and --dec-range box; all of them when neither is given
    """
    import numpy as np

    if args.ra_range is None and args.dec_range is None:
        return patch_jobs

    keep = np.ones(good_cells.size, dtype=bool)
    if args.ra_range is not None:
        keep &= (
            (good_cells['ra_center'] >= args.ra_range[0])
            & (good_cells['ra_center'] <= args.ra_range[1])
        )
    if args.dec_range is not None:
        keep &= (
            (good_cells['dec_center'] >= args.dec_range[0])
            & (good_cells['dec_center'] <= args.dec_range[1])
        )

    in_box = set(zip(
        good_cells['tract'][keep].tolist(),
        good_cells['patch'][keep].tolist(),
    ))
    selected = [
        j for j in patch_jobs if (j['tract'], j['patch']) in in_box
    ]
    print(f'{len(selected)} patches with a good cell in the box '
          f'ra {args.ra_range} dec {args.dec_range}')
    return selected


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


def write_node_jobs(args, rng, patch_jobs):
    """
    write the job lists and slurm scripts, args.patches_per_node
    patches per node.  A partial patch's job line carries its good
    cells as trailing i,j fields
    """
    node_dir = get_node_job_dir()
    os.makedirs(node_dir, exist_ok=True)

    per_node = args.patches_per_node

    nnodes = 0
    for start in range(0, len(patch_jobs), per_node):
        index = nnodes
        nnodes += 1

        joblist = get_node_job_file(index, 'txt')
        slurm_file = get_node_job_file(index, 'slurm')

        with open(joblist, 'w') as fobj:
            for job in patch_jobs[start:start + per_node]:
                tract = job['tract']
                patch = job['patch']
                seed = rng.choice(MAX_SEED)
                outfile = get_outfile(tract=tract, patch=patch)
                cells = format_cells(job['cells'])
                fobj.write(f'{seed} {tract} {patch} {outfile}{cells}\n')

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

    print(f'wrote {nnodes} node jobs for {len(patch_jobs)} patches in '
          f'{node_dir}/ ({per_node} patches per node, {args.nproc} at a '
          f'time)')


def go(args):
    import numpy as np
    import rustfits

    write_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # one row per good cell; group to one job per patch, with the
    # good cells kept for the partial patches
    with rustfits.FITS(args.good_cells) as fits:
        good_cells = fits[1].read(
            columns=[
                'tract', 'patch', 'cell_i', 'cell_j',
                'ra_center', 'dec_center',
            ],
        )
    patch_jobs = group_cells_by_patch(good_cells)
    print(f'{good_cells.size} good cells in {len(patch_jobs)} patches')

    patch_jobs = select_in_box(args, good_cells, patch_jobs)

    # the selectors work on a plain (tract, patch) array; map back
    # to the grouped jobs afterward
    jobmap = {(j['tract'], j['patch']): j for j in patch_jobs}
    patches = np.zeros(
        len(patch_jobs), dtype=[('tract', 'i8'), ('patch', 'i8')],
    )
    patches['tract'] = [j['tract'] for j in patch_jobs]
    patches['patch'] = [j['patch'] for j in patch_jobs]

    patches = select_high_latitude_patches(args, patches)

    if args.njobs is not None:
        ri = rng.choice(patches.size, size=args.njobs, replace=False)
        patches = patches[ri]

    patches = select_with_gaia(patches, args.gaia_pattern)

    patch_jobs = [
        jobmap[(int(p['tract']), int(p['patch']))] for p in patches
    ]

    write_script(args.gaia_pattern, mdet=not args.no_mdet)
    write_node_jobs(args=args, rng=rng, patch_jobs=patch_jobs)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--good-cells', required=True,
                        help='the good-cells fits file (one row per '
                             'good cell); required so a stale '
                             'default cannot be picked up silently')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--njobs', type=int,
                        help='only generate jobs for this many patches, '
                             'chosen at random')
    parser.add_argument('--ra-range', type=float, nargs=2,
                        metavar=('RAMIN', 'RAMAX'),
                        help='only patches with a good cell whose center '
                             'is in this ra range (degrees)')
    parser.add_argument('--dec-range', type=float, nargs=2,
                        metavar=('DECMIN', 'DECMAX'),
                        help='only patches with a good cell whose center '
                             'is in this dec range (degrees)')
    parser.add_argument('--account', default=DEFAULT_ACCOUNT,
                        help='allocation to charge')
    parser.add_argument('--qos', default=DEFAULT_QOS)
    parser.add_argument('--constraint', default=DEFAULT_CONSTRAINT)
    parser.add_argument('--walltime', default=DEFAULT_WALLTIME,
                        help='walltime for each node job, e.g. 03:00:00. '
                             'This must cover the warmup and all the '
                             'waves of patches on the node')
    parser.add_argument('--nproc', type=int, default=DEFAULT_NPROC,
                        help='patches to run concurrently on each node')
    parser.add_argument('--no-mdet', action='store_true',
                        help='leave out --mdet, e.g. for a quick test on '
                             'the debug QOS: a patch then takes under 10 '
                             'minutes instead of 30-40')
    parser.add_argument('--patches-per-node', type=int,
                        help='total patches in each node job; default '
                             'is --nproc, a single wave')
    parser.add_argument('--gaia-pattern', default=GAIA_PATTERN,
                        help='gaia file pattern with {tract} and '
                             'optionally {patch} placeholders; default '
                             'is the lsst-mdet-make-gaia output')
    parser.add_argument('--min-abs-b', type=float, default=DEFAULT_MIN_ABS_B,
                        help='leave out tracts with center galactic '
                             'latitude |b| below this, in degrees, as '
                             'lsst-mdet-make-gaia does; 0 to keep all')
    parser.add_argument('--repo', default=BUTLER_REPO,
                        help='butler repo, for the skymap')
    parser.add_argument('--collections', nargs='+',
                        default=BUTLER_COLLECTIONS)

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
