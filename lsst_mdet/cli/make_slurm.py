"""
slurm job generation for lsst-mdet-process-cells: one job per
patch.  A patch whose full 20x20 cell grid is good runs whole;
a partial patch gets --cells listing its good cells
"""
MAX_SEED = 2**30

# the processed cell grid: cells 1-20 in i and j (the border
# ring is not processed); a patch with all NCELL_FULL cells
# good needs no --cells argument
NCELL_SIDE = 20
NCELL_FULL = NCELL_SIDE ** 2

SCRIPT = r"""#!/usr/bin/bash
if [ $# -lt 4 ]; then
    echo "./run.sh seed tract patch outfile [cells...]"
    exit 1
fi

seed=$1
tract=$2
patch=$3
outfile=$4
shift 4

cells=""
if [ $# -gt 0 ]; then
    # a partial patch: process only the listed good cells
    cells="--cells $*"
fi

export OMP_NUM_THREADS=1

/usr/bin/time -v lsst-mdet-process-cells \
    --repo dp2_prep_future \
    --collections LSSTCam/runs/DRP/DP2 \
    --seed ${seed} \
    --tract ${tract} \
    --patch ${patch} \
    --redo-bg \
    --model exp \
    --deblend \
    --starsub%(extra)s \
    --outfile ${outfile} \
    --mdet ${cells}
"""

SLURM_TEMPLATE = r'''#!/bin/bash
# Name of the job
#SBATCH --job-name=%(job_name)s

#SBATCH --output %(logfile)s

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

# Walltime (job duration)
#SBATCH --time=%(time)s

#SBATCH --partition=milano
#SBATCH --account=rubin:default

./run.sh %(seed)s %(tract)d %(patch)s %(outfile)s%(cells)s
'''


def group_cells_by_patch(good_cells):
    """
    one entry per patch from the per-cell good list

    Parameters
    ----------
    good_cells: array with fields
        One row per good cell, with tract, patch, cell_i,
        cell_j (1-20)

    Returns
    -------
    list of dicts with tract, patch and cells, sorted by
    (tract, patch); cells is the list of good (i, j) or None
    when the patch has the full grid (no --cells needed)
    """
    import numpy as np

    # patch values are 0-99, so this key is unique
    key = good_cells['tract'] * 100 + good_cells['patch']
    s = np.argsort(key, kind='stable')
    _, starts = np.unique(key[s], return_index=True)

    bounds = list(starts) + [key.size]
    out = []
    for k in range(len(starts)):
        rows = good_cells[s[bounds[k]:bounds[k + 1]]]

        cells = None
        if rows.size < NCELL_FULL:
            cells = [
                (int(ci), int(cj))
                for ci, cj in zip(rows['cell_i'], rows['cell_j'])
            ]

        out.append({
            'tract': int(rows['tract'][0]),
            'patch': int(rows['patch'][0]),
            'cells': cells,
        })
    return out


def format_cells(cells):
    """
    the cells as trailing script arguments, ' i,j i,j ...';
    empty for None (a full patch)
    """
    if cells is None:
        return ''
    return ' ' + ' '.join(f'{i},{j}' for i, j in cells)


def write_script(extra=''):
    """
    Write run.sh, with any extra process-cells options appended.
    """
    import os

    fname = 'run.sh'

    print('writing:', fname)
    text = SCRIPT % {'extra': (' \\\n    ' + extra) if extra else ''}
    with open(fname, 'w') as fobj:
        fobj.write(text)

    os.system('chmod 755 %s' % fname)


def get_job_name(tract, patch):
    return f'{tract:05d}-{patch:05d}-mdet'


def get_outfile(tract, patch):
    import os

    job_name = get_job_name(tract=tract, patch=patch)
    fname = f'{job_name}.fits'
    tract_dir = get_tract_dir(tract)
    return os.path.join(tract_dir, fname)


def get_logfile(tract, patch):
    outfile = get_outfile(tract, patch)
    return outfile.replace('.fits', '.log')


def get_tract_dir(tract):
    return f'{tract:05d}'


def get_slurm_file(tract, patch):
    import os
    job_name = get_job_name(tract=tract, patch=patch)
    fname = f'{job_name}.slurm'
    tract_dir = get_tract_dir(tract)
    return os.path.join(tract_dir, fname)


def write_seed(seed):
    fname = 'seed.txt'
    print('writing seed file:', fname)
    with open(fname, 'w') as fobj:
        fobj.write(f'{seed}\n')


def write_slurm(args, rng, patch_jobs):
    import os

    for job in patch_jobs:
        tract = job['tract']
        patch = job['patch']

        job_seed = rng.choice(MAX_SEED)

        job_name = get_job_name(tract=tract, patch=patch)
        slurm_file = get_slurm_file(tract=tract, patch=patch)
        outfile = get_outfile(tract=tract, patch=patch)
        logfile = get_logfile(tract=tract, patch=patch)

        tract_dir = get_tract_dir(tract)

        if not os.path.exists(tract_dir):
            os.makedirs(tract_dir)

        job_text = SLURM_TEMPLATE % {
            'job_name': job_name,
            'logfile': logfile,
            'seed': job_seed,
            'time': args.walltime,
            'tract': tract,
            'patch': patch,
            'outfile': outfile,
            'cells': format_cells(job['cells']),
        }
        with open(slurm_file, 'w') as fobj:
            fobj.write(job_text)


def go(args):
    import numpy as np
    import rustfits

    write_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # one row per good cell; group to one job per patch
    with rustfits.FITS(args.good_cells) as fits:
        good_cells = fits[1].read(
            columns=['tract', 'patch', 'cell_i', 'cell_j'],
        )
    patch_jobs = group_cells_by_patch(good_cells)
    print(f'{good_cells.size} good cells in {len(patch_jobs)} patches')

    if args.tracts is not None:
        keep = set(args.tracts)
        patch_jobs = [j for j in patch_jobs if j['tract'] in keep]
        print(f'{len(patch_jobs)} patches in tracts {sorted(keep)}')

    if args.njobs is not None:
        ri = rng.choice(
            len(patch_jobs), size=args.njobs, replace=False,
        )
        patch_jobs = [patch_jobs[i] for i in sorted(ri)]

    write_script(extra=args.extra_args)

    write_slurm(args=args, rng=rng, patch_jobs=patch_jobs)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--good-cells', required=True,
                        help='the good-cells fits file (one row per '
                             'good cell); required so a stale '
                             'default cannot be picked up silently')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--njobs', type=int,
                        help='only generate jobs for this many '
                             'patches, chosen at random')
    parser.add_argument('--walltime', default='03:00:00',
                        help=('walltime for each job, e.g. 01:00:00'))
    parser.add_argument('--tracts', type=int, nargs='+',
                        help='only these tracts of the good-cells file')
    parser.add_argument('--extra-args', default='',
                        help='options appended to the process-cells '
                             'command in run.sh, e.g. "--gaia-pattern '
                             '... --starsub-method joint --wing-pattern ..."')

    return parser.parse_args()


def main():
    args = get_args()
    go(args)


if __name__ == '__main__':
    main()
