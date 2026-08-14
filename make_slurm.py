#!/usr/bin/env python

MAX_SEED = 2**30

SCRIPT = r"""#!/usr/bin/bash
if [ $# -lt 4 ]; then
    echo "./run.sh seed tract patch outfile"
    exit 1
fi

seed=$1
tract=$2
patch=$3
outfile=$4

export OMP_NUM_THREADS=1

python process_cells.py \
    --seed ${seed} \
    --tract ${tract} \
    --patch ${patch} \
    --outfile ${outfile} \
    --mdet
"""


SLURM_TEMPLATE = r'''#!/bin/bash
# Name of the job
#SBATCH --job-name=%(job_name)s

#SBATCH --output %(logfile)s

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

#SBATCH --mem-per-cpu=1.5G

# Walltime (job duration)
#SBATCH --time=%(time)s

#SBATCH --partition=milano
#SBATCH --account=rubin:default

./run.sh %(seed)s %(tract)d %(patch)s %(outfile)s
'''


def write_script():
    import os

    fname = 'run.sh'

    print('writing:', fname)
    with open(fname, 'w') as fobj:
        fobj.write(SCRIPT)

    os.system('chmod 755 %s' % fname)


def get_job_name(tract, patch):
    return f'{tract:05d}-{patch:05d}'


def get_outfile(tract, patch):
    import os

    job_name = get_job_name(tract=tract, patch=patch)
    fname = f'cat-{job_name}-mdet.fits'
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


def write_slurm(args, rng, good_cells):
    import os

    for good_cell in good_cells:
        tract = good_cell['tract']
        patch = good_cell['patch']

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
        }
        with open(slurm_file, 'w') as fobj:
            fobj.write(job_text)


def go(args):
    import numpy as np
    import rustfits

    write_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    good_cells = rustfits.read(args.good_cells)
    if args.njobs is not None:
        ri = rng.choice(good_cells.size, size=args.njobs, replace=False)
        good_cells = good_cells[ri]

    write_script()

    write_slurm(args=args, rng=rng, good_cells=good_cells)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--good-cells', default='good-cells.fits')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--njobs', type=int)
    parser.add_argument('--walltime', default='02:00:00',
                        help=('walltime for each job, e.g. 01:00:00'))

    return parser.parse_args()


def main():
    args = get_args()
    go(args)


if __name__ == '__main__':
    main()
