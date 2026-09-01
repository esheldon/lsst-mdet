"""
make the total footprint map of a run: the union of the per-patch
footprint maps (<tract>/<tract>-<patch>-mdet-footprint.hsp, written
by lsst-mdet-process-cells beside the catalogs) as one bit-packed
healsparse map.

The map is streamed to disk one coverage pixel at a time, so memory
stays small however large the run, and the output is RICE
compressed by coverage block.  The default output is
<run-dir>/<run>-footprint.hsp

    lsst-mdet-make-footprint --run-dir . --nproc 16
"""
import os

DEFAULT_NPROC = 8


def get_footprint_file(run_dir):
    """
    the default output: <run-dir>/<run>-footprint.hsp
    """
    run = os.path.basename(os.path.abspath(run_dir))
    return os.path.join(run_dir, f'{run}-footprint.hsp')


def read_flist(fname):
    with open(fname) as fobj:
        return [line.strip() for line in fobj if line.strip()]


def go(args):
    from ..hmaps import cat_footprints, get_footprint_flist

    if args.flist is not None:
        flist = read_flist(args.flist)
    else:
        flist = get_footprint_flist(args.run_dir)

    if len(flist) == 0:
        raise RuntimeError(f'no footprint files found under {args.run_dir}')

    output = args.output
    if output is None:
        output = get_footprint_file(args.run_dir)

    print(f'{len(flist)} footprint files')
    print('writing:', output)
    res = cat_footprints(
        flist, output, nproc=args.nproc, clobber=args.clobber,
    )

    overlap = res['n_valid_inputs'] - res['n_valid']
    print(f'files: {res["nfile"]}')
    print(f'coverage pixels: {res["ncov"]}')
    print(f'footprint pixels: {res["n_valid"]}')
    print(f'overlapping input pixels: {overlap}')
    print(f'area: {res["area_deg2"]:.2f} deg^2')
    print(f'size on disk: {os.path.getsize(output) / 1024**2:.1f} MB')


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--run-dir', default='.',
                        help='the run directory holding the '
                             '{tract}/{tract}-{patch}-mdet-footprint.hsp '
                             'maps')
    parser.add_argument('--flist',
                        help='a file listing the footprint maps to '
                             'combine, instead of every map under '
                             'the run directory')
    parser.add_argument('--output',
                        help='output map; default '
                             '<run-dir>/<run>-footprint.hsp')
    parser.add_argument('--nproc', type=int, default=DEFAULT_NPROC,
                        help='processes for the coverage scan and '
                             'block reads')
    parser.add_argument('--clobber', action='store_true',
                        help='overwrite an existing output')

    args = parser.parse_args()
    if args.nproc < 1:
        parser.error('--nproc must be >= 1')
    return args


def main():
    go(get_args())


if __name__ == '__main__':
    main()
