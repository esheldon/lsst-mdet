"""
run many patches on one node with an injection (lsst_mdet.inject),
one of two kinds:

- residual halos: tabulated residual-halo light around the census
  stars, the amplified differential injection test

    lsst-mdet-inject-node --joblist jobs.txt --nproc 32 \\
        --model exp --redo-bg --deblend --starsub --mdet \\
        --inject-profiles prototype-seq.fits --inject-scale 5

- objects: exponential galaxies and point sources from per-patch
  truth tables, added to the coadds as loaded, before any star or
  sky processing

    lsst-mdet-inject-node --joblist jobs.txt --nproc 32 \\
        --model exp --redo-bg --deblend --starsub --mdet \\
        --inject-objects 'truth/{tract:05d}-{patch:05d}-truth.fits'

All the node driver and processing options of lsst-mdet-process-node
apply; use the production per-patch seeds in the job list so the
output pairs object-by-object against the production catalogs.  The
injection settings are recorded in the output provenance.
"""
import sys


def get_args():
    from .process_node import get_node_parser, validate_node_args
    from ..inject import DEFAULT_SCALE, INJECT_GMAX

    parser = get_node_parser()
    parser.description = (
        'run the patches in a job list on one node with residual '
        'halos around the census stars or truth objects injected; '
        'see also lsst-mdet-process-node'
    )

    grp = parser.add_argument_group(
        'injection options (one of --inject-profiles or --inject-objects)'
    )
    grp.add_argument('--inject-profiles',
                     help='residual halos: the tabulated residual '
                          'profiles fits file (the prototype-seq '
                          'format)')
    grp.add_argument('--inject-scale', type=float,
                     default=DEFAULT_SCALE,
                     help='residual halos: amplification of the '
                          'injected field (default %(default)s)')
    grp.add_argument('--inject-gmax', type=float,
                     default=INJECT_GMAX,
                     help='residual halos: inject around stars '
                          'brighter than this (default %(default)s)')
    grp.add_argument('--inject-objects',
                     help='objects: the per-patch truth file pattern, '
                          'with {tract} and {patch} placeholders')

    args = parser.parse_args()
    validate_node_args(parser, args)
    if (args.inject_profiles is None) == (args.inject_objects is None):
        parser.error('give one of --inject-profiles or --inject-objects')
    if args.inject_objects is not None and args.patch_dir is not None:
        parser.error('object injection needs the butler load, not '
                     '--patch-dir')
    return args


def main():
    from ..inject import INJECT_SETTINGS
    from .process_node import go

    args = get_args()

    # set before the node driver forks, so every child inherits
    # the request (the METACAL_SETTINGS pattern)
    INJECT_SETTINGS['profiles'] = args.inject_profiles
    INJECT_SETTINGS['scale'] = args.inject_scale
    INJECT_SETTINGS['gmax'] = args.inject_gmax
    INJECT_SETTINGS['objects'] = args.inject_objects

    sys.exit(go(args))


if __name__ == '__main__':
    main()
