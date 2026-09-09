"""
run many patches on one node with tabulated residual-halo light
injected around the census stars: the amplified differential
injection test (lsst_mdet.inject).  All the node driver and
processing options of lsst-mdet-process-node apply; use the
production per-patch seeds in the job list so the output pairs
object-by-object against the production catalogs.

    lsst-mdet-inject-node --joblist jobs.txt --nproc 32 \\
        --model exp --redo-bg --deblend --starsub --mdet \\
        --inject-profiles prototype-seq.fits --inject-scale 5

The injection settings are recorded in the output provenance.
"""
import sys


def get_args():
    from .process_node import get_node_parser, validate_node_args
    from ..inject import DEFAULT_SCALE, INJECT_GMAX

    parser = get_node_parser()
    parser.description = (
        'run the patches in a job list on one node with residual '
        'halos injected around the census stars; see also '
        'lsst-mdet-process-node'
    )

    grp = parser.add_argument_group('injection options')
    grp.add_argument('--inject-profiles', required=True,
                     help='the tabulated residual profiles fits '
                          'file (the prototype-seq format)')
    grp.add_argument('--inject-scale', type=float,
                     default=DEFAULT_SCALE,
                     help='amplification of the injected field '
                          '(default %(default)s)')
    grp.add_argument('--inject-gmax', type=float,
                     default=INJECT_GMAX,
                     help='inject around stars brighter than this '
                          '(default %(default)s)')

    args = parser.parse_args()
    validate_node_args(parser, args)
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

    sys.exit(go(args))


if __name__ == '__main__':
    main()
