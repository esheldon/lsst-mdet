"""
cli/process_node

run many patches on one node from a single process.  The parent imports
everything once (process_cells.preload) and warms the numba compiled
code by processing a few cells of the first patch, then forks one child
per patch, --nproc at a time.  The children inherit the loaded modules
and compiled code and so do no imports at all, which keeps a node full
of patch jobs from hammering the shared file system.

The job list has one patch per line

    seed tract patch outfile

with the log for each patch written next to its outfile, with a .log
extension.  Any processing options are given on the command line and
apply to every patch, e.g.

    lsst-mdet-process-node --joblist jobs.txt --nproc 128 \\
        --model exp --redo-bg --deblend --starsub --mdet

Children are reaped with wait4, so a child killed by a signal (segfault,
OOM) is reported rather than hanging the run, and the maximum resident
set size of every child is reported for checking memory and contention.
"""
import os
import sys
import time

from .process_cells import get_parser, parse_cell, preload, process_patch

DEFAULT_WARMUP_CELLS = [(10, 10), (11, 11)]


def read_joblist(fname):
    """
    read the job list, one 'seed tract patch outfile [cells...]' per
    line, where the optional trailing 'i,j' cells restrict a partial
    patch to its good cells.  Blank lines and lines starting with #
    are skipped
    """
    jobs = []
    with open(fname) as fobj:
        for line in fobj:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue

            fields = line.split()
            if len(fields) < 4:
                raise ValueError(
                    f'expected "seed tract patch outfile [cells...]", '
                    f'got {line!r}'
                )

            seed, tract, patch, outfile = fields[:4]
            cells = [parse_cell(f) for f in fields[4:]]
            jobs.append({
                'seed': int(seed),
                'tract': int(tract),
                'patch': int(patch),
                'outfile': outfile,
                'logfile': get_logfile(outfile),
                'cells': cells if len(cells) > 0 else None,
            })

    return jobs


def get_logfile(outfile):
    root, ext = os.path.splitext(outfile)
    return root + '.log'


def get_job_args(args, job, outfile=None, cells=None):
    """
    the per-patch namespace: the common processing options plus this
    job's seed, tract, patch and outfile.  The cells override (used
    by the warmup) wins over the job's own cells (a partial patch
    from the job list), which wins over args.cells
    """
    import argparse

    job_args = argparse.Namespace(**vars(args))
    job_args.seed = job['seed']
    job_args.tract = job['tract']
    job_args.patch = job['patch']
    job_args.outfile = job['outfile'] if outfile is None else outfile
    if cells is not None:
        job_args.cells = cells
    elif job.get('cells') is not None:
        job_args.cells = job['cells']
    return job_args


def log(message):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'{stamp} {message}', flush=True)


def get_rss_gb():
    """
    the current resident set size of this process in GB, from
    /proc/self/statm (pages)
    """
    with open('/proc/self/statm') as fobj:
        resident_pages = int(fobj.read().split()[1])
    return resident_pages * os.sysconf('SC_PAGE_SIZE') / 1024**3


def warmup(args, job):
    """
    import everything and process a few cells of the first patch in
    this process, so the forked children inherit the imports and the
    numba compiled code.  The output goes to a temporary directory
    """
    import tempfile
    import traceback

    log(f'preloading with tract {job["tract"]} patch {job["patch"]}')
    preload(
        repo=args.repo,
        collections=args.collections,
        tract=job['tract'],
        patch=job['patch'],
        patch_dir=args.patch_dir,
    )

    cells = ' '.join(f'{i},{j}' for i, j in args.warmup_cells)
    log(f'warming up on cells {cells}')

    with tempfile.TemporaryDirectory(prefix='lsst-mdet-warmup-') as tmpdir:
        outfile = os.path.join(tmpdir, os.path.basename(job['outfile']))
        job_args = get_job_args(
            args, job, outfile=outfile, cells=args.warmup_cells,
        )
        try:
            process_patch(job_args)
        except Exception:
            # not fatal: the children will do the remaining imports
            # and compilation themselves
            traceback.print_exc()
            log('WARNING: warmup failed, continuing without it')


def limit_blas_threads():
    """
    limit the BLAS thread pools to one thread, for this process and
    the forked children, and make ngmix's per-call limiter reuse the
    controller made here.

    ngmix.util.single_core_blas wraps its dense linear algebra in
    threadpoolctl.threadpool_limits, which discovers the loaded BLAS
    libraries by walking the shared libraries with a ctypes callback.
    ctypes callbacks are libffi closures, allocated in executable
    memory that on this system is a MAP_SHARED double mapping of a
    temporary file.  Forked children share that mapping, each with a
    private copy of the allocator state, so children creating closures
    concurrently overwrite each other's trampolines and die with
    segfaults or aborts inside dl_iterate_phdr.  The discovery is done
    once here, before forking; the children then only call
    set_num_threads on the already discovered libraries and create no
    closures.

    Returns the controller, which must be kept alive
    """
    try:
        import threadpoolctl
    except ImportError:
        log('threadpoolctl not available, not limiting BLAS threads')
        return None

    controller = threadpoolctl.ThreadpoolController()
    limiter = controller.limit(limits=1, user_api='blas')
    log('limited BLAS threads to 1 for '
        f'{[c.internal_api for c in controller.lib_controllers]}')

    try:
        import ngmix.util
    except ImportError:
        return controller, limiter

    def single_core_blas():
        return controller.limit(limits=1, user_api='blas')

    ngmix.util.single_core_blas = single_core_blas
    return controller, limiter


def run_child(args, job):
    """
    the child process: send all output to the log file, process the
    patch and exit.  This never returns; os._exit is used so that the
    parent's exit handlers and buffers are not run or flushed twice
    """
    import faulthandler
    import traceback

    fd = os.open(job['logfile'], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    # line buffered so the log is complete even if the process dies
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    # a python traceback in the log on a segfault or abort
    faulthandler.enable(file=sys.stderr, all_threads=True)

    status = 0
    try:
        process_patch(get_job_args(args, job))
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)


def describe_status(status):
    """
    describe a wait status: ok, exit code, or the signal
    """
    if os.WIFSIGNALED(status):
        return f'killed by signal {os.WTERMSIG(status)}'
    code = os.WEXITSTATUS(status)
    if code == 0:
        return 'ok'
    return f'exit {code}'


def run_jobs(args, jobs):
    """
    fork one child per job, args.nproc at a time, reaping with wait4.
    Returns the list of failed jobs
    """
    pending = list(jobs)
    running = {}
    failed = []
    ndone = 0

    while pending or running:
        while pending and len(running) < args.nproc:
            job = pending.pop(0)

            outdir = os.path.dirname(job['outfile'])
            if outdir != '':
                os.makedirs(outdir, exist_ok=True)

            sys.stdout.flush()
            sys.stderr.flush()
            pid = os.fork()
            if pid == 0:
                run_child(args, job)

            running[pid] = (job, time.time())
            log(f'started tract {job["tract"]} patch {job["patch"]} '
                f'pid {pid} running {len(running)} pending {len(pending)}')

            # spread the starts so a node's loads (each opening a
            # butler registry connection and reading its coadds) do
            # not all hit the database and file system at once
            if pending and len(running) < args.nproc:
                time.sleep(args.start_interval)

        pid, status, rusage = os.wait4(-1, 0)
        job, t0 = running.pop(pid)
        ndone += 1

        elapsed = time.time() - t0
        # ru_maxrss is in kB on linux
        maxrss_gb = rusage.ru_maxrss / 1024**2
        desc = describe_status(status)

        log(f'finished tract {job["tract"]} patch {job["patch"]} '
            f'{desc} elapsed {elapsed / 60:.1f} min '
            f'maxrss {maxrss_gb:.2f} GB '
            f'cpu {rusage.ru_utime / 60:.1f} min '
            f'done {ndone}/{len(jobs)}')

        if desc != 'ok':
            failed.append((job, desc))

    return failed


def go(args):
    # the slurm log should show progress as it happens
    sys.stdout.reconfigure(line_buffering=True)

    jobs = read_joblist(args.joblist)
    log(f'read {len(jobs)} jobs from {args.joblist}')

    if args.skip_existing:
        jobs = [job for job in jobs if not os.path.exists(job['outfile'])]
        log(f'{len(jobs)} jobs remain after skipping existing outputs')

    if len(jobs) == 0:
        log('nothing to do')
        return 0

    t0 = time.time()
    if not args.no_warmup:
        warmup(args, jobs[0])
        log(f'warmup took {time.time() - t0:.1f} s')

    # must stay referenced while the children run
    blas = limit_blas_threads()  # noqa: F841

    # the children inherit these pages copy-on-write; their reported
    # maxrss includes them, so this is the baseline to subtract
    log(f'parent rss before forking {get_rss_gb():.2f} GB')

    log(f'running {len(jobs)} jobs with nproc {args.nproc}')
    failed = run_jobs(args, jobs)

    log(f'all jobs finished in {(time.time() - t0) / 60:.1f} min: '
        f'{len(jobs) - len(failed)} ok, {len(failed)} failed')
    for job, desc in failed:
        log(f'FAILED tract {job["tract"]} patch {job["patch"]} {desc} '
            f'see {job["logfile"]}')

    return 1 if failed else 0


def get_node_parser():
    """
    the node driver parser: the process_cells options (per_patch
    disabled) plus the node driver group.  Split out so wrapper
    CLIs (lsst-mdet-inject-node) can extend it before parsing
    """
    from .process_cells import parse_cell

    parser = get_parser(per_patch=False)
    parser.description = (
        'run the patches in a job list on one node; see also '
        'lsst-mdet-process-cells for the processing options'
    )

    node = parser.add_argument_group('node driver options')
    node.add_argument('--joblist', required=True,
                      help='file with one "seed tract patch outfile" '
                           'per line')
    node.add_argument('--nproc', type=int, required=True,
                      help='number of patches to process concurrently')
    node.add_argument('--start-interval', type=float, default=0.5,
                      help='seconds between starting patches, to spread '
                           'the load stage (butler registry connections, '
                           'coadd reads) over time; 0 to start them as '
                           'fast as cores free up')
    node.add_argument('--skip-existing', action='store_true',
                      help='skip patches whose outfile already exists')
    node.add_argument('--no-warmup', action='store_true',
                      help='skip the preload and warmup; the children '
                           'then import and compile independently')
    node.add_argument('--warmup-cells', nargs='+', type=parse_cell,
                      default=DEFAULT_WARMUP_CELLS,
                      help='cells of the first patch to process for '
                           'the warmup, default 10,10 11,11')
    return parser


def validate_node_args(parser, args):
    if args.nproc < 1:
        parser.error('--nproc must be >= 1')


def get_args():
    parser = get_node_parser()
    args = parser.parse_args()
    validate_node_args(parser, args)
    return args


def main():
    args = get_args()
    sys.exit(go(args))


if __name__ == '__main__':
    main()
