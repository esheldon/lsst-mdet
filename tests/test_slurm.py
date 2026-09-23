"""
the slurm makers: one job per patch, with partial patches
carrying their good cells as --cells arguments
"""
import numpy as np
import pytest

from lsst_mdet.cli.make_slurm import (
    NCELL_FULL,
    NCELL_SIDE,
    SCRIPT,
    SLURM_TEMPLATE,
    format_cells,
    group_cells_by_patch,
)


def make_good_cells(patch_cells):
    """
    a synthetic good-cells array from
    {(tract, patch): [(i, j), ...]}
    """
    rows = []
    for (tract, patch), cells in patch_cells.items():
        for ci, cj in cells:
            rows.append((tract, patch, ci, cj))
    arr = np.array(rows, dtype=[
        ('tract', 'i4'), ('patch', 'i4'),
        ('cell_i', 'i2'), ('cell_j', 'i2'),
    ])
    # scramble: grouping must not rely on input order
    rng = np.random.RandomState(5)
    return arr[rng.permutation(arr.size)]


def full_grid():
    return [
        (i, j)
        for i in range(1, NCELL_SIDE + 1)
        for j in range(1, NCELL_SIDE + 1)
    ]


def test_group_cells_by_patch():
    partial = [(1, 1), (10, 12), (20, 20)]
    good_cells = make_good_cells({
        (7235, 86): full_grid(),
        (2877, 72): partial,
        (7235, 12): [(5, 5)],
    })

    jobs = group_cells_by_patch(good_cells)

    # sorted by (tract, patch)
    assert [(j['tract'], j['patch']) for j in jobs] == [
        (2877, 72), (7235, 12), (7235, 86),
    ]

    # the full patch runs whole
    assert jobs[2]['cells'] is None

    # partial patches carry exactly their good cells
    assert sorted(jobs[0]['cells']) == sorted(partial)
    assert jobs[1]['cells'] == [(5, 5)]

    assert len(full_grid()) == NCELL_FULL


def test_format_cells():
    assert format_cells(None) == ''
    assert format_cells([(1, 2), (10, 20)]) == ' 1,2 10,20'


def test_slurm_template_cells():
    text = SLURM_TEMPLATE % {
        'job_name': 'j', 'logfile': 'l', 'seed': 1,
        'time': '01:00:00', 'tract': 7235, 'patch': 86,
        'outfile': 'out.fits',
        'cells': format_cells([(1, 2), (3, 4)]),
    }
    assert './run.sh 1 7235 86 out.fits 1,2 3,4' in text

    text = SLURM_TEMPLATE % {
        'job_name': 'j', 'logfile': 'l', 'seed': 1,
        'time': '01:00:00', 'tract': 7235, 'patch': 86,
        'outfile': 'out.fits',
        'cells': format_cells(None),
    }
    assert './run.sh 1 7235 86 out.fits\n' in text

    # the run script forwards the trailing cells
    assert '--cells $*' in SCRIPT
    assert 'shift 4' in SCRIPT


def test_read_joblist(tmp_path):
    from lsst_mdet.cli.process_node import read_joblist

    fname = str(tmp_path / 'jobs.txt')
    with open(fname, 'w') as fobj:
        fobj.write('# comment\n')
        fobj.write('11 7235 86 out1.fits\n')
        fobj.write('12 2877 72 out2.fits 1,2 10,20\n')

    jobs = read_joblist(fname)
    assert len(jobs) == 2

    assert jobs[0]['cells'] is None
    assert jobs[1]['cells'] == [(1, 2), (10, 20)]
    assert jobs[1]['seed'] == 12

    # bad cells are rejected at read time
    with open(fname, 'w') as fobj:
        fobj.write('11 7235 86 out1.fits 0,5\n')
    with pytest.raises(Exception):
        read_joblist(fname)


def test_job_cells_reach_args(tmp_path):
    from lsst_mdet.cli.process_node import get_job_args
    import argparse

    args = argparse.Namespace(cells=None, seed=0, tract=0,
                              patch=0, outfile='')
    job = {
        'seed': 3, 'tract': 7235, 'patch': 86,
        'outfile': 'o.fits', 'cells': [(1, 2)],
    }
    job_args = get_job_args(args, job)
    assert job_args.cells == [(1, 2)]

    # the warmup override wins
    job_args = get_job_args(args, job, cells=[(9, 9)])
    assert job_args.cells == [(9, 9)]

    # no cells: args.cells stands
    job['cells'] = None
    job_args = get_job_args(args, job)
    assert job_args.cells is None


def test_load_slots(tmp_path):
    """
    the node driver's load cap: nslots holders at once, the next
    waits until one releases; a lock is per open descriptor, so one
    process can hold and contend for the slots itself
    """
    from lsst_mdet.cli.process_cells import LoadSlots

    slots = LoadSlots(str(tmp_path), 2)
    assert len(list(tmp_path.glob('slot-*'))) == 2

    with slots.acquire(timeout=1) as waited:
        assert waited < 1
        with slots.acquire(timeout=1):
            # both taken: a third asks and times out
            with pytest.raises(TimeoutError):
                with slots.acquire(poll=0.05, timeout=0.3):
                    pass
        # one released: the wait ends
        with slots.acquire(poll=0.05, timeout=1) as waited:
            assert waited < 1

    with pytest.raises(ValueError):
        LoadSlots(str(tmp_path), 0)


def test_write_script_node_options(tmp_path, monkeypatch):
    """
    --max-loads and --galaxy-file reach run.sh, as continued option
    lines of the node driver call, and leave it unchanged when absent
    """
    from lsst_mdet.cli.make_slurm_nersc import (
        get_node_options, write_script,
    )

    assert get_node_options() == ''
    monkeypatch.chdir(tmp_path)
    kw = dict(starsub_method='joint', wing_pattern='wing-{band}.fits')
    write_script('gaia-{tract:05d}.fits', 'AZGauss', **kw)
    plain = (tmp_path / 'run.sh').read_text()
    assert '--max-loads' not in plain and '--galaxy-file' not in plain

    write_script('gaia-{tract:05d}.fits', 'AZGauss', max_loads=8,
                 galaxy_file='/some/leda.fits', **kw)
    text = (tmp_path / 'run.sh').read_text()
    assert "    --max-loads 8 \\\n" in text
    assert "    --galaxy-file '/some/leda.fits' \\\n" in text
    # the two lines are the only difference
    lines = [ln for ln in text.splitlines()
             if '--max-loads' not in ln and '--galaxy-file' not in ln]
    assert lines == plain.splitlines()
    # every option line but the last is continued
    body = text[text.index('lsst-mdet-process-node'):].rstrip().splitlines()
    assert all(ln.endswith('\\') for ln in body[:-1])
    assert not body[-1].endswith('\\')


def test_mcal_types_option(tmp_path, monkeypatch):
    """
    --mcal-types is validated and ordered, sets the metacal types
    for the processing, and reaches run.sh only when not the default
    """
    import argparse

    import pytest

    from lsst_mdet.cli.make_slurm_nersc import get_node_options, write_script
    from lsst_mdet.cli.process_cells import (
        DEFAULT_MCAL_TYPES, parse_mcal_types,
    )

    assert parse_mcal_types('2m,1p,noshear,1m,2p') == 'noshear,1p,1m,2p,2m'
    assert parse_mcal_types(' noshear, 1p ,1m') == DEFAULT_MCAL_TYPES
    for bad in ('1p,1m', 'noshear,3p', ''):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_mcal_types(bad)

    assert get_node_options(mcal_types=DEFAULT_MCAL_TYPES) == ''
    assert '--mcal-types noshear,1p,1m,2p,2m' in get_node_options(
        mcal_types='noshear,1p,1m,2p,2m',
    )
    monkeypatch.chdir(tmp_path)
    write_script('gaia-{tract:05d}.fits', 'AZGauss',
                 mcal_types='noshear,1p,1m,2p,2m')
    text = (tmp_path / 'run.sh').read_text()
    assert "    --mcal-types noshear,1p,1m,2p,2m \\\n" in text
