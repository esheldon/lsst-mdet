# lsst-mdet

Metadetection and deblending measurement on LSST cell coadds,
with Gaia-driven bright-star subtraction and masking.

## Install

    pip install .

## Requirements

`requirements.txt` lists the packaged dependencies (most are on
conda-forge; ngmix is conda-forge only).  Two required packages
have no PyPI or conda packages yet and must be installed from
source:

- `metacal`
- `kdeblend`

The LSST science pipelines are required only for the
butler-facing modules (`lsst_mdet.cells`, `lsst_mdet.wcs`) and
the command-line tools; the rest of the package works without
them.  `dev-requirements.txt` adds the test/lint tooling for CI
(`pytest`, `ruff`).

## Command line

- `lsst-mdet-process-cells` — per-cell detection, deblending and
  (optionally) metacal over a patch; `--starsub` enables the
  Gaia star subtraction/masking, `--redo-bg` the background
  redetermination; `--cells 10,10 11,12` restricts to given cells
  for debugging
- `lsst-mdet-process-node` — run a job list of patches (`seed
  tract patch outfile` per line) on one node, `--nproc` at a
  time, with the same processing options.  One process imports
  the stack and warms the numba code once, then forks a child
  per patch, so a full node does not storm the file system with
  imports; each patch logs next to its outfile and the driver
  reports status, wall time and max RSS per patch
- `lsst-mdet-getimages` — extract patch images from the butler
  to FITS (image/var/mask/noise, per-cell psfs, wcs header),
  with the same `--starsub`/`--redo-bg` options
- `lsst-mdet-make-slurm` — slurm job generation for S3DF, one
  single-core job per patch
- `lsst-mdet-make-gaia` — per-tract Gaia DR3 star files (FITS)
  from the DM reference catalog in the butler, for
  `--gaia-pattern`; no network access needed.  Tracts at low
  galactic latitude (`--min-abs-b`, default |b| < 20) are
  skipped here and in the NERSC slurm maker
- `lsst-mdet-make-slurm-nersc` — slurm job generation for
  perlmutter at NERSC: whole-node jobs, each running
  `lsst-mdet-process-node` on a list of patches, with the gaia
  stars from the `lsst-mdet-make-gaia` files


## Package layout

Only `lsst_mdet.cells`, `lsst_mdet.wcs` and the `cli` modules
import the LSST stack; every other module (detection, star
subtraction, deblending, metacal, ...) is importable and
testable without it.
