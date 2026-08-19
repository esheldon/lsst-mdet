# lsst-mdet

Metadetection and deblending measurement on LSST cell coadds,
with Gaia-driven bright-star subtraction and masking.

## Install

    pip install -e .

## Command line

- `lsst-mdet-process-cells` — per-cell detection, deblending and
  (optionally) metacal over a patch; `--starsub` enables the
  Gaia star subtraction/masking, `--redo-bg` the background
  redetermination
- `lsst-mdet-getimages` — extract patch images from the butler
  to FITS (image/var/mask/noise, per-cell psfs, wcs header),
  with the same `--starsub`/`--redo-bg` options
- `lsst-mdet-make-slurm` — slurm job generation

## Package layout

Only `lsst_mdet.cells`, `lsst_mdet.wcs` and the `cli` modules
import the LSST stack; every other module (detection, star
subtraction, deblending, metacal, ...) is importable and
testable without it.
