"""
shared DM mask-plane bits and processing flags
"""
from ngmix.flags import NO_ATTEMPT  # noqa

# flags set in the input mask plans
DM_NO_DATA = 1
DM_SAT = 2
DM_INTRP = 4
DM_DETECTION_EDGE = 16
DM_OUT = DM_NO_DATA | DM_DETECTION_EDGE

# processing flags
PSF_FAILURE = 2 ** 21
BAD_BBOX = 2 ** 22
ZERO_WEIGHTS = 2 ** 23
FLAG_EXTRA_DET_OFF_SEG = 2 ** 24
FLAG_NOT_CONVERGED = 2 ** 25
FLAG_DUPLICATE_EXTRA = 2 ** 26

MIN_GOOD_FRAC = 0.2

# cell geometry: inner-cell overlap for the processing window,
# and the lsst_cells_v2 inner cell size (used by the
# file-backed cell windows in patchfiles)
OVERLAP = 50
OVERLAP_LOW = 50
OVERLAP_HIGH = 200
CELL_SIZE = 150

SKYMAP_VERS = 'lsst_cells_v2'
