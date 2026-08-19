"""
shared DM mask-plane bits and processing flags
"""


# TRIM_TO_PRIMARY = True

# flags set in the input mask plans
DM_NO_DATA = 1


DM_DETECTION_EDGE = 16


DM_OUT = DM_NO_DATA | DM_DETECTION_EDGE


# processing flags
from ngmix.flags import NO_ATTEMPT  # noqa


PSF_FAILURE = 2 ** 21


BAD_BBOX = 2 ** 22


ZERO_WEIGHTS = 2 ** 23


FLAG_EXTRA_DET_OFF_SEG = 2 ** 24


FLAG_NOT_CONVERGED = 2 ** 25


FLAG_DUPLICATE_EXTRA = 2 ** 26


MIN_GOOD_FRAC = 0.2


OVERLAP = 50


OVERLAP_LOW = 50


OVERLAP_HIGH = 200


SKYMAP_VERS = 'lsst_cells_v2'


DM_SAT = 2


DM_INTRP = 4


# lsst_cells_v2 inner cell size, the fallback when the coadd's
# own cell grid cannot be introspected
CELL_SIZE = 150
