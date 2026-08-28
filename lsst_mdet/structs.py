"""
output catalog and cell-info structure definitions
"""
import numpy as np
from .defaults import NO_ATTEMPT


def get_struct(bands, n=1):
    """
    Get the output structure, shared by the single-object
    fitters (am, gauss, wmom) and the kdeblend deblender

    Parameters
    ----------
    bands: sequence of str
        The band names; flux_{band} and flux_err_{band} columns are
        created for each
    n: int, optional
        The number of rows, default 1
    """
    dtype = [
        ('mcal_step', 'U2'),
        ('cell_i', 'i2'),
        ('cell_j', 'i2'),
        ('flags', 'i4'),
        ('extra_det', bool),
        ('is_primary', bool),
        ('numiter', 'i2'),
        ('obj_id', 'i4'),
        ('group_size', 'i2'),
        ('group_id', 'i4'),
        ('deblend_flags', 'i4'),
        ('mfrac', 'f4'),

        ('xcell', 'f4'),
        ('ycell', 'f4'),
        ('x', 'f4'),
        ('y', 'f4'),
        ('x_fit', 'f4'),
        ('y_fit', 'f4'),
        ('ra', 'f8'),
        ('dec', 'f8'),

        ('s2n', 'f4'),
        ('g_flags', 'i4'),
        ('g1', 'f4'),
        ('g1_err', 'f4'),
        ('g2', 'f4'),
        ('g2_err', 'f4'),
        ('T', 'f4'),
        ('T_err', 'f4'),
    ]

    for band in bands:
        dtype += [
            (f'flux_{band}', 'f4'),
            (f'flux_err_{band}', 'f4'),
        ]

    nband = len(bands)
    for i in range(nband - 1):
        first_band = bands[i]
        second_band = bands[i + 1]

        cname = f'{first_band}m{second_band}'

        dtype += [
            (cname, 'f4'),
            (f'{cname}_err', 'f4'),
        ]

    for band in bands:
        dtype += [
            (f'psf_flags_{band}', 'i4'),
            (f'psf_T_{band}', 'f4'),
        ]

    dtype += [
        ('psf_flags', 'i4'),
        ('psf_T', 'f4'),
    ]

    for band in bands:
        dtype += [
            (f'psfrec_flags_{band}', 'i4'),
            (f'psfrec_T_{band}', 'f4'),
            (f'psfrec_g1_{band}', 'f4'),
            (f'psfrec_g2_{band}', 'f4'),
        ]

    dtype += [
        ('psfrec_flags', 'i4'),
        ('psfrec_T', 'f4'),
        ('psfrec_g1', 'f4'),
        ('psfrec_g2', 'f4'),
    ]

    # dtype += [
    #     ('psfrec_Tfrac', 'f4'),
    #     ('psfrec_e1diff', 'f4'),
    #     ('psfrec_e2diff', 'f4'),
    # ]

    return _init_struct(dtype=dtype, n=n)


def _init_struct(dtype, n):
    """
    build the array, initializing flags to 1, extra_det to 0 and
    everything else to nan
    """
    import numpy as np

    output = np.zeros(n, dtype=dtype)

    for name in output.dtype.names:
        if 'flags' in name:
            output[name] = NO_ATTEMPT
        elif name in (
            'numiter', 'group_size', 'cell_i', 'cell_j',
        ):
            output[name] = 0
        elif name == 'obj_id':
            output[name] = np.arange(n)
        elif name == 'group_id':
            # ml-branch rows keep group_id == obj_id; the
            # deblender overwrites with the blend-group id
            output[name] = np.arange(n)
        else:
            output[name] = np.nan

    return output


def get_cell_meta(nband, n=1):
    dtype = [
        ('tract', 'i4'),
        ('patch', 'i4'),
        ('cell_i', 'i4'),
        ('cell_j', 'i4'),
        ('good_frac', 'f4', nband),
        ('kept', bool),
    ]
    return np.zeros(n, dtype=dtype)
