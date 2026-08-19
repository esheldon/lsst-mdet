"""
output file naming and writing
"""
import numpy as np
import os
import rustfits
from .defaults import MIN_GOOD_FRAC


def get_dir(tract):
    return f'{tract:05d}'


def get_fname(tract, patch, with_mdet):

    fl = [f'cat-{tract:05d}-{patch:02d}']
    if with_mdet:
        fl += ['metacal']

    dir = get_dir(tract)
    bname = '-'.join(fl) + '.fits'
    return os.path.join(dir, bname)


def write_output(
    fname,
    st,
    cell_info,
    tract,
    patch,
    model,
    seed,
    with_mdet,
    redo_bg,
    starsub,
    deblend,
    s2_detect,
):
    meta = np.zeros(1, dtype=[
        ('tract', 'i4'),
        ('patch', 'i4'),
        ('seed', 'i8'),
        ('model', 'U5'),
        ('with_mdet', bool),
        ('redo_bg', bool),
        ('starsub', bool),
        ('deblend', bool),
        ('s2_detect', bool),
        ('min_good_frac', 'f4'),
    ])
    meta['tract'] = tract
    meta['patch'] = patch
    meta['seed'] = seed
    meta['with_mdet'] = with_mdet
    meta['model'] = model
    meta['redo_bg'] = redo_bg
    meta['starsub'] = starsub
    meta['deblend'] = deblend
    meta['s2_detect'] = s2_detect
    meta['min_good_frac'] = MIN_GOOD_FRAC

    print('writing:', fname)
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_table(st, extname='cat', compress=True)
        fits.write_table(meta, extname='meta', compress=True)
        fits.write_table(cell_info, extname='cell_info', compress=True)
