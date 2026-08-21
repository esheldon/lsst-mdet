"""
output file naming and writing
"""
import numpy as np
import rustfits
from .defaults import MIN_GOOD_FRAC


def write_output(
    fname,
    st,
    cell_meta,
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
        fits.write_table(cell_meta, extname='cell_meta', compress=True)


def write_patch_files(
    fname, deep_coadd, hdr, psf_stack, cells, ncellx, ncelly,
    cell_size, starmask_plane=None, apod=0.0, star_table=None,
    gsub=None, minrad=None,
):
    """
    write the getimages patch file: image/var/mask/noise/mfrac
    planes, the per-cell psf cube and cell table, and the
    optional star-subtraction products

    Parameters
    ----------
    fname: str
        Output file path
    deep_coadd: deep_coadd
        The (possibly star-subtracted, background-redone)
        coadd whose planes are written
    hdr: dict
        Header for the image extension (wcs etc.)
    psf_stack: array
        (ncell, ny, nx) psf stamps, row k matching row k of
        the cells table
    cells: array with fields
        cellx, celly (patch-frame pixel coordinates), ok
    ncellx, ncelly: int
        Cell counts of the grid
    cell_size: (int, int)
        (csx, csy) cell size in pixels
    starmask_plane: array, optional
        u1 plane: 0 = clear, 1 = taper zone (attenuated), 2 =
        star mask (zeroed when apod > 0)
    apod: float, optional
        The taper width recorded in the starmask header
    star_table: array with fields, optional
        The gaia census with fitted amplitudes
    gsub, minrad: optional
        Census depth and circle floor, recorded in the
        gaia_stars header
    """
    import rustfits

    csx, csy = cell_size
    print('writing:', fname)
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_image(
            deep_coadd.image.array,
            extname='image',
            compress='gzip_2',
            header=hdr,
        )
        fits.write_image(
            deep_coadd.variance.array,
            extname='var',
            compress='gzip_2',
        )
        fits.write_image(
            deep_coadd.mask.array,
            extname='mask',
            compress=rustfits.Gzip2(tile_shape=[1, 3300, 1]),
        )
        fits.write_image(
            deep_coadd.noise_realizations[0].array,
            extname='noise',
            compress='gzip_2',
        )
        # masked-fraction plane, needed for butler-free
        # processing (process_cells --patch-dir)
        fits.write_image(
            deep_coadd.mask_fractions['rejected'].array,
            extname='mfrac',
            compress='gzip_2',
        )
        # (ncell, ny, nx) psf stamps, row k matching row k of
        # the psf_cells table; cell centers are patch-frame
        # pixel coordinates
        fits.write_image(
            psf_stack,
            extname='psfs',
            compress='gzip_2',
        )
        fits.write_table(
            cells,
            extname='psf_cells',
            header={
                'CELLSX': csx, 'CELLSY': csy,
                'NCELLX': int(ncellx),
                'NCELLY': int(ncelly),
            },
        )
        if starmask_plane is not None:
            fits.write_image(
                starmask_plane,
                extname='starmask',
                compress='gzip_2',
                header={'APOD': apod},
            )
        if star_table is not None:
            fits.write_table(
                star_table,
                extname='gaia_stars',
                header={'GSUB': gsub, 'MINRAD': minrad},
            )
