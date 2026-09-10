"""
output file naming and writing
"""
import numpy as np
import rustfits

# inspection color image: make-color-image.py rendering at
# reduced resolution
# stretch = COLOR_NSIG * median band sky sigma; low so the
# diagnostics stretch shows down into the noise
COLOR_NSIG = 2.0
COLOR_Q = 7.0
COLOR_RFAC = 1.2    # scale on r (the blue channel)
COLOR_BIN = 4       # block-average factor
# non-footprint area: blend toward a color wash, visible even
# over dark sky and chosen not to match any single band's
# noise color
COLOR_TINT_FRAC = 0.25
COLOR_TINT = (20.0, 105.0, 105.0)     # teal
# COLOR_TINT = (120.0, 20.0, 120.0)   # magenta
# COLOR_TINT = (85.0, 30.0, 150.0)    # violet


def write_color_image(fname, coadds, wcs=None, footprint=None,
                      binfac=COLOR_BIN):
    """
    reduced-resolution Lupton color jpg of the final masked
    images, for inspecting gross problems.  Band mapping
    r/i/z -> blue/green/red with the make-color-image.py
    settings: stretch COLOR_NSIG x the median per-band sky
    sigma, Q = COLOR_Q, r scaled COLOR_RFAC into blue.

    When a footprint is given, the area outside it (dropped
    cells, the border ring, the tract trim) is tinted red

    Parameters
    ----------
    fname: str
        Output jpg path
    coadds: list
        The [r, i, z] coadds (ButlerCoadd or FilePatchCoadd)
    wcs: ButlerWcs or FileWcs, optional
        Needed with footprint, for pixel positions
    footprint: healsparse map, optional
        The keep-footprint; non-footprint area is tinted
    binfac: int, optional
        Block-average reduction factor
    """
    from astropy.visualization import make_lupton_rgb
    from PIL import Image

    # no blanking of flagged pixels: this is a diagnostics
    # image and shows the values as processing would see them
    # (star masking zeroes the saturated cores when it is on)
    imlist = []
    sigs = []
    for coadd in coadds:
        sigs.append(float(np.sqrt(
            np.nanmedian(coadd.variance.array),
        )))
        imlist.append(_block_average(
            coadd.image.array, binfac,
        ))
    # median over bands, not mean: z is much shallower than
    # r/i and a z-dominated mean over-stretches the deep bands
    sigma = float(np.median(sigs))

    rgb = make_lupton_rgb(
        imlist[2],
        imlist[1],
        imlist[0] * COLOR_RFAC,
        minimum=0,
        stretch=COLOR_NSIG * sigma,
        Q=COLOR_Q,
    )
    if footprint is not None:
        fp = _sample_footprint(
            footprint, wcs, coadds[0].bbox,
            imlist[0].shape, binfac,
        )
        f = COLOR_TINT_FRAC
        tinted = (
            (1 - f) * rgb[~fp] + f * np.array(COLOR_TINT)
        )
        rgb[~fp] = np.clip(tinted, 0, 255).astype('u1')
    print('writing:', fname)
    Image.fromarray(rgb[::-1]).save(fname, quality=92)


def _sample_footprint(footprint, wcs, bbox, shape, binfac):
    """footprint values at the centers of the reduced pixels"""
    ny2, nx2 = shape
    yy, xx = np.mgrid[0:ny2, 0:nx2]
    x = bbox.x.start + (xx.ravel() + 0.5) * binfac - 0.5
    y = bbox.y.start + (yy.ravel() + 0.5) * binfac - 0.5
    ra, dec = wcs.pixelToSkyArray(x, y, degrees=True)
    return footprint.get_values_pos(
        ra, dec, lonlat=True,
    ).reshape(shape)


def _block_average(arr, binfac):
    """block-average reduction, ignoring nan"""
    ny, nx = arr.shape
    ny2, nx2 = ny // binfac, nx // binfac
    cut = arr[:ny2 * binfac, :nx2 * binfac]
    blocks = cut.reshape(ny2, binfac, nx2, binfac)
    out = np.nanmean(blocks, axis=(1, 3))
    return np.nan_to_num(out)


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
    run_options=None,
    starsub_tables=None,
):
    """
    Write the catalog, the meta table and the cell meta.

    The meta table (provenance.make_meta) records the run identity
    and switches, the remaining process_cells options passed as
    run_options (repo, collections, patch_dir, gaia_file, gsub,
    apod_stars, cells), the settings of every processing stage,
    the package versions, and the date, host and command line.

    starsub_tables, when given, is a dict of extname -> table
    written after the cell meta: the joint star route's fit
    (lsst_starsub.starsub.make_fit_tables), from which the
    subtracted sky and star images can be rebuilt
    """
    from .provenance import make_meta

    options = dict(
        tract=tract,
        patch=patch,
        seed=seed,
        model=model,
        with_mdet=with_mdet,
        redo_bg=redo_bg,
        starsub=starsub,
        deblend=deblend,
        s2_detect=s2_detect,
    )
    if run_options is not None:
        options.update(run_options)
    meta = make_meta(options)

    print('writing:', fname)
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_table(st, extname='cat', compress=True)
        fits.write_table(meta, extname='meta', compress=True)
        fits.write_table(cell_meta, extname='cell_meta', compress=True)
        if starsub_tables is not None:
            for extname, table in starsub_tables.items():
                fits.write_table(table, extname=extname, compress=True)


def write_patch_files(
    fname, deep_coadd, hdr, psf_stack, cells, ncellx, ncelly,
    cell_size, starmask_plane=None, apod=0.0, star_table=None,
    skyvar=None,
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
    skyvar: array, optional
        The sky-variance map from redo_background, for the
        pixel weights downstream
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
        if skyvar is not None:
            fits.write_image(
                skyvar,
                extname='skyvar',
                compress='gzip_2',
            )
