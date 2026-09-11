"""
the healsparse keep-footprint for a patch: the union of the
processed cell polygons, with the star masks (out to the
taper) cleared and the whole thing trimmed to the tract inner
boundary

and the union of the per-patch footprints for a whole run,
streamed to disk one coverage pixel at a time (see cat_footprints)
"""
import glob
import os

import numpy as np

NSIDE_COVERAGE = 32
NSIDE = 131072

# files per task when scanning the coverage of the per-patch maps
SCAN_CHUNK = 200

# set bits per byte value, for counting footprint pixels in the
# bit-packed blocks
_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype='i8')


def make_empty_footprint():
    """
    Make an empty bool footprint which we will fill with
    our geometry
    """
    import healsparse
    return healsparse.HealSparseMap.make_empty(
        nside_coverage=NSIDE_COVERAGE,
        nside_sparse=NSIDE,
        dtype=bool,
        bit_packed=True,
    )


def mask_stars_in_footprint(footprint, wcs, bbox, star_table, apod):
    """
    Mask the stars in the footprint: clear the pixels of each
    census star's circle (the mask radius law plus the taper).

    OR-ing a value=False geometry cannot clear pixels, so the
    circle pixels are set False directly

    Parameters
    ----------
    footprint: healsparse map
        The footprint in which stars will be masked
    wcs: ButlerWcs or FileWcs
        Used for the pixel scale
    bbox: bbox
        The patch bounding box, tract frame
    star_table: array
        The gaia census with ra, dec, G
    apod: float
        The star taper width in pixels, added to the circle
        radii
    """
    import healsparse
    from .starsub import circle_radius

    scale = get_pixel_scale(
        wcs,
        0.5 * (bbox.x.start + bbox.x.stop),
        0.5 * (bbox.y.start + bbox.y.stop),
    )

    rad_deg = (circle_radius(star_table['G']) + apod) * scale / 3600

    for cra, cdec, crad in zip(
        star_table['ra'], star_table['dec'], rad_deg,
    ):
        circle = healsparse.Circle(
            ra=cra,
            dec=cdec,
            radius=crad,
            value=True,
        )
        footprint[circle.get_pixels(nside=NSIDE)] = False


def mask_pixels_in_footprint(footprint, wcs, bbox, pixmask):
    """
    Clear the footprint pixels holding any True pixel of an
    image-plane mask.  Any masked image pixel clears the footprint
    pixel containing it; set False directly as in
    mask_stars_in_footprint

    Parameters
    ----------
    footprint: healsparse map
        The footprint to clear
    wcs: ButlerWcs or FileWcs
        For the sky positions of the image pixels
    bbox: bbox
        The patch bounding box, tract frame
    pixmask: bool array
        The image-plane mask, patch frame
    """
    import hpgeom

    iy, ix = np.nonzero(pixmask)
    if iy.size == 0:
        return
    ra, dec = wcs.pixelToSkyArray(
        (ix + bbox.x.start).astype('f8'),
        (iy + bbox.y.start).astype('f8'),
        degrees=True,
    )
    footprint[np.unique(hpgeom.angle_to_pixel(NSIDE, ra, dec))] = False


def trim_footprint_to_tract_bounds(footprint, tract_bounds):
    """
    Clear footprint pixels whose centers fall outside the
    tract inner boundary, using the same test as the catalog
    is_primary cut.

    AND-ing a value=True geometry is a no-op (it never clears
    pixels outside the geometry), so the outside pixels are
    set False directly

    Parameters
    ----------
    footprint: healsparse map
        The footprint to trim
    tract_bounds: (ra_min, ra_max, dec_min, dec_max)
        The tract inner sky bounds in degrees
    """
    import hpgeom
    from .cells import get_tract_primary

    vp = footprint.valid_pixels
    if vp.size == 0:
        return
    ra, dec = hpgeom.pixel_to_angle(
        NSIDE, vp, nest=True, lonlat=True, degrees=True,
    )
    keep = get_tract_primary(tract_bounds, ra, dec)
    footprint[vp[~keep]] = False


def get_pixel_scale(wcs, x, y):
    """arcsec per pixel from the local linearization"""
    m = wcs.linearize_matrix(x, y)
    return float(np.sqrt(np.abs(np.linalg.det(m))))


# the star-exclusion boundary beyond the attenuation zone (mask
# circle + taper), DES style: objects closer than this have their
# moment aperture overlapping the attenuated region, giving a
# tangential shear ring of +3 to +13 x10^-3 confined to ~4 arcsec
# beyond the mask radius law (2026-09 diagnostics, run-dp2-v00
# notes).  The boundary is referenced to the taper edge with
# margin for larger-than-average apertures
EXCLUSION_BOUNDARY = 4.0   # arcsec

# stars fainter than the subtraction/masking limit have no mask
# or taper; their contamination is blending with the unsubtracted
# star light, measured confined to r < 4 arcsec (gamma_t spike
# +5 x10^-3), excluded with a fixed circle with margin
EXCLUSION_FAINT_RADIUS = 6.0   # arcsec
EXCLUSION_FAINT_GMAX = 20.0

EXCLUSION_CHUNK = 20000    # stars per task


def star_exclusion_radii(gmag, boundary=EXCLUSION_BOUNDARY,
                         faint_radius=EXCLUSION_FAINT_RADIUS):
    """
    the star exclusion radius in degrees: for masked stars
    (G < starsub GSUB) the mask radius law plus the taper width,
    at the nominal 0.2 arcsec pixel scale, plus the boundary;
    for fainter (unmasked) stars the fixed faint radius

    Parameters
    ----------
    gmag: array
        Gaia G magnitudes
    boundary: float, optional
        The extra margin beyond the attenuation zone, arcsec
    faint_radius: float, optional
        The circle for unmasked stars, arcsec

    Returns
    -------
    radius in degrees, same shape as gmag
    """
    from .starsub import circle_radius, APOD_STARS, GSUB
    rad_arcsec = np.where(
        np.asarray(gmag) < GSUB,
        (circle_radius(gmag) + APOD_STARS) * 0.2 + boundary,
        faint_radius,
    )
    return rad_arcsec / 3600.0


def _exclusion_pixels(task):
    import hpgeom
    ra, dec, rad_deg = task
    pix = [
        hpgeom.query_circle(
            NSIDE, r, d, rr, nest=True, inclusive=False,
        )
        for r, d, rr in zip(ra, dec, rad_deg)
    ]
    return np.unique(np.concatenate(pix))


def make_star_exclusion_map(ra, dec, gmag,
                            boundary=EXCLUSION_BOUNDARY,
                            faint_radius=EXCLUSION_FAINT_RADIUS,
                            nproc=1):
    """
    build the star exclusion map: True inside a circle of
    star_exclusion_radii around each star.  AND NOT this map
    with a footprint (or look up object positions in it) to
    apply the exclusion

    Parameters
    ----------
    ra, dec: arrays
        The star positions, degrees
    gmag: array
        Gaia G magnitudes
    boundary: float, optional
        The extra margin beyond the attenuation zone, arcsec
    faint_radius: float, optional
        The circle for unmasked stars, arcsec
    nproc: int, optional
        Processes for the circle queries

    Returns
    -------
    healsparse bit-packed bool map at the footprint NSIDE
    """
    rad_deg = star_exclusion_radii(
        gmag, boundary=boundary, faint_radius=faint_radius,
    )
    return make_circle_exclusion_map(ra, dec, rad_deg, nproc=nproc)


def make_circle_exclusion_map(ra, dec, rad_deg, nproc=1):
    """
    a bit-packed bool map at the footprint NSIDE, True inside the
    given circles

    Parameters
    ----------
    ra, dec: arrays
        The circle centers, degrees
    rad_deg: array
        The circle radii, degrees
    nproc: int, optional
        Processes for the circle queries
    """
    chunks = [
        (ra[i:i + EXCLUSION_CHUNK],
         dec[i:i + EXCLUSION_CHUNK],
         rad_deg[i:i + EXCLUSION_CHUNK])
        for i in range(0, ra.size, EXCLUSION_CHUNK)
    ]
    with _get_pool(nproc) as pool:
        results = _pool_map(pool, _exclusion_pixels, chunks)

    exmap = make_empty_footprint()
    for pix in results:
        exmap[pix] = True
    return exmap


def make_patch_polygon(ra, dec):
    """
    Make a polygon for the patch region

    Parameters
    ----------
    ra: array
        Array of ra for the corners
    dec: array
        Array of dec for the corners

    Returns
    -------
    healsparse.Polygon
    """
    import healsparse
    return healsparse.Polygon(ra=ra, dec=dec, value=True)


def get_footprint_flist(run_dir):
    """
    every per-patch footprint map of a run, sorted: the
    <tract>/<tract>-<patch>-mdet-footprint.hsp files that
    lsst-mdet-process-cells writes beside the catalogs
    """
    pattern = os.path.join(run_dir, '[0-9]*', '*-mdet-footprint.hsp')
    return sorted(glob.glob(pattern))


def cat_footprints(flist, outfile, nproc=1, clobber=False):
    """
    The union of per-patch footprint maps, streamed to outfile one
    coverage pixel at a time so the full map is never in memory.

    healsparse.cat_healsparse_files does this for ordinary maps but
    not for bit-packed ones (it takes one row per fine pixel where
    the packed map has one byte per eight) and it writes the output
    uncompressed.  This follows its plan with the packed row unit,
    through healsparse's fits shim: scan the coverage of every
    file, write a stub with the final coverage map, then for each
    output coverage pixel OR the input blocks and append the result
    to the SPARSE image, RICE compressed by block.  Overlapping
    pixels are OR-ed, so the result is the union.

    Parameters
    ----------
    flist: list of str
        The per-patch footprint files, all with the same
        nside_coverage and nside_sparse
    outfile: str
        The output map; written as outfile.incomplete and renamed
    nproc: int
        Processes for the coverage scan and the block reads
    clobber: bool
        Overwrite an existing outfile

    Returns
    -------
    dict with nfile, ncov (coverage pixels), n_valid (footprint
    pixels), n_valid_inputs (summed over the inputs; the excess over
    n_valid is overlap) and area_deg2
    """
    import hpgeom
    from healsparse.fits_shim import HealSparseFits
    from healsparse.healSparseCoverage import HealSparseCoverage

    if len(flist) == 0:
        raise ValueError('no footprint files')
    if os.path.exists(outfile) and not clobber:
        raise RuntimeError(f'{outfile} exists and clobber is False')

    with _get_pool(nproc) as pool:
        print(f'scanning the coverage of {len(flist)} files')
        nside_coverage, nside_sparse, table = scan_footprints(flist, pool)

        # the blocks of each output coverage pixel are contiguous in
        # the pixel-sorted table
        cov_pix, first = np.unique(table['pixel'], return_index=True)
        bounds = np.append(first, table.size)
        cov_map = HealSparseCoverage.make_from_pixels(
            nside_coverage, nside_sparse, cov_pix,
        )
        nbytes = cov_map.nfine_per_cov // 8
        print(f'{cov_pix.size} coverage pixels, {nbytes} bytes per block')

        tmpfile = outfile + '.incomplete'
        _write_footprint_stub(
            tmpfile, cov_map[:], nside_coverage, nside_sparse, nbytes,
        )

        tasks = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            sources = [
                (flist[f], int(s))
                for f, s in zip(table['file'][lo:hi], table['start'][lo:hi])
            ]
            tasks.append((nbytes, sources))

        # batches bound the blocks held in memory
        batch = max(1, 2 * nproc)
        n_valid = 0
        n_valid_inputs = 0
        with HealSparseFits(tmpfile, mode='rw') as fits:
            for i in range(0, len(tasks), batch):
                results = _pool_map(pool, _or_blocks, tasks[i:i + batch])
                for block, nset_inputs in results:
                    fits.append_extension('SPARSE', block.reshape(1, nbytes))
                    n_valid += _count_set_bits(block)
                    n_valid_inputs += nset_inputs
                done = min(i + batch, len(tasks))
                print(f'  coverage pixel {done}/{len(tasks)}', flush=True)

    os.replace(tmpfile, outfile)

    area = hpgeom.nside_to_pixel_area(nside_sparse, degrees=True) * n_valid
    return {
        'nfile': len(flist),
        'ncov': int(cov_pix.size),
        'n_valid': n_valid,
        'n_valid_inputs': n_valid_inputs,
        'area_deg2': area,
    }


def scan_footprints(flist, pool=None):
    """
    Read the coverage of every footprint file

    Parameters
    ----------
    flist: list of str
        The footprint files
    pool: multiprocessing pool, optional
        Used when not None

    Returns
    -------
    nside_coverage, nside_sparse: int
    table: structured array
        One row per (file, coverage pixel) with file (index into
        flist), pixel and start (byte offset of the pixel's block in
        the file's SPARSE image), sorted by pixel then file
    """
    chunks = [
        flist[i:i + SCAN_CHUNK] for i in range(0, len(flist), SCAN_CHUNK)
    ]
    results = _pool_map(pool, _scan_chunk, chunks)

    nside_coverage = None
    nside_sparse = None
    files = []
    pixels = []
    starts = []
    ifile = 0
    for chunk in results:
        for nsc, nss, pix, start in chunk:
            if nside_coverage is None:
                nside_coverage, nside_sparse = nsc, nss
            elif (nsc, nss) != (nside_coverage, nside_sparse):
                raise ValueError(
                    f'{flist[ifile]}: nside_coverage/nside_sparse '
                    f'({nsc}, {nss}) differ from the first file '
                    f'({nside_coverage}, {nside_sparse})'
                )
            files.append(np.full(pix.size, ifile, dtype='i8'))
            pixels.append(pix)
            starts.append(start)
            ifile += 1

    table = np.zeros(
        sum(p.size for p in pixels),
        dtype=[('file', 'i8'), ('pixel', 'i8'), ('start', 'i8')],
    )
    table['file'] = np.concatenate(files)
    table['pixel'] = np.concatenate(pixels)
    table['start'] = np.concatenate(starts)
    table.sort(order=['pixel', 'file'])
    return nside_coverage, nside_sparse, table


def _scan_chunk(fnames):
    return [_scan_footprint(fname) for fname in fnames]


def _scan_footprint(fname):
    """
    the geometry and block table of one bit-packed footprint file:
    nside_coverage, nside_sparse, the coverage pixels and the byte
    offset of each one's block in the SPARSE image
    """
    from healsparse.fits_shim import HealSparseFits
    from healsparse.healSparseCoverage import HealSparseCoverage

    with HealSparseFits(fname) as fits:
        cov = HealSparseCoverage.read(fits)
        hdr = fits.read_ext_header('SPARSE')

    if not hdr.get('BITPACK', False):
        raise ValueError(f'{fname}: not a bit-packed map')
    if hdr.get('RESHAPED', False):
        raise NotImplementedError(
            f'{fname}: reshaped (2-d) input maps are not supported'
        )

    pixels, = np.where(cov.coverage_mask)
    # the coverage index map holds each block's start in the sparse
    # map minus pixel * nfine_per_cov; eight fine pixels per byte
    starts = (cov[:][pixels] + pixels * cov.nfine_per_cov) // 8
    return cov.nside_coverage, cov.nside_sparse, pixels, starts


def _or_blocks(task):
    """
    the OR of the input blocks of one output coverage pixel

    task: (nbytes, [(fname, start), ...])

    Returns the packed block and the set-bit count summed over the
    inputs
    """
    from healsparse.fits_shim import HealSparseFits

    nbytes, sources = task
    block = np.zeros(nbytes, dtype=np.uint8)
    nset_inputs = 0
    for fname, start in sources:
        with HealSparseFits(fname) as fits:
            inblock = fits.read_ext_data(
                'SPARSE', row_range=[start, start + nbytes],
            )
        if inblock.shape != (nbytes,):
            raise ValueError(
                f'{fname}: read block of shape {inblock.shape}, '
                f'expected ({nbytes},)'
            )
        nset_inputs += _count_set_bits(inblock)
        np.bitwise_or(block, inblock, out=block)

    return block, nset_inputs


def _count_set_bits(packed):
    return int(_POPCOUNT[packed].sum())


def _write_footprint_stub(fname, cov_index_map, nside_coverage,
                          nside_sparse, nbytes):
    """
    the output file with the final coverage map and a sparse map
    holding only the overflow block, laid out the way healsparse
    writes a compressed bit-packed map with more than 2**31
    elements: a 2-d image with one coverage block per row
    (RESHAPED), RICE compressed by row, ready for blocks to be
    appended.

    HealSparseMap.write would make the one-block stub 1-d (it only
    reshapes above 2**31 elements) and a 1-d compressed image
    cannot grow past that, so the layout is written directly with
    the headers healsparse.io_map_fits uses
    """
    import rustfits

    c_hdr = {'PIXTYPE': 'HEALSPARSE', 'NSIDE': nside_coverage}
    s_hdr = {
        'PIXTYPE': 'HEALSPARSE',
        'NSIDE': nside_sparse,
        'SENTINEL': False,
        'BITPACK': True,
        'RESHAPED': True,
    }
    with rustfits.FITS(fname, mode='w+') as fits:
        fits.write_image(cov_index_map, extname='COV', header=c_hdr)
        fits.write_image(
            np.zeros((1, nbytes), dtype=np.uint8),
            extname='SPARSE',
            header=s_hdr,
            compress=rustfits.Rice1(tile_shape=(1, nbytes)),
        )


def _get_pool(nproc):
    """
    a fork pool context, or a null context yielding None for one
    process
    """
    import contextlib
    import multiprocessing as mp

    if nproc <= 1:
        return contextlib.nullcontext(None)
    return mp.get_context('fork').Pool(nproc)


def _pool_map(pool, func, tasks):
    if pool is None:
        return [func(task) for task in tasks]
    return pool.map(func, tasks)
