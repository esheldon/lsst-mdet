"""
quality-assurance diagnostics for the star handling: stacked
residual profiles of the masked-and-subtracted stars
"""
import numpy as np

from .defaults import DM_OUT

# stacked-profile brightness bins and radial bin edges
# (distance beyond each star's own mask radius)
QA_GBINS = [(0.0, 13.0), (13.0, 15.2), (15.2, 19.0)]
QA_EDGES = np.concatenate([
    np.arange(0, 60, 10), np.arange(60, 160, 25),
    np.arange(160, 420, 60),
]).astype(float)


def measure_stacked_star_residuals(coadd, star_table, starmask):
    """
    stacked residual profiles of the census stars on the final
    image: azimuthal medians in annuli of distance beyond each
    star's own mask (other detections and all star attenuation
    zones excluded), in units of the sky sigma, median-stacked
    across the stars of each QA_GBINS brightness bin.  Flat
    and zero means the subtraction left nothing behind

    The profiles are referenced to the blank-pixel median of
    the same image: the background zero is mode-like (sep)
    while these are medians, and the sub-threshold source
    carpet skews the median of blank sky a few 0.001 sigma
    above the mode.  Without the reference every profile
    carries that pedestal at all radii; with it, zero means
    "indistinguishable from blank sky"

    Parameters
    ----------
    coadd: ButlerCoadd or FilePatchCoadd
        The final processed band
    star_table: array
        The gaia census (x, y patch frame, G, on_image)
    starmask: array
        The union attenuation-zone mask (bool, patch frame)

    Returns
    -------
    list of dicts, one per brightness bin, with keys
    gmin, gmax, nstars, rmid, stacked, err
    """
    from .starsub import circle_radius, field_segmentation

    image = coadd.image.array
    var = coadd.variance.array
    mask0 = coadd.mask.array[:, :, 0]
    good = (
        np.isfinite(image) & np.isfinite(var) & (var > 0)
        & ((mask0 & DM_OUT) == 0)
    )
    sig = float(np.sqrt(np.nanmedian(var[good])))
    seg = field_segmentation(image, good, sig)
    okbase = good & (seg == 0) & ~starmask

    # the expected pedestal: the blank-sky median sits above
    # the mode-like background zero (see the docstring)
    ped = float(np.median(image[okbase])) / sig

    edges = QA_EDGES
    rmid = 0.5 * (edges[:-1] + edges[1:])
    ny, nx = image.shape

    results = []
    for gmin, gmax in QA_GBINS:
        sel = (
            (star_table['on_image'] == 1)
            & (star_table['G'] >= gmin)
            & (star_table['G'] < gmax)
        )
        profs = []
        for s in star_table[sel]:
            rad = float(circle_radius(float(s['G'])))
            icx = int(round(s['x']))
            icy = int(round(s['y']))
            m = int(rad + edges[-1] + 20)
            x0, x1 = max(0, icx - m), min(nx, icx + m + 1)
            y0, y1 = max(0, icy - m), min(ny, icy + m + 1)
            gy, gx = np.mgrid[y0:y1, x0:x1]
            rr = np.hypot(gy - s['y'], gx - s['x']) - rad
            ok = okbase[y0:y1, x0:x1]
            prof = np.full(rmid.size, np.nan)
            for i, (lo, hi) in enumerate(
                zip(edges[:-1], edges[1:]),
            ):
                w = (rr >= lo) & (rr < hi) & ok
                if w.sum() > 100:
                    prof[i] = np.median(
                        image[y0:y1, x0:x1][w],
                    ) / sig - ped
            if np.isfinite(prof).any():
                profs.append(prof)

        n = len(profs)
        stacked = np.full(rmid.size, np.nan)
        err = np.full(rmid.size, np.nan)
        if n > 0:
            profs = np.array(profs)
            cnt = np.sum(np.isfinite(profs), axis=0)
            wc = cnt >= 3
            if wc.any():
                stacked[wc] = np.nanmedian(
                    profs[:, wc], axis=0,
                )
                # median error from the scatter
                err[wc] = 1.25 * np.nanstd(
                    profs[:, wc], axis=0,
                ) / np.sqrt(cnt[wc])
        results.append(dict(
            gmin=gmin, gmax=gmax, nstars=n,
            rmid=rmid, stacked=stacked, err=err,
        ))
    return results


def write_star_residual_qa(fname, coadds, star_table, starmask):
    """
    measure the stacked residual profiles for every band and
    write the QA figure

    Parameters
    ----------
    fname: str
        Output png path
    coadds: list
        The final processed bands
    star_table: array
        The gaia census
    starmask: array
        The union attenuation-zone mask
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as mplt

    # always a two-column grid (e.g. griz as 2x2); unused
    # slots are turned off
    nb = len(coadds)
    ncols = 2
    nrows = (nb + ncols - 1) // ncols
    fig, axs = mplt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(5.5 * ncols, 5.0 * nrows),
        sharey=True, squeeze=False,
    )
    for ax in axs.flat[nb:]:
        ax.set_visible(False)
    for iband, coadd in enumerate(coadds):
        results = measure_stacked_star_residuals(
            coadd, star_table, starmask,
        )
        ax = axs.flat[iband]
        for res in results:
            if not np.isfinite(res['stacked']).any():
                if res['nstars'] > 0:
                    print(f'    star residual QA {coadd.band} '
                          f"G {res['gmin']:.0f}-"
                          f"{res['gmax']:.0f}: too few stars "
                          f"to stack ({res['nstars']})")
                continue
            label = (f"G {res['gmin']:.0f}-{res['gmax']:.0f} "
                     f"({res['nstars']})")
            ax.errorbar(
                res['rmid'], res['stacked'], yerr=res['err'],
                marker='o', ms=3, lw=1, capsize=2, label=label,
            )
            stmax = np.nanmax(np.abs(res['stacked']))
            print(f'    star residual QA {coadd.band} '
                  f"G {res['gmin']:.0f}-{res['gmax']:.0f}: "
                  f"max |stack| {stmax:.4f} sigma "
                  f"({res['nstars']} stars)")
        ax.axhline(0, color='k', lw=0.7)
        ax.axhspan(-0.01, 0.01, color='gray', alpha=0.25)
        ax.set_xlabel('r - R(mask) [pix]')
        ax.set_title(f'{coadd.band} band')
        ax.set_ylim(-0.15, 0.15)
        if iband % ncols == 0:
            ax.set_ylabel('stacked median residual [sigma]')
        if iband == 0:
            ax.legend(fontsize=8)
    fig.suptitle(
        'stacked star residuals outside the masks '
        '(referenced to the blank-sky median)'
    )
    fig.tight_layout()
    print('writing:', fname)
    fig.savefig(fname, dpi=110)
    mplt.close(fig)
