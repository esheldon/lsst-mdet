"""
deblending measurement
"""
from .colors_and_fluxes import set_fluxes, set_colors
from .defaults import (
    FLAG_DUPLICATE_EXTRA,
    FLAG_EXTRA_DET_OFF_SEG,
    FLAG_NOT_CONVERGED,
    NO_ATTEMPT,
    ZERO_WEIGHTS,
)

R_DUP_FIT = 1.5
GROUP_BOX_PAD = 10

# with maxiter_type 'scaled', the sweep cap applies as configured
# up to this group size and is scaled up linearly with the member
# count beyond it: sweeps to converge grow with group size
# (Gauss-Seidel information propagates about one object per
# sweep; measured on 2000 wldb fields the converged-group median
# numiter goes 11 -> 133 and the p99 102 -> 591 from single
# objects to 17-64 members, and an extras-enabled 31-member
# cluster core needs 1034).  'fixed' is right for wide fields,
# where the long-running groups are almost all non-converging
# limit cycles and the scaled caps only multiply their cost
MAXITER_SIZE_REF = 8.0

# the deblend settings, recorded in the output meta table (see
# io.write_output).  tguess_min/max bound the size guess from the
# sep moments; the unbounded-above sep size is also passed to
# kdeblend as Tdet, the object's footprint size for its weight
# bound
DEBLEND_SETTINGS = dict(
    tol=1.0e-5,
    maxiter=500,
    maxiter_type='fixed',
    recenter=True,
    full_errors=True,
    ap_rad=1.5,  # pixels
    cen_sigma0=0.1,
    e_sigma0=0.0,
    tguess_min=0.05,
    tguess_max=5.0,
    # extra detections are injected with their centers pinned:
    # with every center free the crowded groups fail to converge
    # (simcoadd-mdet docs/detection-adaptive-null, 376 flagged
    # rows in 16 scenes); pinning the extras only, with the sep
    # rows recentered, is the validated stable setting
    extra_fixcen=True,
)


def fit_deblend(
    mbobs,
    sxcat,
    cat,
    seg,
    model,
    rng,
    extra_detections=None,
    extra_fixcen=None,
    show=False,
):
    """
    Deblend and measure all detected objects

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The per-band observations to fit, each holding image,
        weight, psf and the noise field attached by do_metacal,
        from which the per-mode noise power for the flux errors is
        measured.  A single Observation is also accepted
    sxcat: array with fields
        The sep catalog from detect.run_sep on detobs
    seg: array
        The sep segmentation map, used for the fofx grouping
    rng: np.random.RandomState
        The random number generator
    show: bool, optional
        If set to True, show a kdeblend view_blend figure for each
        blend group after its fit: the region of the field bounding
        the group's stamps, with the fitted models, the seg map and
        the stamp boxes, titled with the blend group id
    extra_detections: array, optional
        (N, 2) array of (x, y) 0-offset pixel positions of extra
        objects to inject into the deblend, e.g. peaks found on
        the adaptive-null detection images.  Each position joins the
        blend group of the seg island it lands on and is fit
        jointly with that group, with the configured model (never
        dev-classified), a size guess from the smoothing scale,
        and full kdeblend measurements; the catalog gains one row
        per position, marked with extra_det.  A position landing
        on seg background is not fit; its row gets
        flags=FLAG_EXTRA_DET_OFF_SEG.  The caller chooses which
        positions to inject (e.g. an exclusion radius against the
        sep detections).  Group modes only: extra detections have
        no sep bbox for the stamp cutting, so deblend_mode
        'stamps' raises an error
    extra_fixcen: bool array, optional
        Per extra detection, keep its center fixed at the
        injected position even when recenter is on (kdeblend
        object fixcen).  Frees injected positions whose adaptive
        centers would couple degenerately to nearby members;
        note a fixed-center extra is never duplicate-flagged
        (the duplicate test is defined by fitted centers
        converging together)

    Returns
    -------
    cat, keep

    cat: array with fields
        One row per sxcat detection, followed by one row per
        extra detection (in extra_detections order); see
        structs.get_struct
    keep: bool array
        Which rows of sxcat were kept (all of them; the stamp
        cutting clips at edges rather than failing).  Length
        sxcat.size: the extra-detection rows are not covered
    """
    import numpy as np
    from ngmix.moments import fwhm_to_T
    from ngmix.prepsfadmom.prep import choose_fwhm_smooth
    from ngmix import GMixFatalError

    s = DEBLEND_SETTINGS
    tol = s['tol']
    maxiter = s['maxiter']
    maxiter_type = s['maxiter_type']
    recenter = s['recenter']
    full_errors = s['full_errors']

    ap_rad = s['ap_rad']
    cen_sigma0 = s['cen_sigma0']
    e_sigma0 = s['e_sigma0']
    tguess_range = (s['tguess_min'], s['tguess_max'])

    bands = [obslist[0].meta['band'] for obslist in mbobs]

    for obslist in mbobs:
        if not obslist[0].has_noise():
            raise ValueError(
                'each band observation must have a noise field for '
                'the per-mode noise power'
            )

    jacobian = mbobs[0][0].jacobian
    scale = jacobian.scale
    v, u = jacobian.get_vu(row=sxcat['y'], col=sxcat['x'])

    # the sep isophotal size: clipped both ways as the size guess,
    # floored only as Tdet, the footprint size kdeblend bounds the
    # adaptive weight with (an object's weight may not grow beyond
    # WEIGHT_TMAX_FAC times its footprint plus the smoothing)
    Tiso = (sxcat['x2'] + sxcat['y2']) * scale ** 2
    Tguess = np.clip(Tiso, tguess_range[0], tguess_range[1])
    Tdet = np.maximum(Tiso, tguess_range[0])

    objects = [
        {
            'v': v[i],
            'u': u[i],
            'type': model,
            'Tguess': Tguess[i],
            'Tdet': float(Tdet[i]),
        }
        for i in range(sxcat.size)
    ]

    fwhm_smooth = choose_fwhm_smooth(mbobs, rng=rng)
    Tsmooth = fwhm_to_T(fwhm_smooth)
    cat['fwhm_smooth'] = fwhm_smooth

    groups = get_groups(sxcat=sxcat, seg=seg)

    # add the extra detections: each joins the group of the seg
    # island it lands on, with the configured model and a size
    # guess from the smoothing scale (the compact-start direction
    # is the safe one, see the Tguess comment above); positions on
    # seg background are not fit and their rows flagged
    nsx = sxcat.size
    n_extra = 0
    off_seg = []
    if extra_detections is not None:
        extra_detections = np.atleast_2d(extra_detections)
        n_extra = len(extra_detections)
        groups = [list(g) for g in groups]
        num_to_group = {}
        for gid, group in enumerate(groups):
            for i in group:
                num_to_group[int(sxcat['number'][i])] = gid
        Tguess_inj = float(
            np.clip(Tsmooth, tguess_range[0], tguess_range[1])
        )
        dim_r, dim_c = seg.shape
        for k, (x, y) in enumerate(extra_detections):
            v, u = jacobian.get_vu(row=y, col=x)
            objects.append(dict(
                v=v,
                u=u,
                type=model,
                Tguess=Tguess_inj,
                Tdet=Tguess_inj,
                fixcen=bool(
                    extra_fixcen is not None and extra_fixcen[k]
                ),
            ))
            ir = int(round(y))
            ic = int(round(x))
            label = (
                int(seg[ir, ic])
                if 0 <= ir < dim_r and 0 <= ic < dim_c else 0
            )
            if label in num_to_group:
                groups[num_to_group[label]].append(nsx + k)
            else:
                off_seg.append(nsx + k)

    cat['extra_det'][nsx:] = True

    # extras have no sep flux_auto and keep the nan init
    for idx in off_seg:
        cat['flags'][idx] = FLAG_EXTRA_DET_OFF_SEG

    for gid, group in enumerate(groups):

        cat['group_size'][group] = len(group)
        cat['group_id'][group] = gid
        try:
            res, gextra = fit_one_group(
                group=group,
                mbobs=mbobs,
                sxcat=sxcat,
                nsx=nsx,
                seg=seg,
                objects=objects,
                maxiter=maxiter,
                maxiter_type=maxiter_type,
                fwhm_smooth=fwhm_smooth,
                ap_rad=ap_rad,
                tol=tol,
                rng=rng,
                recenter=recenter,
                cen_sigma0=cen_sigma0,
                e_sigma0=e_sigma0,
                full_errors=full_errors,
            )

            for i, obj_res in zip(group, res['objects']):
                pack_deblend_object(
                    st=cat[i],
                    obj_res=obj_res,
                    bands=bands,
                    jacobian=jacobian,
                )

            cat['numiter'][group] = res['numiter']

            if not res['converged']:
                cat['flags'][group] = FLAG_NOT_CONVERGED
            else:
                cat['flags'][group] = 0

        except GMixFatalError as err:  # noqa
            cat['flags'][group] = ZERO_WEIGHTS
            continue

        if show:
            show_group(
                mbobs=gextra['gmbobs'],
                seg=gextra['gseg'],
                objects=res['objects'],
                group=groups[gid],
                title=f'blend group {gid}',
            )

    # flag extra rows whose fitted centers converged onto a sep
    # row's fitted center: nuisance components of the same object.
    # They stay in the fit -- they soak crowd light and profile
    # mismatch, improving the photometry of the real rows -- but
    # must not enter downstream selections (a fraction would pass
    # the standard cuts).  Convergence is only meaningful when the
    # centers are refit, so this requires recenter
    if n_extra and recenter:
        xf, yf = cat['x_fit'], cat['y_fit']
        for k in range(n_extra):
            i = nsx + k
            if cat['flags'][i] != 0:
                continue
            d2 = np.nanmin(
                (xf[:nsx] - xf[i]) ** 2 + (yf[:nsx] - yf[i]) ** 2
            )
            if d2 < R_DUP_FIT ** 2:
                cat['flags'][i] |= FLAG_DUPLICATE_EXTRA


def fit_one_group(
    group,
    mbobs,
    sxcat,
    nsx,
    seg,
    objects,
    maxiter,
    maxiter_type,
    fwhm_smooth,
    ap_rad,
    tol,
    rng,
    recenter,
    cen_sigma0,
    e_sigma0,
    full_errors,
):
    """
    cut and fit one deblend group with the configured parameters.
    Extracted from the former fit_deblend closure so external
    harnesses (e.g. the port differential rig) can drive the exact
    production per-group path.

    Returns
    -------
    res, boxes, gcut
        the kdeblend result, the per-member boxes, and the group
        cut (gmbobs, box) for group modes (None for stamps mode)
    """
    from kdeblend import deblend

    if maxiter_type == 'scaled':
        # large groups converge in proportionally more
        # sweeps; see MAXITER_SIZE_REF
        group_maxiter = int(round(
            maxiter * max(1.0, len(group) / MAXITER_SIZE_REF)
        ))
    else:
        group_maxiter = maxiter

    scale = mbobs[0][0].jacobian.scale

    # the cutout box and the member seg labels come from
    # the sep members only; extra-detection members carry
    # no sep bbox but land inside the island by
    # construction
    gmbobs, gseg, box = cut_group_mbobs(
        mbobs=mbobs,
        sxcat=sxcat,
        group=[i for i in group if i < nsx],
        seg=seg,
    )

    gobjects = [objects[i] for i in group]
    res = deblend(
        gmbobs,
        gobjects,
        fwhm_smooth=fwhm_smooth,
        ap_rad=ap_rad,
        use_noise_image=True,
        tol=tol,
        maxiter=group_maxiter,
        rng=rng,
        recenter=recenter,
        cen_sigma0=cen_sigma0,
        e_sigma0=e_sigma0,
        full_errors=full_errors,
        anchor_sigma=(
            anchor_covs(sxcat, group, nsx, scale)
            if full_errors and recenter else 0.0
        ),
    )

    gextra = {
        'gmbobs': gmbobs,
        'gseg': gseg,
        'gobjects': gobjects,
        'gbox': box,
    }
    return res, gextra


def pack_deblend_object(st, obj_res, bands, jacobian):
    """
    Pack one kdeblend per-object result into a catalog row.

    flags and numiter are set outside.  deblend_flags is the
    kdeblend flag word as is (see kdeblend.flags); it shares the
    NO_ATTEMPT bit convention with the other flags columns.
    g_flags is the kdeblend e_flags word (ngmix bits, never
    NO_ATTEMPT): zero iff the shape and its errors are usable.  A
    star has no shape by construction, so its g columns are never
    attempted; a demotion to star is the DEBLENDED_AS_PSF bit of
    deblend_flags.

    The flux columns hold the model's total: the family flux for
    exp and bdf, the psf flux for a star, and for a ladder the
    tau-completed total_flux.  A ladder's colors come from its
    adaptive-aperture (gauss) fluxes with their covariance, the
    lower-noise and less contaminated estimator; those fluxes and
    the fixed-minus-adaptive color gradient fill the ladder-only
    columns of the struct.  s2n is the covariance-aware flux s/n of
    the family flux, which for a ladder is the gauss flux
    """
    st['deblend_flags'] = obj_res['deblend_flags']

    if obj_res['type'] == 'star':
        st['g_flags'] = NO_ATTEMPT
    else:
        st['g_flags'] = obj_res['e_flags']

    g1, g2, g1_err, g2_err, g1g2_cov = _e2g(
        e1=obj_res['e1'],
        e2=obj_res['e2'],
        e1_err=obj_res['e1_err'],
        e2_err=obj_res['e2_err'],
        e1e2_cov=obj_res.get('e1e2_cov', float('nan')),
    )

    st['g1'] = g1
    st['g1_err'] = g1_err
    st['g2'] = g2
    st['g2_err'] = g2_err
    st['g1g2_cov'] = g1g2_cov
    st['T'] = obj_res['T']
    st['T_err'] = obj_res['T_err']

    if obj_res['type'] == 'ladder':
        set_fluxes(
            st=st,
            bands=bands,
            flux=obj_res['total_flux'],
            flux_err=obj_res['total_flux_err'],
        )
        gflux = obj_res['gauss_flux']
        gflux_err = obj_res['gauss_flux_err']
        set_colors(
            st=st,
            bands=bands,
            flux=gflux,
            flux_err=gflux_err,
            flux_cov=obj_res.get('gauss_flux_cov'),
        )
        for iband, band in enumerate(bands):
            st[f'gauss_flux_{band}'] = gflux[iband]
            st[f'gauss_flux_err_{band}'] = gflux_err[iband]
        for i in range(len(bands) - 1):
            name = f'gradient_{bands[i]}m{bands[i + 1]}'
            st[name] = obj_res['gradient'][i]
            st[f'{name}_err'] = obj_res['gradient_err'][i]
    else:
        set_fluxes(
            st=st,
            bands=bands,
            flux=obj_res['flux'],
            flux_err=obj_res['flux_err'],
        )
        set_colors(
            st=st,
            bands=bands,
            flux=obj_res['flux'],
            flux_err=obj_res['flux_err'],
            flux_cov=obj_res['flux_cov'],
        )

    st['s2n'] = obj_res['s2n']
    row, col = jacobian.get_rowcol(obj_res['cen'][0], obj_res['cen'][1])
    st['x_fit'] = col
    st['y_fit'] = row


def _e2g(e1, e2, e1_err, e2_err, e1e2_cov=float('nan')):
    """
    convert distortion-convention shapes to reduced shear,
    g = e / (1 + sqrt(1 - e^2)), propagating the shape covariance
    through the jacobian.  The deblender guarantees e^2 < 1 for
    usable shapes (the det condition); the clip only guards float
    rounding at the boundary.

    Returns g1, g2, g1_err, g2_err, g1g2_cov.  A non-finite
    e1e2_cov contributes nothing to the errors and gives a nan
    g1g2_cov
    """
    import numpy as np

    if np.isfinite(e1) and np.isfinite(e2):
        u = e1 * e1 + e2 * e2
        s = np.sqrt(max(1.0 - u, 0.0))
        f = 1.0 / (1.0 + s)
        g1 = e1 * f
        g2 = e2 * f

        # dg_i/de_j = f delta_ij + 2 e_i e_j f'; f' = df/d(e^2)
        if s > 0:
            fp = 1.0 / (2 * s * (1.0 + s) ** 2)
        else:
            fp = 0.0
        j11 = f + 2 * e1 * e1 * fp
        j12 = 2 * e1 * e2 * fp
        j21 = j12
        j22 = f + 2 * e2 * e2 * fp

        c11 = e1_err ** 2
        c22 = e2_err ** 2
        c12 = e1e2_cov if np.isfinite(e1e2_cov) else 0.0

        g1_err = np.sqrt(
            j11 ** 2 * c11 + j12 ** 2 * c22 + 2 * j11 * j12 * c12
        )
        g2_err = np.sqrt(
            j21 ** 2 * c11 + j22 ** 2 * c22 + 2 * j21 * j22 * c12
        )
        g1g2_cov = (
            j11 * j21 * c11 + j12 * j22 * c22
            + (j11 * j22 + j12 * j21) * e1e2_cov
        )
    else:
        g1, g2, g1_err, g2_err, g1g2_cov = [np.nan] * 5

    return g1, g2, g1_err, g2_err, g1g2_cov


def show_group(
    mbobs,
    seg,
    objects,
    group,
    title,
    pad=10,
):
    """
    show the kdeblend view_blend figure for one blend group: the
    region of the field bounding the group's stamps, with the
    fitted models rendered, the seg map as the fourth panel and the
    stamp boxes drawn on every panel, members labeled by catalog
    index.  At most the first three bands are shown.

    was actually fit; the box interior of every shown band is
    overwritten with those pixels so the display shows what the
    fitter saw -- in group-replace mode the external objects'
    pixels are noise there, not the original data
    """
    from kdeblend import vis

    vis.view_blend(
        mbobs,
        objects,
        bands=list(range(min(len(mbobs), 3))),
        seg=seg,
        labels=[str(i) for i in group],
        title=title,
        show=True,
    )


def anchor_covs(sxcat, group, nsx, scale):
    """
    per-member anchor position covariances in arcsec^2 with
    (v, u) ordering, from the sep centroid error moments
    (erry2/errxy/errx2 in pixels^2).  The anchor noise of the
    detection positions is a real error channel for recentered
    tight blends (it can double the flux variance at the
    detection-centroid scale).  Extra-detection members have no
    sep moments and get zero (their errors stay conditional on
    the injected positions).  Non-finite or non-positive-definite
    sep moments are sanitized: bad variances zeroed, the cross
    term clamped inside the PSD bound
    """
    import numpy as np

    out = np.zeros((len(group), 2, 2))
    for k, i in enumerate(group):
        if i >= nsx:
            continue
        vv = float(sxcat['erry2'][i])
        uu = float(sxcat['errx2'][i])
        vu = float(sxcat['errxy'][i])
        if not np.isfinite(vv) or vv < 0:
            vv = 0.0
        if not np.isfinite(uu) or uu < 0:
            uu = 0.0
        lim = 0.99 * np.sqrt(vv * uu)
        if not np.isfinite(vu):
            vu = 0.0
        vu = np.clip(vu, -lim, lim)
        out[k] = scale ** 2 * np.array([
            [vv, vu], [vu, uu],
        ])
    return out


def cut_group_mbobs(mbobs, sxcat, group, seg):
    """
    cut the shared deblending image for a group from every band:
    the box bounding the union of the members' seg-footprint
    bounding boxes, padded by GROUP_BOX_PAD pixels all around and
    clipped at the field edges.  The cut keeps the sky frame of
    the field (the jacobian centers are shifted by the cut
    origin), so the object v/u offsets are unchanged.  The noise
    fields are cut along with the images.

    the pixels of objects outside the group are
    replaced with values from the 180-degree-rotated noise field
    at the same box location: the rotation preserves a stationary
    covariance exactly (C(-d) = C(d), including the metacal
    anisotropy) and is independent of both the image noise and
    the attached noise field away from the field center.  The
    attached noise fields are not modified.

    Returns
    -------
    mbobs, box
        box is (row_start, col_start, nrow, ncol) of the cut in
        the field image
    """
    import numpy as np
    import ngmix

    obs0 = mbobs[0][0]
    dim_r, dim_c = obs0.image.shape

    r0 = max(int(sxcat['ymin'][group].min()) - GROUP_BOX_PAD, 0)
    r1 = min(
        int(sxcat['ymax'][group].max()) + GROUP_BOX_PAD + 1,
        dim_r,
    )
    c0 = max(int(sxcat['xmin'][group].min()) - GROUP_BOX_PAD, 0)
    c1 = min(
        int(sxcat['xmax'][group].max()) + GROUP_BOX_PAD + 1,
        dim_c,
    )

    nrow = r1 - r0
    ncol = c1 - c0
    sl = np.s_[r0:r1, c0:c1]

    segcut = seg[sl]
    foreign = (segcut != 0) & ~np.isin(
        segcut, sxcat['number'][group],
    )
    if not foreign.any():
        foreign = None

    out = ngmix.MultiBandObsList()
    for obslist in mbobs:
        obs = obslist[0]
        jrow, jcol = obs.jacobian.get_cen()
        jacobian = obs.jacobian.copy()
        jacobian.set_cen(row=jrow - r0, col=jcol - c0)

        image = obs.image[sl].copy()
        if foreign is not None:
            image[foreign] = np.rot90(obs.noise, 2)[sl][foreign]

        gobs = ngmix.Observation(
            image=image,
            weight=obs.weight[sl].copy(),
            jacobian=jacobian,
            noise=obs.noise[sl].copy(),
            psf=obs.psf,
            # psf=_cut_psf_obs(psf_obs=obs.psf, nrow=nrow, ncol=ncol),
        )
        ol = ngmix.ObsList()
        ol.append(gobs)
        out.append(ol)

    return out, segcut, (r0, c0, nrow, ncol)


def get_groups(sxcat, seg):
    """
    group objects by the union of seg-touching links (fofx) and
    moment-relevance links

    Parameters
    ----------
    sxcat: array with fields
        The sep catalog, with the 'number' field matching the seg
        map values
    seg: array
        The sep segmentation map

    Returns
    -------
    list of lists of catalog indices
    """
    import fofx

    nobj = sxcat.size
    if nobj == 0:
        # a heavily star-masked cell can legitimately detect
        # nothing; fofx crashes on an empty seg map
        return []
    parent = list(range(nobj))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    fofs = fofx.get_fofs(seg)
    number_to_index = {
        number: i for i, number in enumerate(sxcat['number'])
    }
    first = {}
    for fof_id, number in zip(fofs['fof_id'], fofs['number']):
        if number in number_to_index:
            i = number_to_index[number]
            if fof_id in first:
                union(first[fof_id], i)
            else:
                first[fof_id] = i

    groups = {}
    for i in range(nobj):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
