"""
per-cell processing orchestration
"""
from .coadd import coadd_mbobs
from .deblend import fit_deblend
from .defaults import PSF_FAILURE
from .detect import run_sep
from .extra_detect import get_s2_extra_detections
import numpy as np
from .maxlike import do_single_fits
from .metacal import do_all_metacal
from .mfrac import sample_mfrac, smooth_mfrac_map
from .psf import _set_mcal_psfs, fit_and_set_mcal_psfs
from .structs import get_struct
from .cells import get_cell_primary


def do_metacal_and_process(mbobs, model, deblend, s2_detect, rng, show):
    odict = do_all_metacal(mbobs=mbobs, rng=rng)

    dlist = []
    for key, mcal_mbobs in odict.items():
        st = process_one_mbobs(
            mbobs=mcal_mbobs,
            model=model,
            deblend=deblend,
            s2_detect=s2_detect,
            rng=rng,
            show=show,
        )

        st['mcal_step'] = 'ns' if key == 'noshear' else key
        dlist.append(st)

    return np.concatenate(dlist)


def process_one_mbobs(mbobs, model, deblend, s2_detect, rng, show):
    bands = [obslist[0].meta['band'] for obslist in mbobs]
    detect_obs, weights = coadd_mbobs(mbobs)
    sxcat, seg = run_sep(detect_obs)
    nsx = sxcat.size

    psf_res = fit_and_set_mcal_psfs(
        mbobs=mbobs, weights=weights, rng=rng,
    )

    # we can't do  extra detections without the psf
    if psf_res['psf_flags'] == 0:

        if s2_detect:
            extra_detections = get_s2_extra_detections(
                mbobs=mbobs,
                detobs=detect_obs,
                sxcat=sxcat,
                seg=seg,
                rng=rng,
                prior_extras=None,
            )
            n_extra = len(extra_detections)
        else:
            extra_detections = None
            n_extra = 0

        cat = get_struct(bands=bands, n=sxcat.size + n_extra)

        cat['xcell'][:nsx] = sxcat['x']
        cat['ycell'][:nsx] = sxcat['y']

        if n_extra > 0:
            cat['xcell'][nsx:] = [e[0] for e in extra_detections]
            cat['ycell'][nsx:] = [e[1] for e in extra_detections]

        _set_mcal_psfs(st=cat, psf_res=psf_res)

        if not deblend:
            do_single_fits(
                mbobs=mbobs,
                sxcat=sxcat,
                cat=cat,
                model=model,
                rng=rng,
            )
        else:
            fit_deblend(
                mbobs=mbobs,
                sxcat=sxcat,
                cat=cat,
                seg=seg,
                model=model,
                rng=rng,
                extra_detections=extra_detections,
                show=show,
            )

    else:
        print('psf_failure:', psf_res['psf_flags'])

        import matplotlib.pyplot as mplt
        fig, axs = mplt.subplots(ncols=3)
        for i, obslist in enumerate(mbobs):
            axs[i].imshow(obslist[0].psf.image)
            axs[i].set_title(obslist[0].meta['band'])
        fig.savefig('bad-psfs.png', dpi=150)

        # can't do extra without a psf
        cat = get_struct(bands=bands, n=sxcat.size)

        cat['xcell'] = sxcat['x']
        cat['ycell'] = sxcat['y']

        cat['flags'] = PSF_FAILURE

    # one mfrac for every row (sep, deblend groups, extras)
    # from the smoothed map: equivalent to the per-stamp
    # gaussian-weighted mean, at one convolution per cell
    mfrac_map = smooth_mfrac_map(mbobs)
    cat['mfrac'] = sample_mfrac(
        mfrac_map, cat['xcell'], cat['ycell'],
    )

    cat['is_primary'] = get_cell_primary(cat['xcell'], cat['ycell'])
    return cat
