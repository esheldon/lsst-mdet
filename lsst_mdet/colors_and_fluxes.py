import numpy as np


def set_fluxes(st, bands, flux, flux_err):
    """
    fill the flux_{band} and flux_err_{band} columns; the fitters
    return a scalar for a single band and an array for multiple
    """
    import numpy as np

    flux = np.atleast_1d(flux)
    flux_err = np.atleast_1d(flux_err)
    if flux.size != len(bands):
        raise ValueError(
            f'got {flux.size} fluxes for {len(bands)} bands'
        )
    for iband, band in enumerate(bands):
        st[f'flux_{band}'] = flux[iband]
        st[f'flux_err_{band}'] = flux_err[iband]


def set_colors(st, bands, flux, flux_err, flux_cov):
    """
    Set colors as well as errors based on the full covariance

    For the case of flux_cov None (e.g. for stars), the diagonal
    errors are used
    """
    fac = 2.5 / np.log(10)
    eps = 1.0e-7

    nband = len(bands)

    if flux_cov is None:
        flux_cov = np.diag(flux_err ** 2)

    for i in range(nband - 1):
        first_band = bands[i]
        second_band = bands[i + 1]
        cname = f'{first_band}m{second_band}'

        if flux[i] > eps and flux[i + 1] > eps:
            color = -2.5 * np.log10(flux[i] / flux[i + 1])

            color_var = fac ** 2 * (
                flux_cov[i, i] / flux[i] ** 2
                + flux_cov[i + 1, i + 1] / flux[i + 1] ** 2
                - 2 * flux_cov[i, i + 1] / (flux[i] * flux[i + 1])
            )

            st[cname] = color
            st[f'{cname}_err'] = np.sqrt(color_var)
