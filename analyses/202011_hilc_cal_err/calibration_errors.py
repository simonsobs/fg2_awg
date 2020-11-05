import numpy as np
import fgbuster as fgb
import healpy as hp
import v3_calc as v3
import logging

ALMS_NERSC = '../../releases/202006_hilc_on_planck_so/workspace/cache/alms_{comp}.h5'
NOISE_COV_MATRIX_LOCAL = 'fg_noise.npz'
CMB_SPECTRA_LOCAL = 'cmb.npz'
LMAX = 4000
BIN_WIDTH = 20
FIELDS = 'TT EE BB'.split()
FSKY = 0.3922505853468785  # mean(w^2) / mean(w^4)

def get_bias(delta, field, freqs=v3.Simons_Observatory_V3_LA_bands()):
    """ Post-HILC CMB bias

    Parameters
    ----------
    delta: array
        Array of shape (..., ell, freq) -- or boadcastable to it. The SED of the
        CMB is modeled as flat and equal to 1 (K_CMB units). However, we assume
        that the actual response to the CMB in the data is (1 + delta). The
        array delta can specifys these correction for each fequency and/or
        scales. Examples of shapes,
        * (6,) or (1, 6) -> scale independent, frequency-specific correction
        * (200, 1) -> frequency-independent, scale-specific correction
        * (1000, 200, 6)
          Stack of 1000 correction factors that are scale- and
          frequency-specific
        Note: the size of the ell dimension is hardcoded and equal to 200. Run
        get_bias(0, 'TT') to get the reference ell for each index

    field: str
        Either TT, EE or BB

    Result
    _____
    bias: array
        Bias of the reconstructed CMB at each scale.
        Same shape as delta except for the frequency dimension.

    """
    invN = get_invN(freqs, field)  # (ell, freq, freq)

    cmb_dot_cmb = invN.sum((-1, -2))
    cmb_dot_delta = np.einsum('...lf,...lfn->...l', delta, invN)
    delta_dot_delta = np.einsum('...lf,...lfn,...ln->...l',
                                delta, invN, delta)

    data = np.load(CMB_SPECTRA_LOCAL)
    cmb_ps = data[field]
    ells = data['ells']
    numerator = 1 + cmb_dot_delta / cmb_dot_cmb
    denominator = 1 + cmb_ps * (delta_dot_delta - cmb_dot_delta**2 / cmb_dot_cmb)
    return ells, numerator, denominator


def _import_get_alm():
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir('../../releases/202006_hilc_on_planck_so/')
    from hilc import get_alms
    os.chdir(dir_path)
    return get_alms


def _create_cached_noise_matrix():
    logging.getLogger().setLevel(logging.INFO)
    get_alms = _import_get_alm()
    binned_covs = []
    freqs = []
    for field in range(3):
        alms, freq = get_alms(
            field, ALMS_NERSC,
            'tsz ksz cib synchrotron freefree ame dust noise'.split(),
            'so planck'.split(),
            lmax=LMAX
        )
        cov = fgb.separation_recipes._empirical_harmonic_covariance(alms)
        lbins = np.arange(1, LMAX+BIN_WIDTH, BIN_WIDTH)
        lbins[-1] = LMAX+1
        binned_cov = np.empty(cov.shape[:-1] + lbins[:-1].shape)
        logging.info(f'{FIELDS[field]} cov')
        for i, (lmin, lmax) in enumerate(zip(lbins[:-1], lbins[1:])):
            # Average the covariances in the bin
            lmax = min(lmax, cov.shape[-1])
            dof = 2 * np.arange(lmin, lmax) + 1
            binned_cov[..., i] = (dof / dof.sum() * cov[..., lmin:lmax]).sum(-1)

        freqs.append(freq)
        binned_covs.append(binned_cov/FSKY)

    logging.info('Saving')
    np.savez(NOISE_COV_MATRIX_LOCAL,
        TT=binned_covs[0],
        EE=binned_covs[1],
        BB=binned_covs[2],
        ells=(lbins[:-1] + lbins[1:]) / 2.,
        freq_TT=freqs[0],
        freq_EE=freqs[1],
        freq_BB=freqs[2],
    )
    logging.info('Saved')


def _create_cached_cmb_spectrum():
    get_alms = _import_get_alm()
    binned_cls = []
    for field in range(3):
        alms = get_alms(field, ALMS_NERSC, ['cmb'], ['so'], lmax=LMAX)[0][-1]
        cl = hp.alm2cl(alms)
        lbins = np.arange(1, LMAX+BIN_WIDTH, BIN_WIDTH)
        lbins[-1] = LMAX+1
        binned_cl = np.empty(cl.shape[:-1] + lbins[:-1].shape)
        for i, (lmin, lmax) in enumerate(zip(lbins[:-1], lbins[1:])):
            # Average the covariances in the bin
            lmax = min(lmax, cl.shape[-1])
            dof = 2 * np.arange(lmin, lmax) + 1
            binned_cl[..., i] = (dof / dof.sum() * cl[..., lmin:lmax]).sum(-1)

        binned_cls.append(binned_cl/FSKY)

    np.savez(CMB_SPECTRA_LOCAL,
        TT=binned_cls[0],
        EE=binned_cls[1],
        BB=binned_cls[2],
        ells=(lbins[:-1] + lbins[1:]) / 2.,
    )


def get_invN(freqs, field):
    try:
        assert np.all(get_invN.freqs == freqs)
        assert field in get_invN.invN
    except (AttributeError, AssertionError):
        get_invN.freqs = freqs
        data = np.load(NOISE_COV_MATRIX_LOCAL)
        tot_freqs = list(data[f'freq_{field}'].astype(int))
        freq_idx = np.array([tot_freqs.index(f) for f in freqs])
        N = data[field][freq_idx][:, freq_idx]
        if not hasattr(get_invN, 'invN'):
            get_invN.invN = {}
        get_invN.invN[field] = fgb.separation_recipes._regularized_inverse(N.T)
    return get_invN.invN[field]
