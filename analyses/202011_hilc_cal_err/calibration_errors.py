import numpy as np
import fgbuster as fgb
import healpy as hp
import v3_calc as v3
import logging

ALMS_NERSC = '../../releases/202006_hilc_on_planck_so/workspace/cache/alms_{comp}.h5'
NOISE_COV_MATRIX_LOCAL = 'fg.npz'
CMB_SPECTRA_LOCAL = 'cmb.npz'
LMAX = 4000
BIN_WIDTH = 20
FIELDS = 'TT EE BB'.split()
FSKY = 0.3922505853468785  # mean(w^2) / mean(w^4)

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
            'tsz ksz cib synchrotron freefree ame dust'.split(),
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
        assert get_invN.freqs == freqs
    except (AttributeError, AssertionError):
        get_invN.freqs = freqs
        data = np.load(NOISE_COV_MATRIX_LOCAL)
        tot_freqs = list(data['freqs'].astype(int))
        freq_idx = np.array([tot_freqs.index(f) for f in freqs])
        N = data[FIELDS[field]][freq_idx][:, freq_idx]
        get_invN.invN = fgb.separation_recipes._regularized_inverse(N)
    return get_invN.invN


def get_bias(delta, freqs=v3.Simons_Observatory_V3_LA_bands()[0]):
    invN = fgb.separation_recipes._regularized_inverse(
        np.load(NOISE_COV_MATRIX_LOCAL))

    cmb_dot_cmb = invN.sum((-1, -2))
    cmb_dot_delta = (delta[None] @ invN.sum(-1))[..., 0]
    delta_dot_delta = (delta[None] @ invN @ delta[..., None])[..., 0, 0]

    cmb_ps = np.load(CMB_SPECTRA_LOCAL)
    numerator = 1 + cmb_dot_delta / cmb_dot_cmb
    denominator = 1 + cmb_ps * (delta_dot_delta - cmb_dot_delta**2 / cmb_dot_cmb)
    return numerator, denominator

