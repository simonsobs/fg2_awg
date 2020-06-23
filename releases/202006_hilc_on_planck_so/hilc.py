import os
import os.path as op
import sys
import yaml
import logging
import h5py as h5
import numpy as np
import healpy as hp
import fgbuster as fgb

FREQUENCIES_SO = np.array([27., 39., 93., 145., 225., 280.])
FREQUENCIES_PLANCK = np.array([30., 44, 70, 100, 143, 217, 353, 545, 857])
AMIN_OUT = 2


def write(*res, outdir=None, nside=None):
    """scipy.optimize.res -> files
    """
    logging.info(f'Save output to {outdir}')
    alms = np.stack([r.s[0] for r in res])
    _make_parent_dir_if_not_existing(op.join(outdir, 'cmb_alms.fits'))
    np.savez(op.join(outdir, 'ilc_filter'), T=res[0].W, E=res[1].W, B=res[2].W)
    np.savez(op.join(outdir, 'cl_in'),
             T=res[0].cl_in, E=res[1].cl_in, B=res[2].cl_in, fsky=res[0].fsky)
    np.savez(op.join(outdir, 'cl_out'),
             T=res[0].cl_out, E=res[1].cl_out, B=res[2].cl_out,
             fsky=res[0].fsky)
    hp.write_alm(op.join(outdir, 'cmb_alms.fits'), alms, overwrite=True)
    hp.write_map(op.join(outdir, f'cmb_fwhm_{AMIN_OUT}_amin.fits'),
                 hp.alm2map(alms, nside, fwhm=np.radians(AMIN_OUT/60)),
                 overwrite=True)

def apply_lmax(alm, lmax):
    ell, m = hp.Alm.getlm(lmax)
    lmax_in = hp.Alm.getlmax(alm.shape[-1])
    idx = hp.Alm.getidx(lmax_in, ell, m)
    return alm[..., idx]


def get_alms(field=None, cache_alms=None, components=None, instruments=None,
             lmax=None):
    """Retrieve and sum the components
    """
    freqs = get_freqs(instruments)
    alms = np.zeros((len(freqs), hp.Alm.getsize(lmax)), dtype=np.complex128)

    keep_freqss = np.full((len(components), len(freqs), 1), True)
    with h5.File(cache_alms, 'r') as f_alm:
        for component, keep_freqs in zip(components, keep_freqss):
            logging.info(f'Loading {component}')
            for freq, alm, keep_freq in zip(freqs, alms, keep_freqs):
                logging.info(f'Loading {freq}')
                try:
                    if field and int(freq) in [545, 857]:
                        raise ValueError
                    alm += apply_lmax(f_alm[f'{component}/{int(freq)}'][field],
                                      lmax)
                except ValueError:
                    logging.info(f'Flagging frequency {freq}')
                    keep_freq[0] = False

    keep_freqs = keep_freqss.any(0).reshape(-1)
    removed = 0
    for i in range(len(freqs)):
        if keep_freqs[i]:
            if removed:
                alms[i-removed] = alms[i]
        else:
            logging.info(f"Removing {freqs[i]}")
            removed += 1
    freqs = np.array(freqs)[keep_freqs]
    alms = alms[:len(freqs)]
    return alms, freqs

def get_freqs(instruments):
    """Retrieve and sum the components
    """
    freqs = []
    for instrument in instruments:
        if instrument == 'so':
            freqs += list(FREQUENCIES_SO)
        if instrument == 'planck':
            freqs += list(FREQUENCIES_PLANCK)
    return np.array(freqs)

def _make_parent_dir_if_not_existing(filename, verbose=True):
    dirname = op.dirname(filename)
    if not op.exists(dirname):
        if verbose:
            logging.debug(f"Creating the folder {dirname} for {filename}")
        os.makedirs(dirname)

def set_logfile(log_file, level=None):
    _make_parent_dir_if_not_existing(log_file, verbose=False)
    logging.basicConfig(
        level=(logging.root.level if level is None else level),
        format='%(asctime)-6s %(levelname)-6s %(message)s',
        datefmt='%H:%M:%S', handlers=[logging.FileHandler(log_file),
                                      logging.StreamHandler()])

def main():
    # Parse logging and IO
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)

    if params['subdir']:
        subdir = '_'.join(params['inputs']['instruments'])
        subdir += '__' + '_'.join(params['inputs']['components'])
        params['outdir'] = op.join(params['outdir'], subdir)

    set_logfile(op.join(params['outdir'], f'run.log'), params['logging_level'])
    logging.info(f"PARAMETERS READ\n\n{yaml.dump(params)}\n\n")
    del params['logging_level'], params['subdir']

    # HILC
    res = []
    for i in range(3):  # Avoid loading TEB at the same time
        logging.info(f'Stokes {i+1}/3')
        alms, freqs = get_alms(i, **params['inputs'])
        instrument = fgb.observation_helpers.standardize_instrument(
            {'frequency': freqs})
        logging.info('Do ILC')
        res.append(fgb.separation_recipes._harmonic_ilc_alm(
            [fgb.CMB()], instrument, alms, **params['hilc']))
        res[-1].freqs = freqs
    logging.info('Write output')
    write(*res, outdir=params['outdir'], nside=params['nside'])
    try:
        params['inputs']['components'].remove('cmb')
    except ValueError:
        return
    for i in range(3):  # Avoid loading TEB at the same time
        logging.info(f'Noise, Stokes {i+1}/3')
        alms, freqs = get_alms(i, **params['inputs'])
        logging.info('Apply filter')
        res[i].s = fgb.separation_recipes._apply_harmonic_W(res[i].W, alms)
        logging.info('Compute spectra')
        res[i].cl_in = np.array([hp.alm2cl(alm) for alm in alms])
        res[i].cl_out = np.array([hp.alm2cl(alm) for alm in res[i].s])
        if res[i].fsky is not None:
            res[i].cl_in /= res[i].fsky
            res[i].cl_out /= res[i].fsky

    logging.info('Write output')
    write(*res, outdir=op.join(params['outdir'], 'residuals'),
          nside=params['nside'])


if __name__ == '__main__':
    main()
