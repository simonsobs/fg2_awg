""" HILC run for a given scan strategy

Given
* weights (i.e. scan strategy)
* input components
* power spectrum parameters (lmax, delta_ell, etc.)

Run HILC. Output
* CMB alms
* CMB map
* Empirical covariance

Output structure
./workspace
    inputs/  # Signal only
        comp_list_1/
            map_XX_GHz.fits
    runs/
        scan_1/
            weights.fits
            cov.npz
            inputs/  # Signal and noise
                comp_list_1/
                    alm_XX_GHz.fits
            outputs/
                map_cmb.fits
                alm_cmb.fits
"""
import sys
import os
import os.path as op
from glob import glob
import numpy as np
import logging
import yaml

import healpy as hp
import pysm3.units as u
import fgbuster as fgb

#FIXME: load these data from mapsims
FREQUENCIES_SO = '27, 39, 93, 145, 225, 280'.split(', ')
BEAMS_SO = [7.4, 5.1, 2.2, 1.4, 1.0, 0.9]

FREQUENCIES_PLANCK = '30, 44, 70, 100, 143, 217, 353, 545, 857'.split(', ')
BEAMS_PLANCK = [33.102652125, 27.94348615, 13.07645961, 9.682, 7.303, 5.021,
                4.944, 4.831, 4.638]

FREQUENCIES = FREQUENCIES_SO + FREQUENCIES_PLANCK
BEAMS = BEAMS_SO + BEAMS_PLANCK


def _make_parent_dir_if_not_existing(filename):
    dirname = op.dirname(filename)
    if not op.exists(dirname):
        logging.info(f"Creating the folder {dirname} for {filename}")
        os.makedirs(dirname)

def get_weights():
    return

def _so_or_planck_freq(freq):
    if freq in FREQUENCIES_SO:
        return 'so'
    if freq in FREQUENCIES_PLANCK:
        return 'planck'

    raise ValueError(
        f"{freq} GHz is neither a SO or Planck frequency\n"
        f"SO: {list(FREQUENCIES_SO)}\n"
        f"Planck: {list(FREQUENCIES_PLANCK)}"
    )


def prepare_and_get_signal_map_file(freq, comps, paths, workspace):
    comps = [c for c in comps if c != 'noise']
    cached = op.join(
        f"{workspace}/inputs/{'_'.join(sorted(comps))}/map_{freq:0<3}_GHz.fits")
    if op.exists(cached):
        return cached

    logging.info(f"Could not find {cached}, computing")

    stack = None
    for comp in comps:
        comp_file = paths[_so_or_planck_freq(freq)]
        comp_file = [f for f in glob(comp_file.replace('*', comp+'*', 2))
                     if f'{freq:0<3}' in f]
        assert len(comp_file) == 0, (
            "{len(comp_file)} matches for {freq} {comp}")
        comp_file = comp_file[0]
        logging.info(f" - Reading {comp_file}")
        if stack is None:
            stack = hp.read_map(comp_file, field=None)
        else:
            stack += hp.read_map(comp_file, field=None)
    logging.info(f"Saving {cached}")
    _make_parent_dir_if_not_existing(cached)
    hp.write_map(cached, stack)
    return cached


def _planck_noise_to_C_and_uKCMB(alm, freq):
    freq = int(freq)
    if freq in '545 857':
        logging.info(" - MJy/sr -> uKCMB")
        alm *= (1 * u.MJy / u.sr).to(
            u.uK_CMB, equivalencies=u.cmb_equivalencies(float(freq) * u.GHz)
        ).value
    elif freq in '30 44 70 100 143 217 353':
        logging.info(" - KCMB -> uKCMB")
        alm *= 1e6
    else:
        raise ValueError(
            f"{freq} not in Planck frequencies: {FREQUENCIES_PLANCK}")
    logging.info(" - Rotating: G -> C")
    hp.Rotator(coord='gc').rotate_alm(alm, inplace=True)


def _file2alm(weights, lmax, fwhm, map_file):
    logging.info(f"Loading {map_file}")
    try:
        maps = hp.read_map(map_file, field=(0, 1, 2))
        logging.info("- T, Q, U")
    except IndexError:
        maps = hp.read_map(map_file)
        logging.info("- T-only")

    nside_map = hp.get_nside(maps)
    nside = hp.get_nside(weights)  # NOTE: the weights define the nside
    if nside_map != nside:
        logging.warning(
            f"Maps have nside {nside} but weights have nside "
            f"{nside_weights}. ud_grading the former to the latter")
        maps = hp.ud_grade(maps, nside)
    
    if lmax > 3 * nside - 1:
        logging.warning(
            f"Lmax is {lmax} but the nside is only {nside}. Computing alms "
            f"up to {3 * nside-1} and filling the rest with white noise")
        logging.info("Computing Alms")
        alm = hp.map2alm(hp.ma(maps)*weights, lmax=3*nside-1)
        pad_width = [(0, lmax - 3 * nside + 1)]
        if alm.ndim == 2:
            pad_width = [(0, 0)] + pad_width
        cl = np.pad(hp.alm2cl(alm), pad_width,
                    mode='edge')  # Extend with the power of the last multipole
        if maps.ndim == 2:  # Set TE, TB and EB to zero in the added alm
            cl[3:, 3*nside_map:] = 0
        full_alm = hp.synalm(cl, new=True)
        # Indices of the computed alms in the extended alms
        full_idx = hp.Alm.getidx(lmax, *hp.Alm.getlm(3*nside-1))
        full_alm[..., full_idx] = alm  # Ellipsis to handle T-only or TEB
        alm = full_alm
    else:
        logging.info("Computing Alms")
        alm = hp.map2alm(hp.ma(maps)*weights, lmax=lmax)
    bl = hp.gauss_beam(np.radians(fwhm/60.0), lmax, pol=(alm.ndim == 2))
    if alm.ndim == 1:
        alm = [alm]
        bl = [bl]
    else:
        bl = bl.T

    for i_alm, i_bl in zip(alm, bl):
        hp.almxfl(i_alm, 1.0/i_bl, inplace=True)

    return alm


def get_noise_alms(hdu, freq, weights, lmax, paths, workspace_run):
    cached = op.join(
        f"{workspace_run}/inputs/noise/alm_{freq:0<3}_GHz.fits")
    try:
        logging.info(f"Reading {cached}")
        return hp.read_alm(cached, hdu)
    except IOError:
        logging.info(f"Could not find {cached}, computing")

    experiment = _so_or_planck_freq(freq)
    fwhm = BEAMS[FREQUENCIES.index(freq)]
    if experiment == 'so':
        NotImplementedError("Scan strategy stuff")
    else:
        noise_file = paths['planck_noise']
        noise_file = [f for f in glob(noise_file.replace('*', 'noise', 1))
                      if f'{freq:0<3}' in f]
        assert len(noise_file) == 0, (
            "{len(noise_file)} matches for {freq} {comp}")
        noise_file = noise_file[0]
        alm = _file2alm(weights, lmax, fwhm, noise_file)
        _planck_noise_to_C_and_uKCMB(alm, freq)

    logging.info(f"Saving {cached}")
    _make_parent_dir_if_not_existing(cached)
    hp.write_alm(cached, alm)

    return alm[hdu-1]


def _boost_ell_lt_30(alm):
    logging.info(f"Boosting ell < 30")
    l, m = np.array([[l, m] for l in range(30) for m in range(l)]).T
    idx = hp.Alm.getidx(hp.Alm.getlmax(alm.shape[-1]), l, m)
    l[l == 0] = 1
    alm[..., idx] *= (3 / (1 + np.exp(l - 30)) + 1) * (l/30)**(-2)


def _get_scan_tag(**scan_kw):
    return None


def _get_weights(**scan_kw):
    return None


def get_alm(hdu, freq, weights, lmax, comps, paths, workspace_run, workspace):
    cached = op.join(
        f"{workspace_run}/inputs/{'_'.join(sorted(comps))}/alm_{freq:4d}.fits")
    try:
        logging.info(f"Reading {cached}")
        return hp.read_alm(cached, hdu=hdu)
    except IOError:
        logging.info(f"Could not find {cached}, computing")

    fwhm = BEAMS[FREQUENCIES.index(freq)]
    map_file = prepare_and_get_signal_map_file(freq, comps, paths, workspace)
    alm = _file2alm(weights, lmax, fwhm, map_file)
    if 'noise' in comps:
        alm += get_noise_alms([1, 2, 2], freq, weights, lmax,
                              paths, workspace_run)

    logging.info(f"Saving {cached}")
    _make_parent_dir_if_not_existing(cached)
    hp.write_alm(cached, alm)

    return alm[hdu-1]


def get_all_alm(hdu, freqs, weights, lmax, comps, paths, workspace_run, workspace):
    shape = freqs.shape + np.array(hdu).shape + (hp.Alm.getsize(lmax),)
    alms = np.zeros(shape, dtype=np.complex128)
    for alm, freq in zip(alms, freqs):
        alm[:] = get_alm(hdu, freq, weights, lmax, comps,
                         paths, workspace_run, workspace)

    return alm


def set_logfile(log_file, level=None):
    _make_parent_dir_if_not_existing(log_file)
    logging.basicConfig(
        level=(logging.root.level if level is None else level),
        format='%(asctime)-6s %(levelname)-6s %(message)s',
        datefmt='%H:%M:%S', handlers=[logging.FileHandler(log_file),
                                      logging.StreamHandler()])


def write(*res, outdir=None, nside=None, amin_out=2):
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
    hp.write_map(op.join(outdir, f'cmb_fwhm_{amin_out}_amin.fits'),
                 hp.alm2map(alms, nside, fwhm=np.radians(amin_out/60)),
                 overwrite=True)


def main():
    # Parse logging and IO
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)

    workspace_run = f"{params['workspace']}/runs/{_get_scan_tag(params['scan'])}"
    set_logfile(f"{workspace_run}/run.log", params['logging_level'])
    logging.info(f"PARAMETERS READ\n\n{yaml.dump(params)}\n\n")

    weights = get_weights(params['scan'], workspace_run)

    freqs = []
    if 'so' in params['instruments']:
        freqs += FREQUENCIES_SO
    if 'planck' in params['instruments']:
        freqs += FREQUENCIES_PLANCK

    # HILC
    res = []
    for hdu in range(1, 4):  # Avoid loading TEB at the same time
        logging.info(f'Stokes {hdu}/3')
        alms = get_all_alm(
            hdu, freqs, weights, params['lmax'], params['components'],
            params['input'], workspace_run, params['workspace'])
        instrument = fgb.observation_helpers.standardize_instrument(
            {'frequency': np.array(freqs).astype(float)})
        logging.info('Do ILC')
        res.append(fgb.separation_recipes._harmonic_ilc_alm(
            [fgb.CMB()], instrument, alms, **params['hilc']))
        res[-1].freqs = freqs

    logging.info('Write output')
    write(*res, outdir=workspace_run, nside=params['nside'],
          amin_out=params['amin_out'])
    try:
        params['inputs']['components'].remove('cmb')
    except ValueError:
        return
    for hdu, res_stokes in zip(range(1, 4), res):
        logging.info(f'Noise, Stokes {hdu}/3')
        alms = get_all_alm(
            hdu, freqs, weights, params['lmax'], params['components'],
            params['input'], workspace_run, params['workspace'])
        logging.info('Apply filter')
        res_stokes.s = fgb.separation_recipes._apply_harmonic_W(res_stokes.W,
                                                                alms)
        logging.info('Compute spectra')
        res_stokes.cl_in = np.array([hp.alm2cl(alm) for alm in alms])
        res_stokes.cl_out = np.array([hp.alm2cl(alm) for alm in res_stokes.s])
        if res_stokes.fsky is not None:
            res_stokes.cl_in /= res_stokes.fsky
            res_stokes.cl_out /= res_stokes.fsky

    logging.info('Write output')
    write(*res, outdir=op.join(params['outdir'], 'residuals'),
          nside=params['nside'])


if __name__ == '__main__':
    main()
