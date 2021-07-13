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
import matplotlib.pyplot as plt

import healpy as hp
import pysm3.units as u
import fgbuster as fgb

#FIXME: load these data from mapsims
FREQUENCIES_SO = '27, 39, 93, 145, 225, 280'.split(', ')
BEAMS_SO = [7.4, 5.1, 2.2, 1.4, 1.0, 0.9]
uK_AMIN_SO = [71.0,    36.0,   8.0,    10.0,   22.0,   54.0]
FREQ_TAG_SO = {
    '27': 'LF1',
    '39': 'LF2',
    '93': 'MFF1',
    '145': 'MFF2',
    '225': 'UHF1',
    '280': 'UHF2',
}

FREQUENCIES_PLANCK = '30, 44, 70, 100, 143, 217, 353, 545, 857'.split(', ')
BEAMS_PLANCK = [33.102652125, 27.94348615, 13.07645961, 9.682, 7.303, 5.021,
                4.944, 4.831, 4.638]

FREQUENCIES = FREQUENCIES_SO + FREQUENCIES_PLANCK
BEAMS = BEAMS_SO + BEAMS_PLANCK

THRESHOLD_WEIGHTS = 1e-4


def _make_parent_dir_if_not_existing(filename):
    dirname = op.dirname(filename)
    if not op.exists(dirname):
        logging.info(f"Creating the folder {dirname} for {filename}")
        os.makedirs(dirname)


def get_weights(workspace_run=None, nside=None, hits_map=None, gal_mask=None,
                rel_threshold=None, homogeneous=True, **kw):
    if not (hits_map and gal_mask):
        return np.ones(hp.nside2npix(nside))

    if workspace_run is None:
        logging.warning(f"Not loading/storing the weights")
    else:
        weights_file = op.join(workspace_run, 'weights.fits')
        if op.exists(weights_file):
            logging.info(f"Loading weights {weights_file}")
            return hp.read_map(weights_file)

    logging.info("Computing weights")

    mask_gal = hp.read_map(gal_mask) > 0.
    hits = hp.read_map(hits_map)
    if nside is None:
        nside = hp.get_nside(hits)
    elif hp.get_nside(hits) != nside:
        # upgrading with map2alm + alm2map is smoother than ud_grade
        hits = hp.map2alm(hits, iter=5)
        hits = hp.alm2map(hits, nside)
    mask_hits = hits > rel_threshold * hits.max()

    if hp.get_nside(mask_gal) != nside:
        mask_gal = hp.ud_grade(mask_gal, nside)
        
    mask = mask_gal * mask_hits
    lat_lim = kw.get('lat_lim')
    if lat_lim:
        lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)[1]
        mask *= (lat > lat_lim[0]) * (lat < lat_lim[1])

    apo_deg = kw.get('apo_deg')
    if homogeneous:
        logging.info("Homogeneous weights")
        weights = mask.astype(float)
    else:
        logging.info("Hits-based weights")
        weights = hits * mask / hits.max()
    weights = hp.smoothing(weights, np.radians(apo_deg))
    weights *= weights > THRESHOLD_WEIGHTS
    
    if workspace_run:
        logging.info(f"Saving weights {weights_file}")
        _make_parent_dir_if_not_existing(weights_file)
        hp.write_map(weights_file, weights)
        hp.mollview(weights, title='Weights')
        plt.savefig(weights_file.replace('.fits', '.pdf'))
        plt.close()

    return weights

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


def prepare_and_get_signal_map_file(freq, comps, paths, workspace, nside):
    comps = [c for c in comps if c != 'noise']
    if len(comps) == 0:
        logging.info("No components to be loaded")
        return

    cached = op.join(
        f"{workspace}/inputs/{'_'.join(sorted(comps))}/map_{freq:0>3}_GHz.fits")
    if op.exists(cached):
        return cached

    logging.info(f"Could not find {cached}, computing")

    stack = None
    for comp in comps:
        if comp == 'cib' and freq == '27':
            logging.info("Skipping the missing CIB at 27 GHz")
            continue

        comp_regex = paths[_so_or_planck_freq(freq)].replace('*', comp, 1)
        comp_file = [f for f in glob(comp_regex)
                     if f'{freq:0>3}' in f or FREQ_TAG_SO.get(freq, 'None') in f]
        if len(comp_file) == 0:
            alternate_regex = comp_regex.replace(
                '201905_extragalactic',
                '201904_highres_foregrounds_equatorial')
            comp_file = [f for f in glob(alternate_regex)
                         if f'{freq:0>3}' in f or FREQ_TAG_SO.get(freq, 'None') in f]

        assert len(comp_file) == 1, (
            f"{len(comp_file)} matches for {comp_regex}")

        comp_file = comp_file[0]
        logging.info(f" - Reading {comp_file}")
        if stack is None:
            stack = hp.read_map(comp_file, field=None)
        else:
            try:
                comp_map = hp.read_map(comp_file, field=(0, 1, 2))
                stack += comp_map
            except IndexError:  # No polarization in the comp
                if stack.ndim == 1:  # No polarization in the stack
                    stack += hp.read_map(comp_file, field=None)
                else:
                    stack[0] += hp.read_map(comp_file, field=None)
            except ValueError:  # Polarization in the comp, not in the stack
                comp_map[0] += stack
                stack = comp_map

    if nside != hp.get_nside(stack):
        logging.info(f"{hp.get_nside(stack)} -> {nside}")
        stack = hp.alm2map(hp.map2alm(stack, lmax=3*nside-1), nside=nside)

    logging.info(f"Saving {cached}")
    _make_parent_dir_if_not_existing(cached)
    hp.write_map(cached, stack)
    return cached


def _planck_noise_to_C_and_uKCMB(alm, freq):
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
    return alm


def _file2alm(weights, lmax, fwhm, map_file, alm_modifier=None):
    logging.info(f"Loading {map_file}")
    try:
        maps = hp.read_map(map_file, field=(0, 1, 2))
        logging.info("- T, Q, U")
    except IndexError:
        maps = hp.read_map(map_file)
        logging.info("- T-only")

    nside_map = hp.get_nside(maps)
    nside = hp.get_nside(weights)  # NOTE: the weights define the nside
    if alm_modifier is not None:  # NOTE: alm modified on map *before* weights
        maps = hp.alm2map(alm_modifier(hp.map2alm(maps)), nside_map)
    if nside_map != nside:
        logging.warning(
            f"Maps have nside {nside_map} but weights have nside "
            f"{nside}. ud_grading the former to the latter")
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
        alm = alm[None]
        bl = bl[None]
    else:
        bl = bl.T

    for i_alm, i_bl in zip(alm, bl):
        hp.almxfl(i_alm, 1.0/i_bl, inplace=True)

    return alm


def _save_maps_from_alm_file(alm_file, nside):
    try:
        alms = hp.read_alm(alm_file, (1, 2, 3))
    except IndexError:
        alms = hp.read_alm(alm_file, (1,))

    maps = hp.alm2map(alms, nside)
    maps = maps.reshape(-1, maps.shape[-1])
    try:
        for i, stokes in enumerate('TQU'):
            hp.mollview(maps[i])
            plt.savefig(alm_file.replace('.fits', f'_{stokes}_map.pdf'))
            plt.close()
        for i, stokes in enumerate('EB'):
            hp.mollview(hp.alm2map(alms[i+1], nside))
            plt.savefig(alm_file.replace('.fits', f'_{stokes}_map.pdf'))
            plt.close()
    except IndexError:
        pass


def get_noise_alms(hdu, freq, weights, lmax, paths, workspace_run):
    cached = op.join(
        f"{workspace_run}/inputs/noise/alm_{freq:0>3}_GHz.fits")
    try:
        logging.info(f"Reading {cached}")
        return hp.read_alm(cached, hdu)
    except IOError:
        logging.info(f"Could not find {cached}, computing")

    experiment = _so_or_planck_freq(freq)
    fwhm = BEAMS[FREQUENCIES.index(freq)]
    if experiment == 'so':
        noise_regex = paths['so_noise'].replace('*', 'noise', 1)
        if 'map.fits' in noise_regex:
            alm = _file2alm(weights, lmax, fwhm, noise_regex)
            alm *= (uK_AMIN_SO[FREQUENCIES_SO.index(freq)]
                    / uK_AMIN_SO[FREQUENCIES_SO.index('145')]
                    * 1e6
                    / 4
                    )
        elif 'wcov.fits' in noise_regex:
            logging.info("Simulate noise using T only cov")
            std = hp.read_map(noise_regex)**0.5
            maps = np.random.normal(size=(3,)+std.shape) * std
            maps[1:] *= 2**0.5
            alm = hp.map2alm(maps * weights, lmax=lmax)
            alm *= (uK_AMIN_SO[FREQUENCIES_SO.index(freq)]
                    / uK_AMIN_SO[FREQUENCIES_SO.index('145')]
                    * 1e6
                    / 4
                    )
            fwhm = BEAMS[FREQUENCIES.index(freq)]
            bl = hp.gauss_beam(np.radians(fwhm/60.0), lmax, pol=True).T
            for i_alm, i_bl in zip(alm, bl):
                hp.almxfl(i_alm, 1.0/i_bl, inplace=True)
        else:
            noise_file = [f for f in glob(noise_regex)
                          if f'{freq:0>3}' in f or FREQ_TAG_SO[freq] in f]
            assert len(noise_file) == 1, (
                f"{len(noise_file)} matches for {noise_regex}")
            noise_file = noise_file[0]
            alm = _file2alm(weights, lmax, fwhm, noise_file, _boost_ell_lt_30)
    else:
        noise_regex = paths['planck_noise'].replace('*', f'{freq:0>3}', 1)
        noise_file = [f for f in glob(noise_regex)]
        assert len(noise_file) == 1, (
            f"{len(noise_file)} matches for {noise_regex}")
        noise_file = noise_file[0]
        alm = _file2alm(weights, lmax, fwhm, noise_file,
                        lambda x: _planck_noise_to_C_and_uKCMB(x, freq))

    logging.info(f"Saving {cached}")
    _make_parent_dir_if_not_existing(cached)
    hp.write_alm(cached, alm)
    _save_maps_from_alm_file(cached, 512)

    return alm[np.array(hdu)-1]


def _boost_ell_lt_30(alm):
    logging.info("Boosting ell < 30")
    l, m = np.array([[l, m] for l in range(30) for m in range(l)]).T
    idx = hp.Alm.getidx(hp.Alm.getlmax(alm.shape[-1]), l, m)
    l[l == 0] = 1
    alm[..., idx] *= (3 / (1 + np.exp(l - 30)) + 1) * (l/30)**(-2)
    return alm


def get_alm(hdu, freq, weights, lmax, comps, paths, workspace_run, workspace):
    cached = op.join(
        f"{workspace_run}/inputs/{'_'.join(sorted(comps))}/alm_{freq:0>3}_GHz.fits")
    try:
        logging.info(f"Reading {cached}")
        return hp.read_alm(cached, hdu=hdu)
    except IndexError:
        return hp.read_alm(cached, 1) * np.zeros_like(hdu)[..., None]
    except IOError:
        logging.info(f"Could not find {cached}, computing")

    fwhm = BEAMS[FREQUENCIES.index(freq)]

    noise_only = ''.join(comps) == 'noise'
    if noise_only:
        alm = np.zeros((3, hp.Alm.getsize(lmax)), dtype=np.complex128)
    else:
        map_file = prepare_and_get_signal_map_file(
            freq, comps, paths, workspace, hp.get_nside(weights))
        alm = _file2alm(weights, lmax, fwhm, map_file)
        if alm.ndim == 1 or len(alm) == 1:
            alm = np.array([[1, 0, 0]]).T * alm

    if 'noise' in comps:
        try:
            alm += get_noise_alms([1, 2, 3], freq, weights, lmax,
                                  paths, workspace_run)
        except IndexError:
            logging.info(f"Polarization not found for {freq} GHz: "
                          "Setting P to 0")
            alm += get_noise_alms(1, freq, weights, lmax,
                                  paths, workspace_run)
            alm[1:] = 0

    assert alm.ndim == 2
    assert len(alm) == 3
    if not noise_only:
        logging.info(f"Saving {cached}")
        _make_parent_dir_if_not_existing(cached)
        hp.write_alm(cached, alm)
        _save_maps_from_alm_file(cached, 512)

    return alm[np.array(hdu)-1]


def get_all_alm(hdu, freqs, weights, lmax, comps, paths, workspace_run, workspace):
    shape = (len(freqs),) + np.array(hdu).shape + (hp.Alm.getsize(lmax),)
    alms = np.zeros(shape, dtype=np.complex128)
    for alm, freq in zip(alms, freqs):
        alm[:] = get_alm(hdu, freq, weights, lmax, comps,
                         paths, workspace_run, workspace)

    return alms


def set_logfile(log_file, level):
    _make_parent_dir_if_not_existing(log_file)
    print(f'setting to {level}')
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)-6s %(levelname)-6s %(message)s',
        datefmt='%H:%M:%S', handlers=[logging.FileHandler(log_file),
                                      logging.StreamHandler()])


def write(*res, outdir=None, nside=None, amin_out=2, component='cmb'):
    """scipy.optimize.res -> files
    """
    logging.info(f'Save output to {outdir}')
    alms = np.stack([r.s[0] for r in res])
    _make_parent_dir_if_not_existing(op.join(outdir, f'{component}_alms.fits'))
    logging.info('- ILC filter')
    np.savez(op.join(outdir, 'ilc_filter'), T=res[0].W, E=res[1].W, B=res[2].W)
    logging.info('- Covariance')
    np.savez(op.join(outdir, 'covariance'),
             T=res[0].cov, E=res[1].cov, B=res[2].cov)
    logging.info('- Inverse Covariance')
    np.savez(op.join(outdir, 'inv_covariance'),
             T=res[0].inv_cov, E=res[1].inv_cov, B=res[2].inv_cov)
    logging.info('- cl auto input')
    np.savez(op.join(outdir, 'cl_in'),
             T=res[0].cl_in, E=res[1].cl_in, B=res[2].cl_in, fsky=res[0].fsky)
    logging.info('- cl out')
    np.savez(op.join(outdir, f'{component}_cl_out'),
             T=res[0].cl_out, E=res[1].cl_out, B=res[2].cl_out,
             fsky=res[0].fsky)
    logging.info('- alms')
    hp.write_alm(op.join(outdir, f'{component}_alms.fits'), alms, overwrite=True)
    logging.info('- map')
    maps = hp.alm2map(alms, nside, fwhm=np.radians(amin_out/60))
    map_file = op.join(outdir, f'{component}_fwhm_{amin_out}_amin.fits')
    hp.write_map(map_file, maps, overwrite=True)
    logging.info('- plotting')
    for i, stokes in enumerate('TQU'):
        hp.mollview(maps[i])
        plt.savefig(map_file.replace('.fits', f'_{stokes}.pdf'))
        plt.close()
    for i, stokes in enumerate('EB'):
        hp.mollview(hp.alm2map(alms[i+1], nside, fwhm=np.radians(amin_out/60)))
        plt.savefig(map_file.replace('.fits', f'_{stokes}.pdf'))
        plt.close()
    ell = np.arange(res[0].cl_out.shape[-1])
    to_dl = ell * (ell + 1) / 2 / np.pi
    for i, stokes in enumerate('TEB'):
        plt.loglog(res[i].cl_out[0] * to_dl, label=stokes)
    plt.savefig(op.join(outdir, f'{component}_cl_out.pdf'))
    plt.close()
    logging.info(f'All outputs saved to {outdir}')


def main():
    # Parse logging and IO
    with open(sys.argv[1], 'r') as stream:
        params = stream.read()
    try:
        params = params.format(scan_tag=sys.argv[2])
    except IndexError:
        pass
    params = yaml.safe_load(params)

    workspace_run = f"{params['workspace']}/runs/{params['scan']['tag']}"
    log_file = f"{workspace_run}/run.log"
    _make_parent_dir_if_not_existing(log_file)
    set_logfile(log_file, params['logging_level'])
    logging.info(f"PARAMETERS READ\n\n{yaml.dump(params)}\n\n"
                 .replace('\n  -', ','))

    weights = get_weights(workspace_run, params['nside'], **params['scan'])
    fsky = np.mean(weights**2)**2 / np.mean(weights**4)
    components = []
    if params['fit_for'] == 'tsz':
        components.append(fgb.ThermalSZ())
    elif params['fit_for'] == 'cmb':
        components.append(fgb.CMB())
    else:
        raise ValueError(f"{params['fit_for']} unsupported")

    freqs = []
    if 'so' in params['inputs']['instruments']:
        freqs += FREQUENCIES_SO
    if 'planck' in params['inputs']['instruments']:
        freqs += FREQUENCIES_PLANCK

    # HILC
    res = []
    for hdu in range(1, 4):  # Avoid loading TEB at the same time
        logging.info(f'Stokes {hdu}/3')
        alms = get_all_alm(
            hdu, freqs, weights, params['lmax'], params['inputs']['components'],
            params['inputs']['paths'], workspace_run, params['workspace'])
        instrument = fgb.observation_helpers.standardize_instrument(
            {'frequency': np.array(freqs).astype(float)})
        logging.info('Do ILC')
        res.append(fgb.separation_recipes.harmonic_ilc_alm(
            components, instrument, alms, fsky=fsky, **params['hilc']))
        res[-1].freqs = freqs

    logging.info('Write output')
    odir = op.join(workspace_run, '_'.join(params['inputs']['instruments']))
    write(*res, outdir=odir, nside=params['nside'],
          amin_out=params['amin_out'], component=params['fit_for'])
    try:
        params['inputs']['components'].remove('cmb')
    except ValueError:
        pass
    else:
        for hdu, res_stokes in zip(range(1, 4), res):
            logging.info(f'Noise, Stokes {hdu}/3')
            alms = get_all_alm(
                hdu, freqs, weights, params['lmax'], params['inputs']['components'],
                params['inputs']['paths'], workspace_run, params['workspace'])
            logging.info('Apply filter to cmb-free alms')
            res_stokes.s = fgb.separation_recipes._apply_harmonic_W(
                res_stokes.W, alms)
            logging.info('Compute spectra')
            res_stokes.cl_in = np.array([hp.alm2cl(alm) for alm in alms])
            res_stokes.cl_out = np.array([hp.alm2cl(alm) for alm in res_stokes.s])
            if res_stokes.fsky is not None:
                res_stokes.cl_in /= res_stokes.fsky
                res_stokes.cl_out /= res_stokes.fsky

        logging.info('Write output')
        write(*res, outdir=op.join(odir, 'noise_foregrounds'), nside=params['nside'],
              amin_out=params['amin_out'], component=params['fit_for'])

    try:
        params['inputs']['components'].remove('noise')  # CMB already removed
    except ValueError:
        return
    else:
        for hdu, res_stokes in zip(range(1, 4), res):
            logging.info(f'Noise, Stokes {hdu}/3')
            alms = get_all_alm(
                hdu, freqs, weights, params['lmax'], params['inputs']['components'],
                params['inputs']['paths'], workspace_run, params['workspace'])
            logging.info('Apply filter noise- and CMB-free alms')
            res_stokes.s = fgb.separation_recipes._apply_harmonic_W(res_stokes.W,
                                                                    alms)
            logging.info('Compute spectra')
            res_stokes.cl_in = np.array([hp.alm2cl(alm) for alm in alms])
            res_stokes.cl_out = np.array([hp.alm2cl(alm) for alm in res_stokes.s])
            if res_stokes.fsky is not None:
                res_stokes.cl_in /= res_stokes.fsky
                res_stokes.cl_out /= res_stokes.fsky

        logging.info('Write output')
        write(*res, outdir=op.join(odir, 'foregrounds'), nside=params['nside'],
              amin_out=params['amin_out'], component=params['fit_for'])



if __name__ == '__main__':
    main()
