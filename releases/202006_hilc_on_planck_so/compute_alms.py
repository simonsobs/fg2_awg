""" HILC on MBS
"""
# pylint: disable=W1203
import sys
import logging
import os
import os.path as op
from glob import glob
import h5py as h5
import matplotlib.pyplot as plt
import yaml
import numpy as np
import healpy as hp
import  pysm3.units as u

#FIXME: load these data from mapsims
FREQUENCIES_SO = np.array([27., 39., 93., 145., 225., 280.])
BEAMS_SO = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])

FREQUENCIES_PLANCK = np.array([30., 44, 70, 100, 143, 217, 353, 545, 857])
BEAMS_PLANCK = np.array([33.102652125, 27.94348615, 13.07645961, 9.682, 7.303, 5.021, 4.944, 4.831, 4.638])

FREQUENCIES = np.concatenate([FREQUENCIES_SO, FREQUENCIES_PLANCK])
BEAMS = np.concatenate([BEAMS_SO, BEAMS_PLANCK])

NSIDE = 4096
THRESHOLD_WEIGHTS = 1e-4

#EXTRAGALACTIC = 'cmb tsz ksz cib'.split()
#GALACTIC = 'dust synchrotron freefree ame'.split()

def _make_parent_dir_if_not_existing(filename, verbose=True):
    dirname = op.dirname(filename)
    if not op.exists(dirname):
        if verbose:
            logging.debug(f"Creating the folder {dirname} for {filename}")
        os.makedirs(dirname)


def get_weights(hits_map=None, rel_threshold=0, gal_mask=None,
                lat_lim=None, apo_deg=5, tag=None, workspace=None):
    """ Weights map

    They are defined by the hits map and the galactic mask.
    First it mask out the galaxy and pixles with an insufficient number of hits.
    Second, it applies a latitude gut.
    Third, it apodize the unmasked region.

    Parameters
    ----------
    hits_map: string
        fits file of the hits map
    rel_threshold: float
        mask pixels with a numer of hits lower than the maximum number of hits
        multiplied by rel_threshold
    gal_map: string
        fits file of the galactic map
    lat_lim: touple
        Minimum and maximum latitude (before smoothing)
    apo_deg: float
        Apodization lenght
    tag: string
        Label of this configuration for the weights calculation
    workspace: string
        Path to be used as a workspace. The weights map is saved in
        workspace/tag/weights
    """
    if not (hits_map and gal_mask):
        return np.ones(hp.nside2npix(NSIDE))

    if bool(workspace) != bool(tag):
        logging.warning(f"Not loading/storing the weights:"
                        " both tag and workspace have to be set"
                        " but tag is {tag} and workspace is {workspace}")

    if tag and workspace:
        weights_file = op.join(workspace, tag, 'weights.fits')
        if op.exists(weights_file):
            logging.info(f"Loading weights {weights_file}")
            return hp.read_map(weights_file)

    logging.info("Computing weights")

    mask_gal = hp.read_map(gal_mask) > 0.
    #mask_gal = hp.ud_grade(hp.read_map(gal_mask), NSIDE) > 0.
    hits = hp.read_map(hits_map)
    if hp.get_nside(hits) != NSIDE:
        # upgrading with map2alm + alm2map is smoother than ud_grade
        hits = hp.map2alm(hits, iter=5)
        hits = hp.alm2map(hits, NSIDE)
    mask_hits = hits > rel_threshold * hits.max()

    mask = mask_gal * mask_hits
    if lat_lim:
        lat = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)[1]
        mask *= (lat > lat_lim[0]) * (lat < lat_lim[1])

    weights = hp.smoothing(mask.astype(float), np.radians(apo_deg))
    weights *= weights > THRESHOLD_WEIGHTS

    if tag and workspace:
        logging.info(f"Saving weights {weights_file}")
        _make_parent_dir_if_not_existing(weights_file)
        hp.write_map(weights_file, weights)
        hp.mollview(weights, title='Weights')
        plt.savefig(weights_file.replace('.fits', '.pdf'))
        plt.close()

    return weights


def get_frequency_files(comp, paths):
    '''
    if comp in GALACTIC:
        so = sorted(glob(paths['so_gal'].replace('*', comp, 1)))
    elif comp in EXTRAGALACTIC:
        so = sorted(glob(paths['so_extra'].replace('*', comp, 1)))
    elif comp == 'noise':
    '''
    if comp == 'noise':
        so = sorted(glob(paths['so_noise'].replace('*', comp, 1)))
    else:
        so = sorted(glob(paths['so'].replace('*', comp, 1)))
        #raise ValueError(comp)

    if comp == 'noise':
        planck = sorted(glob(paths['planck_noise']))
    else:
        planck = sorted(glob(paths['planck'].replace('*', comp, 1)))
    return so + planck


def _planck_to_uKCMB(alm, freq, comp, pass_not_planck=True):
    if comp != 'noise':
        return
    freq = int(freq)
    if freq in [545, 857]:
        logging.info("MJy/sr -> uKCMB")
        alm *= (1 * u.MJy / u.sr).to(
                    u.uK_CMB, equivalencies=u.cmb_equivalencies(freq * u.GHz)
                    ).value
        hp.Rotator(coord='gc').rotate_alm(alm, inplace=True)
    elif freq in [30, 44, 70, 100, 143, 217, 353]:
        logging.info("KCMB -> uKCMB")
        alm *= 1e6
        hp.Rotator(coord='gc').rotate_alm(alm, inplace=True)
    elif pass_not_planck:
        pass
    else:
        raise ValueError(
            f"{freq} not in Planck frequencies: {FREQUENCIES_PLANCK}")


def _boost_ell_lt_30_so(alm, freq, comp, pass_not_so=True):
    if comp != 'noise':
        return
    freq = int(freq)
    if freq in list(FREQUENCIES_SO.astype(int)):
        logging.info(f"Boosting ell < 30 for freq {freq}")
        l, m = np.array([[l, m] for l in range(30) for m in range(l)]).T
        idx = hp.Alm.getidx(hp.Alm.getlmax(alm.shape[-1]), l, m)
        l[l==0] = 1
        alm[..., idx] *= (3 / (1 + np.exp(l - 30)) + 1) * (l/30)**(-2)
    elif pass_not_so:
        pass
    else:
        raise ValueError(
            f"{freq} not in SO frequencies: {FREQUENCIES_SO}")


def compute_file(freq, beam, map_file, comp, outfile, lmax, weights):
    freq = str(int(freq))
    try:
        with h5.File(outfile, 'r') as f:
            f[f'{comp}/{freq}']
        logging.info(f"Alms already computed for {map_file}")
        return
    except (KeyError, OSError):
        pass

    logging.info(f"Loading {map_file}")
    try:
        maps = hp.read_map(map_file, field=(0, 1, 2))
    except IndexError:
        maps = hp.read_map(map_file)
    nside_map = hp.get_nside(maps)
    nside_weights = hp.get_nside(weights)
    if nside_map != nside_weights:
        logging.warning(
            f"Maps have nside {nside_map} but weights have nside "
            f"{nside_weights}. ud_grading the latter to the former")
        weights = hp.ud_grade(weights, hp.get_nside(maps))
        logging.info("Weights ud_graded")
    
    if lmax > 3 * nside_map - 1:
        logging.warning(
            f"Lmax is {lmax} but the nside is only {nside_map}. Computing alms "
            f"up to {3 * nside_map-1} and filling the rest with white noise")
        logging.info(f"Computing Alms")
        alm = hp.map2alm(hp.ma(maps)*weights, lmax=3*nside_map-1)
        pad_width = [(0, lmax - 3 * nside_map + 1)]
        if alm.ndim == 2:
            pad_width = [(0, 0)] + pad_width
        logging.info("Filling with white noise")
        cl = np.pad(hp.alm2cl(alm), pad_width, mode='edge')
        if maps.ndim == 2:  # Set TE, TB and EB to zero in the added alm
            cl[3:, 3*nside_map:] = 0
        full_alm = hp.synalm(cl, new=True)
        full_idx = hp.Alm.getidx(
            lmax, *hp.Alm.getlm(3*nside_map-1,
                                np.arange(hp.Alm.getsize(3*nside_map-1)))
            )
        if maps.ndim == 2:
            full_alm[:, full_idx] = alm
        else:
            full_alm[full_idx] = alm
        alm = full_alm
    else:
        logging.info(f"Computing Alms")
        alm = hp.map2alm(hp.ma(maps)*weights, lmax=lmax)
    _planck_to_uKCMB(alm, freq, comp)
    _boost_ell_lt_30_so(alm, freq, comp)
    bl = hp.gauss_beam(np.radians(beam/60.0), lmax, pol=(alm.ndim==2))
    if alm.ndim == 1:
        alm = [alm]
        bl = [bl]
    else:
        bl = bl.T

    for i_alm, i_bl in zip(alm, bl):
        hp.almxfl(i_alm, 1.0/i_bl, inplace=True)
    logging.info(f"Saving the alms {comp} {freq} GHz")
    with h5.File(outfile, 'a') as f:
        f[f'{comp}/{freq}'] = alm


def check_files(inputs=None):
    for comp in inputs['components']:
        for f in  get_frequency_files(comp, inputs['paths']):
            print(comp, f)

def set_logfile(log_file, level=None):
    _make_parent_dir_if_not_existing(log_file, verbose=False)
    logging.basicConfig(
        level=(logging.root.level if level is None else level),
        format='%(asctime)-6s %(levelname)-6s %(message)s',
        datefmt='%H:%M:%S', handlers=[logging.FileHandler(log_file),
                                      logging.StreamHandler()])

def _main():
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)

    set_logfile(op.join(params['outdir'], f'run.log'), params['logging_level'])
    del params['logging_level']
    logging.info(f"PARAMETERS READ\n\n{yaml.dump(params)}\n\n")

    weights = get_weights(workspace=params['outdir'], **params['weights'])
    outfile = op.join(params['outdir'], f'alms.h5')
    for comp in params['inputs']['components']:
        comp_files = get_frequency_files(comp, params['inputs']['paths'])
        for freq, beam, map_file in zip(FREQUENCIES, BEAMS, comp_files):
            compute_file(freq, beam, map_file, comp, outfile,
                         params['lmax'], weights)


if __name__ == '__main__':
    _main()
