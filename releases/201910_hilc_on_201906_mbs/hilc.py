""" HILC on MBS
"""
# pylint: disable=W1203
import sys
import logging
import os
import os.path as op
from glob import glob
import yaml
import numpy as np
import xarray as xr
import healpy as hp
import fgbuster as fgb
import matplotlib.pyplot as plt

#FIXME: load these data from mapsims
FREQUENCIES = np.array([27., 39., 93., 145., 225., 280.])
BEAMS = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])

NSIDE = 4096
THRESHOLD_WEIGHTS = 1e-4

TEB = 'T E B'.split()
SPEC = 'TT EE BB TE TB EB'.split()

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

def get_input_alms(inputs=None, weights=None, lmax=None, workspace=None,
                   cache=False):
    """ Alms of the input components

    Parameters
    ----------
    inputs: dict
        Dictionary of type {comp_name: regex_of_component_maps}
    weights: array
        Weights map to be applied to the maps before alm computation
    lmax: int
        Compute alms up to ell = lmax
    workspace: string
        Path to be used as a workspace. Fits files containing the alms will be
        saved in workspace
    cache: bool
        If True, keep the alms of each component in memory.
    """
    tot_alms = None
    def add_alms(alms_to_add):
        nonlocal tot_alms
        if tot_alms is None:
            tot_alms = alms_to_add.copy()
        else:
            tot_alms += alms_to_add

    for comp in inputs:
        if not hasattr(get_input_alms, comp):
            comp_map_regex = inputs[comp]

        if workspace:
            alms_files = [
                op.join(workspace, op.basename(m).replace('.fits', '.npy'))
                for m in sorted(glob(comp_map_regex))]

        if cache and hasattr(get_input_alms, comp):
            # alms were cached
            logging.info(f"Using cached {comp}")
            add_alms(getattr(get_input_alms, comp))
        elif workspace and all([op.exists(f) for f in alms_files]):
            # Precomputed alms are on disk
            logging.info(f"Loading {comp} stored in {workspace}")
            alms = np.load(alms_files[0])
            alms = np.pad(alms[np.newaxis],
                          [(0, len(alms_files)-1), (0, 0), (0, 0)],
                          mode='constant')
            for alm, f in zip(alms[1:], alms_files[1:]):
                alm[:] = np.load(f)

            if cache:
                assert not hasattr(get_input_alms, comp)
                setattr(get_input_alms, comp, alms)
            add_alms(alms)
        else:
            # Compute the alms
            logging.info(f"Computing the alms for {comp}")
            try:
                maps = np.array([hp.read_map(m, field=(0,1,2))
                                 for m in sorted(glob(comp_map_regex))])
            except IndexError:
                maps = np.array([hp.read_map(m)
                                 for m in sorted(glob(comp_map_regex))])
                maps = np.pad(maps[:, np.newaxis],
                              [(0, 0), (0, 2), (0, 0)], mode='constant')

            alms = fgb.separation_recipies._get_alms(maps, BEAMS, lmax, weights)
            if cache:
                assert not hasattr(get_input_alms, comp)
                setattr(get_input_alms, comp, alms)
            if workspace:
                logging.info(f"Saving the alms for {comp} to {workspace}")
                for f, alm in zip(alms_files, alms):
                    logging.debug(f"\t{f}")
                    np.save(f, alm)

                plt.figure(figsize=(12, 8))
                fsky = np.mean(weights**2)**2 / np.mean(weights**4)
                for lw, alm in zip(np.linspace(1, 3, len(alms)), alms):
                    cls = hp.alm2cl(alm)
                    ell = np.arange(cls.shape[-1])
                    for c, cl in zip('brg', cls):
                        plt.loglog(cl * ell * (ell + 1) / 2 / np.pi / fsky,
                                   color=c, lw=lw)
                plt.ylim(1e-2, 1e5)
                plt.title(comp)
                plt.tight_layout()
                plt.savefig(op.join(workspace, comp+'.pdf'))
                plt.close()

            add_alms(alms)

    return tot_alms


def _get_components(comp_names):
    components = []
    for comp_name in comp_names:
        if comp_name == 'cmb':
            components.append(fgb.CMB())
        elif comp_name == 'tsz':
            components.append(fgb.ThermalSZ())
        else:
            raise ValueError("Unsupported component {comp_name}")

    return components



def hilc(inputs=None, outputs=None, weights=None,
         lmax=None, delta_ell=None, workspace=None, outdir=None):
    """ Harmonic ILC

    Parameters
    ----------
    inputs: dict
        Dictionary of type {comp_name: regex_of_component_maps}
    outputs: list
        Desired output components (among cmb and tsz)
    weights: dict
        Arguments for the creation of the weights. See `get_weights`
    lmax: int
        Compute alms up to ``ell = lmax``
    delta_ell: int
        Size of the ILC bins
    workspace: string
        Path to be used as a workspace. Fits files containing the alms will be
        saved in workspace
    outdir: string
        Path to store the output maps in standard healpix format

    Note
    ----
    All the output spectra adopt the simple fsky-corretion.
    """
    # Preparing inputs
    logging.info("Preparing for ILC")
    weights_map = get_weights(workspace=workspace, **weights)
    fsky = np.mean(weights_map**2)**2 / np.mean(weights_map**4)
    alms = get_input_alms(
        inputs=inputs, lmax=lmax, weights=weights_map, cache=True,
        workspace=op.join(workspace, weights['tag']))
    components = _get_components(outputs)
    instrument = dict(Frequencies=FREQUENCIES, Beams=BEAMS)
    lbins = np.arange(0, lmax+delta_ell, delta_ell)

    # ILC
    logging.info("Starting ILC")
    res = fgb.separation_recipies._harmonic_ilc_alm(
        components, instrument, alms, lbins, fsky)

    # Output
    logging.info("Output")
    res.s_alms = res.s
    res.s_alms = xr.DataArray(res.s_alms,
                              dims='output TEB lm'.split(),
                              coords=dict(output=outputs, TEB=TEB))
    res.weights = weights_map
    logging.info("Alm2map")
    res.s = np.array([hp.alm2map(alm, NSIDE, fwhm=np.radians(BEAMS[-1]/60.))
                      for alm in res.s])
    res.s = xr.DataArray(res.s,
                         dims='output Stokes pixel'.split(),
                         coords=dict(output=outputs, Stokes='I Q U'.split()))
    logging.info("Correlation with the input")
    corr_with = [[i] for i in inputs]
    if 'cmb' in inputs and 'ksz' in inputs:
        corr_with.append(['cmb', 'ksz'])

    res.io_corr = _correlate_with_inputs(res.s_alms, corr_with, res.W, lbins)
    res.io_corr /= fsky
    res.cl_in = xr.DataArray(res.cl_in,
                             dims='freq spec ell'.split(),
                             coords=dict(freq=FREQUENCIES, spec=SPEC))
    res.cl_out = xr.DataArray(res.cl_out,
                              dims='comp spec ell'.split(),
                              coords=dict(comp=outputs, spec=SPEC))
    _to_disk(res, outdir)


def _plot_corr(corr, outdir):
    ell = np.arange(corr.shape[-1])
    for o_comp in corr.output.values:
        for s in 'TEB':
            plt.figure(figsize=(12, 8))
            for i_comp in corr.input.values:
                cl = corr.sel(output=o_comp, input=i_comp, iTEB=s, oTEB=s)
                plt.loglog(cl.values* ell * (ell + 1) / 2 / np.pi, label=i_comp)
            plt.legend()
            plt.savefig(op.join(outdir, 'extras', f'{o_comp}_{s}{s}.pdf'))
            plt.close()


def _to_disk(res, outdir):
    # Output maps and alms
    for comp in res.s.output.data:
        if comp == 'cmb':
            unit = 'uKCMB'
        elif comp == 'tsz':
            unit = 'y'
        else:
            unit = ''

        filename = op.join(
            outdir, 'maps', f'simonsobs_{comp}_{unit}_la_nside{NSIDE}.fits')
        _make_parent_dir_if_not_existing(filename)
        hp.write_map(filename, res.s.sel(output=comp).data, overwrite=True)

        filename = op.join(
            outdir, 'alms', f'simonsobs_{comp}_{unit}_la_nside{NSIDE}.npy')
        _make_parent_dir_if_not_existing(filename)
        np.save(filename, res.s_alms.sel(output=comp).data)

    # Extras
    filename = op.join(outdir, 'extras', 'weights.fits')
    _make_parent_dir_if_not_existing(filename)
    hp.write_map(filename, res.weights, overwrite=True)

    filename = op.join(outdir, 'extras', 'cross_w_input.hdf5')
    res.io_corr.to_netcdf(filename)
    _plot_corr(res.io_corr, outdir)

    for io in 'in out'.split():
        filename = op.join(outdir, 'extras', f'cl_{io}.hdf5')
        res[f'cl_{io}'].to_netcdf(filename)

    filename = op.join(outdir, 'extras', 'res.npy')


def _correlate_with_inputs(s_alms, inputs, W, lbins):
    lmax = hp.Alm.getlmax(s_alms.shape[-1])
    ells = hp.Alm.getlm(lmax)[0]
    bin_ells = np.digitize(ells, lbins)
    W = W[:, bin_ells]  # (teb, lm, comp, freq)
    # Alms of all inputs, shape is (i_comp, o_comp, teb, lm)
    all_alms = np.empty((len(inputs),) + s_alms.shape, dtype=s_alms.dtype)
    for i_input, comp_input in enumerate(inputs):
        in_alm = get_input_alms(inputs=comp_input, cache=True)
        np.einsum('slof,fsl->osl', W, in_alm, out=all_alms[i_input])
    all_alms = np.swapaxes(all_alms, 0, 1)  # (o_comp, i_comp, teb, lm)

    # Correlation, shape is (o_comp, teb, i_comp, teb, l)
    corr = np.empty(s_alms.shape[:2] + all_alms.shape[1:3] + (lmax+1,))
    # For loop necessary to avoid excessive memory consumption
    # Note that pushing the iteration over TEB inside alm2cm would result in
    # losing the ET BT and BE spectra
    for o_s_alms, o_all_alms, o_corr in zip(s_alms.data, all_alms, corr):
        for o_teb_s_alms, o_teb_corr in zip(o_s_alms, o_corr):
            for o_i_all_alms, o_teb_i_corr in zip(o_all_alms, o_teb_corr):
                for o_i_teb_all_alms, o_teb_i_teb_corr in zip(o_i_all_alms,
                                                              o_teb_i_corr):
                    o_teb_i_teb_corr[:] = hp.alm2cl(
                        o_i_teb_all_alms, o_teb_s_alms, lmax=lmax)

    input_coords = [' '.join(i) if isinstance(i, (tuple, list)) else i
                    for i in inputs]
    return xr.DataArray(corr,
                        dims='output oTEB input iTEB ell'.split(),
                        coords=dict(output=s_alms.output, oTEB=TEB,
                                    input=input_coords, iTEB=TEB))

def _main():
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)
    log_file = op.join(params['outdir'], 'run.log')
    _make_parent_dir_if_not_existing(log_file, verbose=False)
    logging.basicConfig(
        level=params['logging_level'],
        format='%(asctime)-6s %(levelname)-6s %(message)s',
        datefmt='%H:%M:%S', handlers=[logging.FileHandler(log_file),
                                      logging.StreamHandler()])
    del params['logging_level']
    logging.info(f"PARAMETERS READ\n\n{yaml.dump(params)}\n\n")
    hilc(**params)


if __name__ == '__main__':
    _main()
