Harmonic ILC on MBS Oct 2019
-----------------------------

## Configuration

See the `yaml` files and the following sections.

### Harmonic ILC

The component separation method is the (simplest possible) internal linear
combination (ILC) applied in harmonic domain. More in detail, the recipe is the
following.

1. Multiply the input frequency-maps by the weights map (see next section)
2. Compute the pseudo-SH coefficients up to `lmax`
3. Bin the multipoles in `delta_ell`-wide bins
4. For each bin, compute the frequency-frequency empirical covariance matrix
5. Use the (known) SEDs of the desired output components and the empirical
   covariance matrices to build a bin-dependent un-mixing matrix
6. Apply the un-mixing matrices to the pseudo-SH coefficients to get the
   pseudo-SH of the output components

### Wights Map

We use uniform weights. We build a binary mask based on a galactic mask
(`gal_mask`) and a hits map (`hits_map`,  masking pixels under
`rel_threshold` times the maximum). A further latitude cut is applied.
The binary mask is then apodized with a `apo_deg`-FWHM gaussian kernel 

### Inputs

LAT map-based simulations

Galactic (dust, synchrotron, AME, free-free): `/project/projectdirs/sobs/v4_sims/mbs/201904_highres_foregrounds_equatorial/4096/*/0000/simonsobs_*_uKCMB_la*.fits`
Extra-galactic (cmb, tsz, ksz, cib): `/project/projectdirs/sobs/v4_sims/mbs/201905_extragalactic/4096/*/0000/simonsobs_*_uKCMB_la*.fits`
Noise: `/global/cscratch1/sd/zonca/simonsobs/mapsims_runs/201906_noise_no_lowell/4096/noise/0000/simonsobs_*_uKCMB_la*.fits`

### Outputs

- CMB
- CMB and tSZ

Note that if multiple output components are requested their are orthogonalized
with respect to each other.

Output are available at

`/global/cscratch1/sd/dpoletti/analyses_home/so_analysis/fg_awg/draft_201909_hilc_on_201906_mbs`

at NERSC both as SH coefficients and as
maps convolved with the LAT smallest beam: 0.9 arcmin.

As we agree on the correctness and the format of the output, we should probably
move it to the project dir.

## Running codes on your own

### Reproduce the analysis

Choose the `config.yaml` file and run the main `hilc.py` on it. In order to run the SHT on NSIDE = 4096 maps, make sure you are exploiting (in-node) parallelism as much as you can. The typical call on Cori is 

```
OMP_NUM_THREADS=64 python hilc.py cmb/config.yaml
```

## Run on your own simulations

The simplest approach is to write your own `config.yaml` file.
If you want to embed the separation in some other code, have a look at
the source of `hilc.py`, it is fairly modular and documented. Note that the core
ILC routines are in `fgbuster.separation_recipies`.

## Get in touch

For any bug report, feature request, further explanation, please write to the `#so_fg`
Slack channel or open an issue.

