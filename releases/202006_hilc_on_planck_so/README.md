# Harmonic ILC on SO and Planck

### Input
* SO noise: [`201906_noise_no_lowell`](https://github.com/simonsobs/map_based_simulations/tree/master/201906_noise_no_lowell)
* SO components: [`201906_highres_foregrounds_extragalactic_tophat`](https://github.com/simonsobs/map_based_simulations/tree/master/201906_highres_foregrounds_extragalactic_tophat)
* Planck components: [`201909_highres_foregrounds_extragalactic_planck_deltabandpass`](https://github.com/simonsobs/map_based_simulations/tree/master/201909_highres_foregrounds_extragalactic_planck_deltabandpass)
* Planck noise: FFP10

### Output
* CMB for different configurations
  - SO + Planck
  - SO
  - Planck

### Method
Harmonic ILC with bins 30-multipole wide up to 1000

### Use the cleaned CMB
It is available at NERSC both as a map (convolved with a 2 arcmin FWHM beam)

```
/global/cscratch1/sd/dpoletti/analyses_home/fg2_awg/releases/202006_hilc_on_planck_so/hilc_out/so_planck__cmb_tsz_ksz_cib_dust_synchrotron_freefree_ame_noise/cmb_fwhm_2_amin.fits
```

and as alms

```
/global/cscratch1/sd/dpoletti/analyses_home/fg2_awg/releases/202006_hilc_on_planck_so/hilc_out/so_planck__cmb_tsz_ksz_cib_dust_synchrotron_freefree_ame_noise/cmb_alms.fits
```

The SO-only and Planck-only maps are available in the
neighboring folders.

### Use the codes
`compute_alms.py` computes the alms of all the components at all frequencies.
`hilc.py` selectively loads them, depending on the configuration, and runs HILC. 

If you just want to do component separation of your own alms or want to
integrate HILC in your pipeline, call directly
[`harmonic_ilc_alm`](https://fgbuster.github.io/fgbuster/api/fgbuster.separation_recipes.html#fgbuster.separation_recipes.harmonic_ilc_alm)

As a general remark, don't hesitate to get in touch either via an issue or slack
