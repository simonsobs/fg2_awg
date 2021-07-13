# Harmonic ILC on SO and Planck

### Input
* SO noise: [`201906_noise_no_lowell`](https://github.com/simonsobs/map_based_simulations/tree/master/201906_noise_no_lowell) or [`scanning strategy simulations`](http://simonsobservatory.wikidot.com/pwg:time-domain-sims-log:scan-s0001) (not yet)
* SO components: [`201905_extragalactic`](https://github.com/simonsobs/map_based_simulations/tree/master/201905_extragalactic) [`201904_highres_foregrounds_equatorial`](https://github.com/simonsobs/map_based_simulations/tree/master/201904_highres_foregrounds_equatorial)

but also Planck can be included easily

* Planck components: [`201909_highres_foregrounds_extragalactic_planck_deltabandpass`](https://github.com/simonsobs/map_based_simulations/tree/master/201909_highres_foregrounds_extragalactic_planck_deltabandpass)
* Planck noise: FFP10

The scanning strategy being used can be configured.

### Output
* CMB for different configurations
  - SO
  - SO + Planck (not on disk)
  - Planck (not on disk)
* y map (not on disk)
* alms and maps of input and output stored

See the `probe_scan_strategy.py` header for the output structure.


### Method
Harmonic ILC. Bins width and lmax is tunable.

### Use the cleaned CMB
It is available at NERSC both as a map (convolved with a 2 arcmin FWHM beam)

```
/global/cscratch1/sd/dpoletti/analyses_home/fg2_awg/analyses/202106_hilc_scan_strat/strategies/runs/*/so/cmb_fwhm_2_amin.fits
```

and as alms

```
/global/cscratch1/sd/dpoletti/analyses_home/fg2_awg/analyses/202106_hilc_scan_strat/strategies/runs/*/so/cmb_alms.fits
```

### Use the codes
Launch with
```
python probe_scan_strategy.py config_hilc.yaml
```
or look at `launcher_scan_strat.sl`

If you just want to do component separation of your own alms or want to
integrate HILC in your pipeline, call directly
[`harmonic_ilc_alm`](https://fgbuster.github.io/fgbuster/api/fgbuster.separation_recipes.html#fgbuster.separation_recipes.harmonic_ilc_alm)

A slightly different run, or the products not on disk can be produced easily.

As a general remark, don't hesitate to get in touch either via an issue or
slack.
