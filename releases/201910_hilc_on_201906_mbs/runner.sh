#!/usr/bin bash

OMP_NUM_THREADS=4 python hilc.py cmb/config.yaml
OMP_NUM_THREADS=4 python hilc.py cmb_tsz/config.yaml
OMP_NUM_THREADS=4 python hilc.py tsz/config.yaml
