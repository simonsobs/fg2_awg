#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def plot_cmb():
    d = np.load('cmb.npz')
    dl = d['ells']**2 / 2 / np.pi
    plt.plot(d['ells'], d['TT']*dl)
    plt.plot(d['ells'], d['EE']*dl)
    plt.plot(d['ells'], d['BB']*dl)
    plt.show()


def plot_cov():
    d = np.load('fg.npz')
    dl = d['ells']**2 / 2 / np.pi

    plt.figure()
    freqs = d['freq_TT']
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", plt.cm.viridis(np.linspace(0, 1, len(freqs))))

    for i in range(len(freqs)):
        plt.loglog(d['ells'], d['TT'][i, -1]*dl, label=str(freqs[i]))
    plt.legend()
    plt.xlim(None, 2000)
    plt.figure()
    freqs = d['freq_EE']
    for i in range(len(freqs)):
        plt.loglog(d['ells'], d['EE'][i, -1]*dl, label=str(freqs[i]))
    plt.xlim(None, 2000)
    plt.legend()
    plt.figure()
    freqs = d['freq_BB']
    for i in range(len(freqs)):
        plt.loglog(d['ells'], d['BB'][i, -1]*dl, label=str(freqs[i]))
    plt.xlim(None, 2000)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_cov()
