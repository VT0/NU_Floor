import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from experiments import *
from likelihood import *
from helpers import *
from rate_UV import *

import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
import glob
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)

mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=18
mpl.rcParams['ytick.labelsize']=18


path = os.getcwd()
test_plots = os.getcwd() + '/Test_Plots/'

def neutrino_specturm(Emin=0.1, Emax=1000., fs=18, save=True):
    filename = test_plots + 'NeutrinoFlux.pdf'
    ylims = [10**-4., 10**13.]

    erb8 = np.logspace(np.log10(Emin), np.log10(16.36), 400)
    b8nu = np.loadtxt(path + '/Nu_Flux/B8NeutrinoFlux.dat')
    b8nu_spectrum = interp1d(b8nu[:, 0], 5.69 * 10. ** 6. * b8nu[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)

    erb7 = np.logspace(np.log10(Emin), np.log10(1.), 400)
    b7nu1 = np.loadtxt(path + '/Nu_Flux/B7NeutrinoLine1.dat')
    b7nu1_spectrum = interp1d(b7nu1[:, 0] * 10.**-3 + 0.8613, 4.84 * 10. ** 9. * b7nu1[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)

    b7nu2 = np.loadtxt(path + '/Nu_Flux/B7NeutrinoLine2.dat')
    b7nu2_spectrum = interp1d(b7nu2[:, 0]* 10.**-3 + .3843,  4.84 * 10. ** 9. * b7nu2[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)

    erpp = np.logspace(np.log10(Emin), np.log10(0.42341), 400)
    ppnu = np.loadtxt(path + '/Nu_Flux/PPNeutrinoFlux.dat')
    ppnu_spectrum = interp1d(ppnu[:, 0], 5.99 * 10.**10. * ppnu[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)

    er015 = np.logspace(np.log10(Emin), np.log10(1.732), 400)
    o15nu = np.loadtxt(path + '/Nu_Flux/O15NeutrinoFlux.dat')
    o15nu_spectrum = interp1d(o15nu[:, 0], 2.33 * 10.**8. * o15nu[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)

    erhep = np.logspace(np.log10(Emin), np.log10(18.784), 400)
    hepnu = np.loadtxt(path + '/Nu_Flux/HEPNeutrinoFlux.dat')
    hepnu_spectrum = interp1d(hepnu[:, 0], 7.93 * 10.**3. * hepnu[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)

    pl.figure()
    ax = pl.gca()

    ax.set_xlabel(r'$E_\nu$  [MeV]', fontsize=fs)
    ax.set_ylabel(r'Neutrino Flux  [$cm^{-2} s^{-1} MeV^{-1}$]', fontsize=fs)

    pl.plot(erb8, b8nu_spectrum(erb8), 'r', lw=1)
    pl.plot(erb7, b7nu1_spectrum(erb7), 'b', lw=1)
    pl.plot(erb7, b7nu2_spectrum(erb7),  'b', lw=1)
    pl.plot(erpp, ppnu_spectrum(erpp), 'g', lw=1)
    pl.plot(er015, o15nu_spectrum(er015), 'k', lw=1)
    pl.plot(erhep, hepnu_spectrum(erhep), 'purple', lw=1)


    plt.xlim(xmin=Emin, xmax=Emax)
    plt.ylim(ymin=ylims[0],ymax=ylims[1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    if save:
        plt.savefig(filename)
    return


def neutrino_recoils(Emin=0.001, Emax=100., element='germanium', fs=18, save=True,
                     mass=6., sigmap=4.*10**-45., model='sigma_si', fnfp=1.,
                     delta=0., GF=False, time_info=False):
    filename = test_plots + 'NeutrinoRecoils_in_' + element + '.pdf'

    experiment_info, Qmin, Qmax = Element_Info(element)

    b8_nu_max = 16.18
    b8_flux = Nu_spec()
    erb8 = np.logspace(np.log10(Emin), np.log10(b8_flux.max_er_from_nu(b8_nu_max, np.min(experiment_info[:,0]))),
                       200)
    nub8spec = np.zeros_like(erb8)
    for iso in experiment_info:
        nub8spec += b8_flux.nu_rate('B8', erb8, iso)
    
    coupling = "fnfp" + model[5:]

    drdq_params = default_rate_parameters.copy()
    drdq_params['element'] = element
    drdq_params['mass'] = mass
    drdq_params[model] = sigmap
    drdq_params[coupling] = fnfp
    drdq_params['delta'] = delta
    drdq_params['GF'] = GF
    drdq_params['time_info'] = time_info

    er_list = np.logspace(np.log10(Emin), np.log10(Emax), 200)
    time_list = np.zeros_like(er_list)
    dm_spec = dRdQ(er_list, time_list, **drdq_params) * 10. ** 3. * s_to_yr

    pl.figure()
    ax = pl.gca()

    ax.set_xlabel(r'Recoil Energy  [keV]', fontsize=fs)
    ax.set_ylabel(r'Event Rate  [${\rm ton}^{-1} {\rm yr}^{-1} {\rm keV}^{-1}$]', fontsize=fs)

    pl.plot(erb8, nub8spec, 'r', lw=1)
    pl.plot(er_list, dm_spec, 'b', lw=1)
    
    plt.xlim(xmin=Emin, xmax=Emax)
    plt.ylim(ymin=10.**-5., ymax=10.**8.)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if save:
        plt.savefig(filename)
    return
