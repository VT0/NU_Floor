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

#solar nu flux taken from SSM 1104.1639

def neutrino_spectrum(Emin=0.1, Emax=1000., fs=18, save=True):
    filename = test_plots + 'NeutrinoFlux.pdf'
    ylims = [10**-4., 10**13.]

    erb8 = np.logspace(np.log10(Emin), np.log10(16.36), 400)
    b8nu = np.loadtxt(path + '/Nu_Flux/B8NeutrinoFlux.dat')
    b8nu_spectrum = interp1d(b8nu[:, 0], 5.58 * 10. ** 6. * b8nu[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)

    erb7l1 = np.logspace(np.log10(0.37), np.log10(0.39), 1000)
    b7nu1 = np.loadtxt(path + '/Nu_Flux/B7NeutrinoLine1.dat')
    b7nu1_spectrum = interp1d(b7nu1[:, 0], (0.1) * 5 * 10. ** 9. * b7nu1[:, 1], kind='linear',
                             fill_value=0., bounds_error=False)

    erb7l2 = np.logspace(np.log10(0.85), np.log10(0.87), 1000)							 
    b7nu2 = np.loadtxt(path + '/Nu_Flux/B7NeutrinoLine2.dat')
    b7nu2_spectrum = interp1d(b7nu2[:, 0], (0.9) * 5 * 10. ** 9. * b7nu2[:, 1], kind='linear',
                             fill_value=0., bounds_error=False)
							 
    erpepl1 = np.logspace(np.log10(1.43), np.log10(1.45), 1000)							 
    pepnu1 = np.loadtxt(path + '/Nu_Flux/PEPNeutrinoLine1.dat')
    pepnu1_spectrum = interp1d(pepnu1[:, 0], 1.44 * 10. ** 8. * pepnu1[:, 1], kind='linear',
                             fill_value=0., bounds_error=False)

#    b7nu2 = np.loadtxt(path + '/Nu_Flux/B7NeutrinoLine2.dat')
 #   b7nu2_spectrum = interp1d(b7nu2[:, 0]* 10.**-3 + .3843,  5.00 * 10. ** 9. * b7nu2[:, 1], kind='cubic',
 #                            fill_value=0., bounds_error=False)

    erpp = np.logspace(np.log10(Emin), np.log10(0.42341), 400)
    ppnu = np.loadtxt(path + '/Nu_Flux/PPNeutrinoFlux.dat')
    ppnu_spectrum = interp1d(ppnu[:, 0], 5.98 * 10.**10. * ppnu[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)

    ero15 = np.logspace(np.log10(Emin), np.log10(1.732), 400)
    o15nu = np.loadtxt(path + '/Nu_Flux/O15NeutrinoFlux.dat')
    o15nu_spectrum = interp1d(o15nu[:, 0], 2.23 * 10.**8. * o15nu[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)
							 
    ern13 = np.logspace(np.log10(Emin), np.log10(1.199), 400)
    n13nu = np.loadtxt(path + '/Nu_Flux/N13NeutrinoFlux.dat')
    n13nu_spectrum = interp1d(n13nu[:, 0], 2.96 * 10.**8. * n13nu[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)
							 
    erf17 = np.logspace(np.log10(Emin), np.log10(1.740), 400)
    f17nu = np.loadtxt(path + '/Nu_Flux/F17NeutrinoFlux.dat')
    f17nu_spectrum = interp1d(f17nu[:, 0], 5.52 * 10.**6. * f17nu[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)

    erhep = np.logspace(np.log10(Emin), np.log10(18.784), 400)
    hepnu = np.loadtxt(path + '/Nu_Flux/HEPNeutrinoFlux.dat')
    hepnu_spectrum = interp1d(hepnu[:, 0], 8.04 * 10.**3. * hepnu[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)
							 
    eratmnue = np.logspace(np.log10(14), np.log10(944), 600)
    atmnue = np.loadtxt(path + '/Nu_Flux/atmnue_noosc_fluka_flux_norm.dat')
    atmnue_spectrum = interp1d(atmnue[:, 0], 1.27 * 10. ** 1 * atmnue[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)
							 
    eratmnuebar = np.logspace(np.log10(14), np.log10(944), 600)
    atmnuebar = np.loadtxt(path + '/Nu_Flux/atmnuebar_noosc_fluka_flux_norm.dat')
    atmnuebar_spectrum = interp1d(atmnuebar[:, 0], 1.17 * 10. ** 1 * atmnuebar[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)
	
    eratmnumu = np.logspace(np.log10(14), np.log10(944), 600)
    atmnumu = np.loadtxt(path + '/Nu_Flux/atmnumu_noosc_fluka_flux_norm.dat')
    atmnumu_spectrum = interp1d(atmnumu[:, 0], 2.46 * 10. ** 1 * atmnumu[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)
							 
    eratmnumubar = np.logspace(np.log10(14), np.log10(944), 600)
    atmnumubar = np.loadtxt(path + '/Nu_Flux/atmnumubar_noosc_fluka_flux_norm.dat')
    atmnumubar_spectrum = interp1d(atmnumubar[:, 0], 2.45 * 10. ** 1 * atmnumubar[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)
							 
    erdsnb3mev = np.logspace(np.log10(Emin), np.log10(100), 600)
    dsnb3mev = np.loadtxt(path + '/Nu_Flux/dsnb_3mev_flux_norm.dat')
    dsnb3mev_spectrum = interp1d(dsnb3mev[:, 0], 4.55 * 10. ** 1 * dsnb3mev[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)

    erdsnb5mev = np.logspace(np.log10(Emin), np.log10(100), 600)
    dsnb5mev = np.loadtxt(path + '/Nu_Flux/dsnb_5mev_flux_norm.dat')
    dsnb5mev_spectrum = interp1d(dsnb5mev[:, 0], 2.73 * 10. ** 1 * dsnb5mev[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)

    erdsnb8mev = np.logspace(np.log10(Emin), np.log10(100), 600)
    dsnb8mev = np.loadtxt(path + '/Nu_Flux/dsnb_8mev_flux_norm.dat')
    dsnb8mev_spectrum = interp1d(dsnb8mev[:, 0], 1.75 * 10. ** 1 * dsnb8mev[:, 1], kind='cubic',
                             fill_value=0., bounds_error=False)


    pl.figure()
    ax = pl.gca()

    ax.set_xlabel(r'$E_\nu$  [MeV]', fontsize=fs)
    ax.set_ylabel(r'Neutrino Flux  [$cm^{-2} s^{-1} MeV^{-1}$]', fontsize=fs)

	
    pl.plot(erb8, b8nu_spectrum(erb8), 'r', lw=1)
    pl.plot(erb7l1, b7nu1_spectrum(erb7l1), 'b', lw=2)
    pl.plot(erb7l2, b7nu2_spectrum(erb7l2),  'b', lw=2)
    pl.plot(erpepl1, pepnu1_spectrum(erpepl1),  'r', lw=2)
    pl.plot(erpp, ppnu_spectrum(erpp), 'g', lw=1)
    pl.plot(ero15, o15nu_spectrum(ero15), 'k', lw=1)
    pl.plot(ern13, n13nu_spectrum(ern13), 'k', lw=1)
    pl.plot(erf17, f17nu_spectrum(erf17), 'k', lw=1)
    pl.plot(erhep, hepnu_spectrum(erhep), 'purple', lw=1)
    pl.plot(eratmnue, atmnue_spectrum(eratmnue), 'r--', lw=1)
    pl.plot(eratmnuebar, atmnuebar_spectrum(eratmnuebar), 'r--', lw=1)
    pl.plot(eratmnumu, atmnumu_spectrum(eratmnumu), 'r--', lw=1)
    pl.plot(eratmnumubar, atmnumubar_spectrum(eratmnumubar), 'r--', lw=1)
    pl.plot(erdsnb3mev, dsnb3mev_spectrum(erdsnb3mev), 'b--', lw=1)
    pl.plot(erdsnb5mev, dsnb5mev_spectrum(erdsnb5mev), 'b--', lw=1)
    pl.plot(erdsnb8mev, dsnb8mev_spectrum(erdsnb8mev), 'b--', lw=1)

    plt.tight_layout()

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
	
    hep_nu_max = 18.77
    hep_flux = Nu_spec()
    erhep = np.logspace(np.log10(Emin), np.log10(hep_flux.max_er_from_nu(hep_nu_max, np.min(experiment_info[:,0]))),
                       200)
    nuhepspec = np.zeros_like(erhep)
	
    for iso in experiment_info:
        nub8spec += b8_flux.nu_rate('B8', erb8, iso)
        nuhepspec += b8_flux.nu_rate('B8', erb8, iso)
        nuhepspec += hep_flux.nu_rate('hep', erhep, iso)
    
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
    pl.plot(erhep, nuhepspec, 'r', lw=1)
    pl.plot(er_list, dm_spec, 'b', lw=1)
	
    plt.tight_layout()
    
    plt.xlim(xmin=Emin, xmax=Emax)
    plt.ylim(ymin=10.**-5., ymax=10.**8.)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if save:
        plt.savefig(filename)
    return
