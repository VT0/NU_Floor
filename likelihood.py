"""

Code info:

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from rate_UV import *
from helpers import *
import os

path = os.getcwd()

b8nu = np.loadtxt(path + '/Nu_Flux/B8NeutrinoFlux.dat')
b8nu_spectrum = interp1d(b8nu[:,0], b8nu[:,1], kind='cubic', fill_value=0., bounds_error=False)

gF = 1.16637 * 10. ** -5. # Gfermi in GeV^-2
sw = 0.2312 # sin(theat_weak)^2
MeVtofm = 0.0050677312
s_to_yr = 3.154*10.**7.


class Likelihood_analysis(object):

    def __init__(self, model, coupling, mass, dm_sigp, fnfp, exposure, element, isotopes,
                 energies, times, Qmin, Qmax, time_info=False, GF=False):
        
        self.events = energies
        self.element = element
        self.mass = mass
        self.dm_sigp = dm_sigp
        self.exposure = exposure
        
        self.Qmin = Qmin
        self.Qmax = Qmax

        self.times = times

        like_params =default_rate_parameters.copy()

        like_params['element'] = element
        like_params['mass'] = mass
        like_params[model] = dm_sigp
        like_params[coupling] = fnfp

        like_params['GF'] = GF
        like_params['time_info'] = time_info

        # TODO: Right now im cutting corners. All sigmas and fnfps need to be passed directly into this function
        # and values need to be controled in main.py
        self.dm_recoils = dRdQ(energies, times, **like_params) * 10.**3. * s_to_yr
        
        self.dm_integ = R(Qmin=self.Qmin, Qmax=self.Qmax, **like_params) * 10.**3. * s_to_yr

        eng_lge = np.logspace(np.log10(self.Qmin), np.log10(self.Qmax), 400)
        b8_resp = np.zeros_like(eng_lge)
        for iso in isotopes:
            # isotope list [mT, Z, A, frac]
            b8_resp += Nu_spec().nu_rate('B8', eng_lge, iso)
        self.b8_resp = interp1d(eng_lge, b8_resp, kind='cubic', bounds_error=False, fill_value=0.)
        self.b8_int_resp = np.trapz(self.b8_resp(eng_lge), eng_lge)

    def like_multi_grad(self, norms):
        # Not used at moment
        b8_norm = norms[0]
        sig_dm = norms[1]
        return self.likelihood(b8_norm, sig_dm, return_grad=True)

    def like_multi_wrapper(self, norms):
        b8_norm = norms[0]
        sig_dm = norms[1]
        return self.likelihood(b8_norm, sig_dm, return_grad=False)


    def likelihood(self, b8_norm, sig_dm, return_grad=False):
        # TODO: consider implimenting gradient if minimization has difficulties
        # - 2 log likelihood
        # b8_norm in units of cm^-2 s^-1, sig_dm in units of cm^2
        like = 0.
        grad_x = 0.
        grad_nu = 0.

        n_obs = len(self.events)
        # total rate contribution

        dm_events = 10. ** sig_dm * self.dm_integ * self.exposure
        b8_events = 10. ** b8_norm * self.exposure * self.b8_int_resp
        like += 2. * (dm_events + b8_events)

        grad_x += 2. * np.log(10.) * dm_events
        grad_nu += 2. * np.log(10.) * b8_events
        #print 'Nobs: , B8 events: , DM events: ',n_obs, b8_events, dm_events
        
        # Differential contribution
        diff_dm = self.dm_recoils * self.exposure
        diff_b8 = self.b8_resp(self.events) * self.exposure

        lg_vle = (10. ** sig_dm * diff_dm + 10. ** b8_norm * diff_b8)
        for i in range(len(lg_vle)):
            if lg_vle[i] > 0.:
                like += -2. * np.log(lg_vle[i])
        #print lg_vle.sum()
        #grad_x += -2.*np.log(10.)*dm_events / lg_vle.sum()
        #grad_nu += -2. * np.log(10.) * b8_events / lg_vle.sum()

        # nu normalization contribution
        like += self.nu_gaussian(b8_norm)

        #grad_nu += self.nu_gaussian(b8_norm, return_deriv=True)

        if return_grad:
            return [grad_nu, grad_x]
        else:
            return like


    def nu_gaussian(self, flux_n, return_deriv=False):
        # - 2 log of gaussian flux norm comp
        # TODO generalize this function to any nu component
        b8_mean_f = 5.69 * 10. ** 6. # cm^-2 s^-1
        b8_sig = 0.91 * 10. ** 6.
        if not return_deriv:
            return b8_mean_f**2. * (10. ** flux_n - 1.)**2. / b8_sig**2.
        else:
            return b8_mean_f**2. * (10. ** flux_n - 1.) / b8_sig**2. * 2.*np.log(10.)
    

class Nu_spec(object):
    # Think about defining some of these neutino parameters as variables in constants.py (e.g. mean flux)
    
    def nu_rate(self, nu_component, er, element_info):
        mT, Z, A, xi = element_info
        conversion_factor = xi / mT * s_to_yr * (0.938 / (1.66 * 10.**-27.)) \
            * 10**-3. / (0.51 * 10.**14.)**2.
        # Where is this (A-Z)/100 coming from? Should not be there???
    
        diff_rate = np.zeros_like(er)
        for i,e in enumerate(er):
            e_nu_min = np.sqrt(mT * e / 2.)
            e_nu_max = 16.18 # B8 only
            b8_mean_f = 5.69 * 10. ** 6. # B8 cm^-2 s^-1
            diff_rate[i] = quad(self.nu_recoil_spec, e_nu_min, e_nu_max,
                                args=(e, mT, Z, A, nu_component), limit=50)[0]
            diff_rate[i] *= b8_mean_f * conversion_factor

        return diff_rate

    def max_er_from_nu(self, enu, mT):
        # return 2. * enu**2. / (mT + 2. * enu) # -- This formula is in 1307.5458, but it is
        # not consistent with the numbers they use in other papers...
        return 2. * enu**2. / mT
    
    def nu_recoil_spec(self, enu, er, mT, Z, A, nu_comp):
        if nu_comp == 'B8':
            return self.nu_csec(enu, er, mT, Z, A) * b8nu_spectrum(enu)
        else:
            return 0.

    def nu_csec(self, enu, er, mT, Z, A):
        Qw = (A - Z) - (1. - 4. * sw) * Z
        if er < self.max_er_from_nu(enu, mT):
            return gF ** 2. / (4. * np.pi) * Qw**2. * mT * \
                   (1. - mT * er / (2. * enu**2.)) * self.helm_ff(er, A, Z, mT)
        else:
            return 0.

    def helm_ff(self, er, A, Z, mT):
        q = np.sqrt(2. * mT * er) * MeVtofm
        rn = np.sqrt((1.2 * A**(1./3.))**2. - 5.)
        return (3. * np.exp(- q**2. / 2.) * (np.sin(q * rn) - q * rn * np.cos(q * rn)) / (q*rn)**3.)**2.

