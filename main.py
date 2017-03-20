"""

Code info:

"""

import numpy as np
from experiments import *
import numpy.random as random
from likelihood import *
from scipy.optimize import minimize, basinhopping
from scipy.stats import chi2
from scipy.integrate import quad
from scipy.interpolate import interp1d, RectBivariateSpline
from rate_UV import *
from helpers import *
import os
from scipy.stats import poisson

path = os.getcwd()

model = "sigma_si"  # [sigma_si, sigma_sd, sigma_anapole, sigma_magdip, sigma_elecdip,
                    # sigma_LS, sigma_f1, sigma_f2, sigma_f3, sigma_si_massless,
                    # sigma_sd_massless, sigma_anapole_massless, sigma_magdip_massless,
                    # sigma_elecdip_massless, sigma_LS_massless, sigma_f1_massless,
                    # sigma_f2_massless, sigma_f3_massless]

coupling = "fnfp" + model[5:]
mass = 6.
sigmap = 7.*10.**-45.
fnfp = 1.
delta = 0.  # Only works for elastic scattering at this point

element = 'germanium'  # ['germanium', 'xenon', 'argon', 'silicon', 'fluorine']
exposure = 1.  # Ton-yr

GF = False
time_info = False

print 'Run Info:'
print 'Experiment: ', element
print 'Model: ', model
print 'Coupling: ', coupling, fnfp
print 'Mass: {:.0f}, Sigma: {:.2e}'.format(mass, sigmap)

file_info = path + '/Saved_Files/'
file_info += element + '_' + model + '_' + coupling + '_{:.0f}'.format(fnfp)
file_info += '_Exposure_{:.1f}_tonyr_DM_Mass_{:.0f}_GeV'.format(exposure, mass)
print 'Output File: ', file_info
print '\n'
experiment_info, Qmin, Qmax = Element_Info(element)

drdq_params = default_rate_parameters.copy()
drdq_params['element'] = element
drdq_params['mass'] = mass
drdq_params[model] = sigmap
drdq_params[coupling] = fnfp

drdq_params['GF'] = GF
drdq_params['time_info'] = time_info

# 3\sigma for Chi-square Dist with 1 DoF means q = 8.85
q_goal = 8.85


# make sure there are enough points for numerical accuracy/stability
er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), 500)
time_list = np.zeros_like(er_list)

# DM spectrum -- Careful, this needs to be implimented in a better way!
dm_spec = dRdQ(er_list, time_list, **drdq_params) * 10. ** 3. * s_to_yr
dm_rate = R(Qmin=Qmin, Qmax=Qmax, **drdq_params) * 10. ** 3. * s_to_yr * exposure
dm_pdf = dm_spec / dm_rate
cdf_dm = dm_pdf.cumsum()
cdf_dm /= cdf_dm.max()
dm_events_sim = int(dm_rate * exposure)

#np.savetxt(file_info + '_DM.dat', np.column_stack((er_list, dm_spec)))
print 'Predicted Number of DM events: {}'.format(dm_events_sim)
nevts_dm = poisson.rvs(int(dm_events_sim))

# TODO generalize beyond B8 + DM
nu_comp = 'B8'
# neutrino ER spectrum
nuspec = np.zeros_like(er_list)
for iso in experiment_info:
    nuspec += Nu_spec().nu_rate(nu_comp, er_list, iso)

nu_rate = np.trapz(nuspec, er_list)
nu_pdf = nuspec / nu_rate
cdf_nu = nu_pdf.cumsum()
cdf_nu /= cdf_nu.max()
Nu_events_sim = int(nu_rate * exposure)
#np.savetxt(file_info + '_nub8.dat', np.column_stack((er_list, nuspec)))

print 'Predicted Number of Nu events: {}'.format(Nu_events_sim)
nevts_n = poisson.rvs(int(Nu_events_sim))


# Simulate events
Nevents = nevts_n + nevts_dm
print 'Simulation {:.0f} events...'.format(Nevents)
u = random.rand(Nevents)
# Generalize to rejection sampling algo for time implimentation
e_sim = np.zeros(Nevents)
for i in range(Nevents):
    if i < int(nevts_n):
        e_sim[i] = er_list[np.absolute(cdf_nu - u[i]).argmin()]
    else:
        e_sim[i] = er_list[np.absolute(cdf_dm - u[i]).argmin()]
times = np.zeros_like(e_sim)
#print e_sim

print 'Running Likelihood Analysis...'
# Minimize likelihood -- MAKE SURE THIS MINIMIZATION DOESNT FAIL. CONSIDER USING GRADIENT INFO
like_init_nodm = Likelihood_analysis(model, coupling, mass, 0., fnfp,
                                     exposure, element, experiment_info, e_sim, times,
                                     Qmin=Qmin, Qmax=Qmax, time_info=time_info, GF=False)
max_nodm = minimize(like_init_nodm.likelihood, np.array([0.]), args=(np.array([-100.])), tol=0.01)
#print max_nodm

like_init_dm = Likelihood_analysis(model, coupling, mass, 1., fnfp,
                                   exposure, element, experiment_info, e_sim, times,
                                   Qmin=Qmin, Qmax=Qmax, time_info=time_info, GF=False)
max_dm = minimize(like_init_dm.like_multi_wrapper, np.array([0., np.log10(sigmap)]), tol=0.01,
                jac=False)


print 'BF Neutrino normalization without DM: {:.2e}'.format(10.**max_nodm.x[0])
print 'BF Neutrino normalization with DM: {:.2e}'.format(10.**max_dm.x[0])
print 'BF DM sigma_p: {:.2e} \n\n'.format(10.**max_dm.x[1])

test_stat = np.max([max_nodm.fun - max_dm.fun, 0.])

pval = chi2.sf(test_stat,1)

print 'True DM mass: ', mass
print 'True DM sigma_p: ', sigmap
print 'TS: ', test_stat
print 'p-value: ', pval


