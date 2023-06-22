#!/usr/bin/python
# -*- coding: utf-8 -*-

#raman.py
#Vojta Laitl in DIFFER, 2023

##minimal working example of using the RoRaP program scripts
##example evaluation of 26 spectra read from within a single fiber array
###WARNING!!!###
##due to the data protection policies of the data provider, the calibration is fully arbitraty and thus FALSE
##any reasons drawn from this example do NOT any have physical meaning
##before running a serious data sweep, please tackle your own calibration and pay attention to the initial conditions below
#the authors of this minimal example are NOT responsible for manipulating any results drawn herewith but certainly DO welcome constructive feedback regarding the script and data handling
###WARNING!!!###
import numpy as np
import pylab as pl
pl.ion()
import os
import sys
rorap_path = '../'
##assuming the program is stored in a preceding directory (as copied from GitHub) -- to be changed accordingly
##importing the customised modules and defining physical constants
sys.path.append(rorap_path)
import rorap.fiber as fiber
import rorap.raman as raman
import rorap.noise as ns
import rorap.bline as bl
import rorap.pdetect as pd
from scipy.constants import h,c,k
import lmfit
##filters out customised warnings if searching for the noise in too short data
##initial settings to the noise and baseline estimation should be looked into before a serious data sweep
import warnings
warnings.filterwarnings('ignore')

##defining a list of molecules to be looked into
molecules = ['CO2', 'CO', 'O2-X']
##'O2-a' and 'O2-b' may optionally be added -- vide supra

def O2_partition(T_e):
    '''Estimates an approximate partition sum to the three lowest O2 electronic states and calculates their ratios.
    
    Parameters
    ----------
    T_e : float
        electron temperature of the system
        *T_e* must be temperature in kelvins

    Returns
    -------
    x_O2_X : float
        ratio of the O2 ground state
    x_O2_a : float
        ratio of the O2-a state
    X_O2_b : float
        ratio of the O2-b state
    '''

    q_O2 = 3. + 5.*np.exp(-h*c*100.*7882.4/(k*T_e)) + 1.*np.exp(-h*c*100.*13120.9/(k*T_e))
    ##wavenumbers taken from https://en.wikipedia.org/wiki/Singlet_oxygen
 
    x_O2_X = 3./q_O2
    x_O2_a = 5.*np.exp(-h*c*100.*7882.4/(k*T_e))/q_O2
    x_O2_b = 1.*np.exp(-h*c*100.*13120.9/(k*T_e))/q_O2

    return x_O2_X, x_O2_a, x_O2_b

def plasma(params, wn_0, spectrum):
    '''Solves a parameters' offset for relative intensity calibration.
    
    Parameters
    ----------
    params : lmfit.parameter.Parameters
        generic *lmfit* parameters object
        *params* may contain temperature or other possibly offset parameters to optimise
        *params*'s units must correspond to those used for plotting a theoretical Raman spectrum (e.g., kelvins for temperature)
    wn : numpy.array
        experimentally recorded wavenumber axis
        *wn* are formal wavenumbers in cm-1
        *wn* are strictly increasing
    spectrum : numpy.array
        experimental rotational Raman spectrum
        *spectrum* must correspond to strictly increasing *wn*
        *spectrum* must be normalised

    Returns
    -------
    res : numpy.array
        array containing absolute residua mesured between experimental and theoretical Raman spectrum   
    '''
    
    ##retrieving free parameters
    T_rot = params['T_rot'].value
    T_vib = params['T_rot'].value
    ##assuming the reactor is fully thermalised -- may be changed to `T_vib = params['T_vib'].value'
    #T_e = params['T_rot'].value
    ##only needed when looking into O2 metastables
    ##assuming the reactor is fully thermalised -- may be changed to `T_e = params['T_e'].value'    
    x_CO2 = params['x_CO2'].value
    x_CO = params['x_CO'].value
    x_O2 = params['x_O2'].value

    wn_off = params['wn_off'].value
    fwhm = params['fwhm'].value

    wn = wn_0+wn_off

    J = np.arange(351)
    delta_J = 2.
 
    #x_O2_X, x_O2_a, x_O2_b = x_O2*np.array(O2_partition(T_e))
    ##to be uncommented when looking into O2 metastables

    abundances = {'CO2': x_CO2, 'CO': x_CO, 'O2-X': x_O2}
    ##to be replaced with `abundances = {'CO2': x_CO2, 'CO': x_CO, 'O2-X':x_O2_X, 'O2-a': x_O2_a, 'O2-b': x_O2_b}' when looking into O2 metastables

    I_wn = np.zeros(len(wn))

    for species in molecules:

        E_J, E_delta_J = raman.E_J(J, raman.B0[species], raman.D0[species], delta_J)
        nu_J = raman.nu_J(E_J, E_delta_J)
        
        if species == 'CO2':

            phi = raman.vib_weight_CO2(T_vib)
            g_J = raman.g_J(J, raman.g_s[species], raman.g_a[species], phi)

        else:

            g_J = raman.g_J(J, raman.g_s[species], raman.g_a[species])

        b_J = raman.b_J(J)
        Q_rot = raman.Q_rot(raman.B0[species], raman.sigma_0[species], T_rot)
        pop_J = raman.pop_J(abundances[species], J, g_J, Q_rot, E_J, T_rot)
        dSigma_dOhm = raman.dSigma_dOhm(raman.g2[species], b_J, nu_0, nu_J)
        I_J = raman.roRaman_uncoupled(pop_J, dSigma_dOhm)

        if species == 'CO2':

            nu_vJ = nu_J
            I_vJ = I_J

        else:

            nu_vJ, I_vJ = raman.rovib_coupling(J, nu_J, I_J, raman.we[species], raman.wexe[species], raman.alpha[species], T_vib, raman.gamma_v[species])

        signal_molecule = raman.roRaman_spectrum(wn, nu_vJ, I_vJ, np.ones(len(nu_vJ))*fwhm)[1]
        I_wn += signal_molecule

    I_wn_0 = I_wn/max(I_wn)
    res = I_wn_0-spectrum

    return res

##declaring the incident laser wavelength and an experimental rotational peak FWHM's estimate
nu_0 = 1e07/532.
fwhm = 1.45

##declaring the fiber positions for a wavelength-calibrated system
fiber_pos = [
[[0,17],384], #1
[[18,37],393], #2
[[38,57],395], #3
[[58,77],405], #4
[[78,97],411], #5
[[98,117],423], #6
[[118,136],425], #7
[[137,156],428], #8
[[157,176],430], #9
[[177,195],435], #10
[[196,215],440], #11
[[216,235],440], #12
[[236,255],445], #13
[[256,275],440], #14
[[276,295],444], #15
[[296,315],437], #16
[[316,335],440], #17
[[336,355],440], #18
[[356,375],430], #19
[[376,395],430], #20
[[396,415],420], #21
[[416,435],416], #22
[[436,455],406], #23
[[456,475],400], #24
[[476,495],388], #25
[[496,512],376] #26
]

##declaring exemplified experimental data
##background image follows a real measurement
experiment = [
'minimal_example.spe',
'minimal_example-BCKG.spe'
]

##declaring calibration factors to be loaded as `InterpolatedUnivariateSpline' instances
###WARNING!!!###
##due to the data protection policies of the data provider, the calibration is fully arbitraty and thus FALSE
##any reasons drawn from this example do NOT any have physical meaning
##before running a serious data sweep, please tackle your own calibration and pay attention to the initial conditions below
#the authors of this minimal example are NOT responsible for manipulating any results drawn herewith but certainly DO welcome constructive feedback regarding the script and data handling
###WARNING!!!### 
calibration_FALSE = np.load('calibration_FALSE.npy', allow_pickle=True)

##declaring the fiber bundle positioning, the sensor dispersion and the cutoff pixels for the central fiber relative to the laser's wavelength
pos_0 = lambda y: y/26.5 + 7332.5/26.5
dispersion = 0.011550
cutoff = 108
lambda_0 = 532.
##declaring the binning scheme (unbinned sensor hereinafter)
binning = [1,1]

data = fiber.fiber_load(experiment)
fiber_spectra = fiber.fiber_spectra(data,pos_0,lambda_0,fiber_pos,binning,dispersion,cutoff)[1]

##declaring initial conditions
T_rot = 5000.
T_vib = 7000.
T_e = 10000.
x_CO2 = 0.50
x_CO = 0.20
x_O2 = 0.10
##declaring the offset wavenumbers as a parameter to be optimised for wavelength readout misalignment
wn_off = 0.

##declaring free parameters based on the initial conditions
param = lmfit.Parameters()
param.add('T_rot', value=T_rot, vary=True, min=1500., max=7500.)
param.add('T_vib', value=T_vib, vary=True, min=T_rot, max=2.*T_rot)
#param.add('T_e', value=T_e, vary=True, min=T_vib, max=2.*T_vib)
##only needed when looking into O2 metastables
param.add('x_CO2', value=x_CO2, vary=True, min=0., max=1.)
param.add('x_CO', value=x_CO, vary=True, min=0.1, max=1.)
param.add('x_O2', value=x_O2, vary=True, min=0.1, max=1.)
param.add('wn_off', value=wn_off, vary=True, min=-2., max=2.)
param.add('fwhm', value=fwhm, vary=True, min=0.75*fwhm, max=1.25*fwhm)

##enumerating along the list of fiber array members
for i in range(len(fiber_spectra)):
    
    ##reading and FALSE-calibrating a spectrum
    wn = fiber_spectra[i][:,0]
    signal = fiber_spectra[i][:,1]
    noise = ns.noise_estim_ma_xdependent(wn, signal, 13, 25, 25)
    bline = bl.base(wn,signal,25,noise)
    ##normalising the result
    spectrum = calibration_FALSE[i](wn)*(signal-bline)/(max(signal-bline))

    ##detecting high-frequency signal peaks
    px, py = pd.detect(wn, spectrum, noise/max(signal), 3., 'up')
    
    ##simulating the spectrum as a sum of Voigt line profiles
    fit = np.ones(len(wn))

    for ix in range(len(px)):

        fit += raman.voigt(wn, px[ix], py[ix]*fwhm, fwhm)
    
    ##mimicking noise-free experimental spectrum with a fit
    spectrum_0 = fit+bline/max(signal)

    ##optimising FALSE physical conditions onto a (re-)normalised fitted spectrum
    diagnostics = lmfit.minimize(plasma, param, args=(wn,spectrum_0/max(spectrum_0)))
    ##printing FALSE results
    ##saving of any ouputs may be customised AFTER fine-tackling the calibration data and initial conditions to the optimisation
    print(diagnostics.message)
    print(diagnostics.params)
    ##estimating the optimisation routine's root mean square deviation
    rmsd = np.sqrt(diagnostics.residual**2/len(spectrum))
    ##FALSE example plotting
    fig = pl.figure(figsize=(7.71,6.71))
    pl.plot(wn,signal/max(signal),'b',label='signal')
    pl.plot(wn,bline/max(signal),'grey',label='baseline')
    pl.plot(wn,spectrum_0,'green',label='fit')
    pl.plot(wn,rmsd,'red',label='RMSD')
    pl.xlabel(r'$\tilde{\nu}$ (cm$^{-1}$)',size=20) 
    pl.ylabel(r'Signal (arb. u.)',size=20)
    pl.title('FALSE example data', size=20)
    pl.legend(loc='upper right', prop={'size': 15})
    pl.tick_params(labelsize=20)   

