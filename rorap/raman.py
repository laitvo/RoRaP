#!/usr/bin/python
# -*- coding: utf-8 -*-

#raman.py
#Vojta Laitl in DIFFER, 2023


'''
Theoretical Raman spectra module

This simpler library contains functions drafted for generating a theoretical rotational Raman spectrum based on Dunham series's expansion coefficients and under conditions relevant for plasma experiments (e.g., those incentivising rotovibrational coupling and altering rotational degeneracies). CO2 and N2 case studies are proposed by the molecular constants dictionaries shown below.
'''
#partially reproduced from Alex v/d Steeg

import numpy as np
import pylab as pl
pl.ion()
import scipy.constants as cons
from scipy.constants import h, c, k, epsilon_0
from scipy.special import wofz
import random
import time


###Molecular constant dictionaries
B0 = {'CO2': 0.39020,    'CO': 1.92252,      'O2-X': 1.43780,  'N2': 1.98973,  'O2-a': 1.4264  ,'O2-b':    1.4003}
#rotational constant for equilibrium separation (cm-1), Y01 in Dunham series
D0 = {'CO2': 0.12e-6,    'CO': 6.1193e-6,    'O2-X': 4.8e-6,   'N2': 0.0,      'O2-a': 4.86e-6 ,'O2-b':    5.35e-6}
#centrifugal distortion constant (cm-1), -Y02 in Dunham series
###sectic and higher distortions are neglected
we    = {'CO2': None,          'CO': 2169.81,      'O2-X': 1580.19,  'N2': 2358.58,  'O2-a': 1483.5  ,'O2-b':    1432.77}
#fundamental vibration frequencies for diatomics (CO2 NoneTyped, vibrations corrected elsewhere) (cm-1), Y10 in Dunham series
wexe  = {'CO2': None,          'CO': 13.2883,      'O2-X': 11.98 ,   'N2': 14.324,   'O2-a': 12.9    ,'O2-b':    14.00}
#first anharmonicity correction for diatomics (CO2 NoneTyped, vibrations corrected elsewhere) (cm-1), -Y20 in Dunham series
alpha = {'CO2': 0,          'CO': 0.01750,      'O2-X': 0.0159 ,  'N2': 0.01731,  'O2-a': 0.0171,'O2-b':    0.01820}
#first-order rotational constant (cm-1), -Y11 in Dunham series
sigma_0 = {'CO2': 2., 'CO': 1., 'O2-X': 2., 'N2': 2., 'O2-a': 2., 'O2-b': 2.}
#classical symmetry number (2 for D(inf)h, 1 for C(inf)v)
g2 = {'CO2': 23.74  ,    'CO': 6.09      ,   'O2-X': 12.67  ,  'N2': 7.91   ,  'O2-a': 1.26446 ,'O2-b':    1.26446}
#second hyperpolarisibility (F2m4)
gamma_v = {'CO2': lambda v: 1., 'O2-X': lambda v: (1.097 + 0.0339*v*1.445 + 0.00040*v**2*1.445)**2/(1.097)**2,  'CO': lambda v: 1.+(0.055 * v), 'N2':  lambda v: (0.719 + 0.0177*v*1.445 + 0.000150*1.445*v**2)**2/(0.719)**2, 'O2-a': lambda v: (1.097 + 0.0339*v*1.445 + 0.00040*v**2*1.445)**2/(1.097)**2, 'O2-b': lambda v: (1.097 + 0.0339*v*1.445 + 0.00040*v**2*1.445)**2/(1.097)**2}
#polarisibilty matrix elements fractioned as a function of vibrational quanta *v*:
    #CO2: unity function, compensated elsewhere
    #O2: https://doi.org/10.1016/S0022-2852(02)00012-7
    #CO: https://doi.org/10.1364/OL.7.000440
    #N2: https://doi.org/10.1016/S0022-2852(02)00012-7
g_s   = {'CO2': 1.,          'CO': 1.,            'O2-X': 0.,        'N2': 6.,        'O2-a': 1.       ,'O2-b':    1.}
#statistical weight/degeneracy for 's' symmetry transitions (even for singlet terms)
g_a   = {'CO2': 0.,          'CO': 1.,            'O2-X': 1.,        'N2': 3.,        'O2-a': 1.       ,'O2-b':    0.}
#statistical weight/degeneracy for 'a' symmetry transition (odd for singlet terms)


def gauss(wn, mu, a, fwhm):
    '''Calculates the Gaussian profile function from the input parameters.

    Parameters
    ----------
    wn : numpy.array
        array of wavenumbers to be fitted against
        *wn* must be wavenumbers in cm-1
        *wn* must be strictly increasing
    mu : float
        peak center position
        *mu* must be formal wavenumbers in cm-1
    a : float
        peak area
    fwhm : float
        peak full width at half maximum
        *fwhm* must be formal wavenumbers in cm-1

    Returns
    -------
    G_wn : numpy.array
        array containing the Gaussian peak profile
        corresponding to the independent variable *wn*
    '''

    sigma = fwhm/(2.*np.sqrt(2.*np.log(2.)))
    G_wn = ((1./(sigma * np.sqrt(2.*np.pi))) *
            np.exp(-0.5 * ((wn - mu)/sigma)**2)) * a

    return G_wn 


def lorentz(wn, mu, a, fwhm):
    '''Calculates the Lorentzian profile function from the input parameters.

    Parameters
    ----------
    wn : numpy.array
        array of wavenumbers to be fitted against
        *wn* must be wavenumbers in cm-1
        *wn* must be strictly increasing
    mu : float
        peak center position
        *nu_0* must be formal wavenumbers in cm-1
    a : float
        peak area
    fwhm : float
        peak full width at half maximum
        *fwhm* must be formal wavenumbers in cm-1

    Returns
    -------
    L_wn : numpy.array
        array containing the Lorentzian peak profile
        corresponding to the independent variable *wn*
    '''

    gamma = fwhm/2.
    L_wn = ((1./np.pi) * ((0.5*gamma)/((wn - mu)**2 + (0.5*gamma)**2))) * a

    return L_wn

def voigt(wn, mu, a, fwhm):
    '''Applies the Faddeeva function to estimate the Voigt profile function from the input parameters.

    Parameters
    ----------
    wn : numpy.array
        array of wavenumbers to be fitted against
        *wn* must be wavenumbers in cm-1
        *wn* must be strictly increasing
    mu : float
        peak center position
        *mu* must be formal wavenumbers in cm-1
    a : float
        peak area
    fwhm : float
        peak full width at half maximum
        *fwhm* must be formal wavenumbers in cm-1

    Returns
    -------
    V_wn : numpy.array
        array containing the Voigt peak profile
        corresponding to the independent variable *wn*
    '''

    sigma = fwhm/(2.*np.sqrt(2.*np.log(2.)))
    gamma = fwhm/2.

    V_wn = a * np.real(wofz((wn - mu + 1j*gamma)/sigma/np.sqrt(2.))) / sigma /np.sqrt(2.*np.pi)
    
    return V_wn

profiles = {'gauss': gauss, 'lorentz': lorentz, 'voigt': voigt}


def vib_weight_CO2(T_vib):
    '''Calculates the vibrationally coupled weighing for CO2 rotational quanta (https://doi.org/10.1364/AO.57.005694).
    Only the carbon dioxide case is exemplified here.

    Parameters
    ----------
    T_vib : float
        vibrational temperature
        *T_vib* must be temperature in kelvins

    Returns
    -------
    vib_weight : float
        vibrationally coupled weighing factor
    '''

    nu2 = 667.3 #bending mode wavenumbers (cm-1)
    nu3 = 2349.3 #asymmetric stretching mode wavenumbers (cm-1)
    
    z2 = np.exp(-h*c*100.*nu2/(k*T_vib))
    z3 = np.exp(-h*c*100.*nu3/(k*T_vib))

    vib_weight = 0.5+0.5*((1.-z2)/(1.+z2))*((1.-z3)/(1.+z3))
    
    return vib_weight

def g_J(J, g_s, g_a, vib_weight=None):
    '''Calculates the rotational statistical weights

    Parameters
    ----------
    J : numpy.array
        array of rotational quantum numbers applied
    g_s : float
        statistical weight for 's' symmetry transitions
    g_a : float
        statistical weight for 'a' symmetry transitions
    vib_weight : float OR NoneType
        vibrationally coupled s/a weighing
        by default, *vib_weight* is NoneTyped (no vibrational coupling is applied; the opposite case is CO2)

    Returns
    -------
    g_J : numpy.array
        array containing the statistical weights corresponding to rotational quanta *J*
    '''

    if vib_weight == None:
        vib_weight = g_s/(g_s+g_a)
    else:
        pass

    g_J = (g_s + g_a) * ((1 - (J % 2)) * vib_weight + (J % 2) * (1 - vib_weight))

    return g_J

def E_J(J, B0, D0, delta_J):
    '''Calculates rotational energies of a linear molecule with a 1st-degree non-rigidity approximation

    Parameters
    ----------
    J : numpy.array
        array of rotational quantum numbers applied
    B0 : float
        rotational constant for equilibrium separation
        *B0* must be formal wavenumbers in cm-1
    D0: float
        centrifugal distortion constant
        *D0* must be formal wavenumbers in cm-1
    delta_J : float
        allowed transitions' change of *J* quanta
        by default, *delta_J* equals *2.0*

    Returns
    -------
    E_J : numpy.array
        array containing the rotational energies
        *E_J* are formal wavenumbers in cm-1
    E_delta_J : numpy.array
        array containing the rotational energies of *J+delta_J* quanta
        *E_delta_J* are formal wavenumbers in cm-1      
    '''

    E_J = B0*J*(J+1) - D0*(J**2)*(J+1)**2
    E_delta_J = B0*(J+delta_J)*(J+delta_J+1) - D0*((J+delta_J)**2)*((J+delta_J)+1)**2

    return E_J, E_delta_J

def Q_rot(B0, sigma_0, T_rot):
    '''Calculates the rotational partition function of a linear molecule by using a 2nd degree expansion of (http://dx.doi.org/10.1063/1.454608)'s. Only the most abundant isotopologues (composed of 12C and/or 16O nuclei) are currently considered.

    Parameters
    ----------
    B0 : float
        rotational constant for equilibrium separation
        *B0* must be formal wavenumbers in cm-1
    sigma_0 : float
        classical symmetry number
    T_rot: float
        rotational temperature
        *T_rot* must be temperature in kelvins

    Returns
    -------
    Q_rot : float
        rotational partition function
    '''

    beta = h*c*100.*B0/(k*T_rot)
    Q_rot = (1./sigma_0)*np.exp(beta/3.)*(1.+(beta**2)/90.)

    return Q_rot

def pop_J(n_tot, J, g_J, Q_rot, E_J, T_rot):
    '''Partitions the rotational populations of a molecule.

    Parameters
    ----------
    n_tot : float
        total abundance of a given molecule
    J : numpy.array
        array of rotational quantum numbers applied
    g_J : numpy.array
        array containing the statistical weights corresponding to rotational quanta *J*
    Q_rot : float
        rotational partition function
    E_J : numpy.array
        array containing the rotational energies
        *E_J* must be formal wavenumbers in cm-1
    T_rot: float
        rotational temperature
        *T_rot* must be temperature in kelvins

    Returns
    -------
    pop_J : numpy.array
        fractioned populations corresponding to rotational quanta *J*
    '''

    pop_J_Stokes = n_tot*g_J*(2.*J+1.)*np.exp(-h*c*100.*E_J/(k*T_rot))/Q_rot
    pop_J_antiStokes = pop_J_Stokes[::-1]

    pop_J = np.concatenate([pop_J_antiStokes,pop_J_Stokes])

    return pop_J

def nu_J(E_J, E_delta_J):
    '''Calculates the rotational Raman's transition wavenumbers

    Parameters
    ----------
    E_J : numpy.array
        array containing the rotational energies
        *E_J* are formal wavenumbers in cm-1
    E_delta_J : numpy.array
        array containing the rotational energies of *J+delta_J* quanta
        *E_delta_J* are formal wavenumbers in cm-1  
        

    Returns
    -------
    nu_J : numpy.array
        rotational Raman transitions' wavenumbers corresponding to the *J* quanta
        *nu_J* are wavenumbers in cm-1
        *nu_J* are strictly increasing
    '''

    nu_Stokes = E_delta_J - E_J
    nu_antiStokes = (E_J - E_delta_J)[::-1]

    nu_J = np.concatenate([nu_antiStokes, nu_Stokes])

    return nu_J

def b_J(J):
    '''Calculates the J-dependent line strength factors of a linear molecule by means of Placzek-Teller coefficients 
    (https://pure.tue.nl/ws/portalfiles/portal/46924564/841494-1.pdf)

    Parameters
    ----------
    J : numpy.array
        array of rotational quantum numbers applied        

    Returns
    -------
    b_J : numpy.array
        line strenght factors of rotational transitions spaced by the *J* quanta
        *b_J* corresponds to strictly increasing *nu_J*
    '''

    b_Stokes = 3.*(J+1.)*(J+2.)/(2.*(2.*J+1)*(2.*J+3.))
    b_antiStokes = (3.*J*(J-1.)/(2.*(2.*J+1.)*(2.*J-1.)))[::-1]

    b_J = np.concatenate([b_antiStokes, b_Stokes])

    return b_J

def dSigma_dOhm(g2, b_J, nu_0, nu_J, pol_weight = 7./45.):
    '''Calculates the differential scattering cross section of defined transitions

    Parameters
    ----------
    g2 : float
        second hyperpolarisibility of the species considered
        *g2* must be a scalar in (F2 m4)
    b_J : numpy.array
        line strenght factors of rotational transitions spaced by the *J* quanta
    nu_0 : float
        formal wavenumbers corresponding to the exciting laser's frequency
        *nu_0* must be wavenumbers in cm-1
    nu_J : numpy.array
        rotational Raman transitions' corresponding to the *J* quanta
        *nu_J* must be wavenumbers in cm-1
        *nu_J* must be strictly increasing
    pol_weight : float
        weighing factor compensating for various polarisations collected
        *pol_weight* must be a positive floating point number
        by default *pol_weight* equals *7./45.* (i.e., both polarisation directions are imaged, 
        https://pure.tue.nl/ws/portalfiles/portal/46924564/841494-1.pdf)
        

    Returns
    -------
    dSigma_dOhm : numpy.array
        rotational Raman transitions' differential cross sections
        *dSigma_dOhm* is a formal cross section derivative in 1e-031 (cm2 sr-1)
        *dSigma_dOhm* corresponds to strictly increasing *nu_J*
    '''

    dSigma_dOhm = pol_weight*((16.*np.pi**4)/epsilon_0**2)*b_J*(((nu_0 + nu_J)*1e02)**4)*((g2*1e-41)**2)*1e04*1e031

    return dSigma_dOhm

def roRaman_uncoupled(pop_J, dSigma_dOhm):
    '''Calculates the stick diagram of a vibrationally uncoupled rotational Raman spectrum

    Parameters
    pop_J : numpy.array
        fractioned populations corresponding to rotational quanta *J*
    dSigma_dOhm : numpy.array
        rotational Raman transitions' differential cross sections
        *dSigma_dOhm* must be a formal cross section derivative in the multiples of (cm2 sr-1)

    Returns
    -------
    I_J : numpy.array
        stick diagram of unnormalised rotational Raman intensities
        *I_J* corresponds to strictly increasing *nu_J*

    '''

    I_J = pop_J*dSigma_dOhm
    
    return I_J

def Nv_diatomic(we, wexe, T_vib, v_max=6):
    '''Partitions vibrational populations of a diatomic (or a single vibrational mode)

    Parameters
    ----------
    we : float
        fundamental vibrational frequency
        *we* must be formal wavenumbers in cm-1
    wexe : float
        first-order anharmonicity correction
        *wexe* must be formal wavenumbers in cm-1
    T_vib : float
        vibrational temperature
        *T_vib* must be temperature in kelvins
    v_max : integer
        maximal vibrational quanta to be partitioned
        by default, *v_max* equals *6*

    Returns
    -------
    v : numpy.array
        array of vibrational quanta [0, *v_max*] to be partitioned
    x_V : numpy.array
        relative populations of vibrational energy states spaced by their quanta *v*s
    '''

    v = np.arange(v_max+1)
    E_v = (v+0.5)*h*c*100.*we - ((v+0.5)**2)*h*c*100.*wexe

    N_v = np.exp(-E_v/(k*T_vib))
    Q_v = 1./(1.-np.exp(-h*c*100.*we/(k*T_vib)))
    x_v = N_v/Q_v

    return v, x_v

def rovib_coupling(J, nu_J, I_J, we, wexe, alpha, T_vib, gamma_v, delta_J=2., v_max=6):
    '''Couples rotational Raman transitions onto vibrational populations via Dunham first-order anharmonicity correction  

    Parameters
    ----------
    J : numpy.array
        array of rotational quantum numbers applied 
    nu_J : numpy.array
        rotational Raman transitions' corresponding to the *J* quanta
        *nu_J* must be wavenumbers in cm-1
        *nu_J* must be strictly increasing
    I_J : numpy.array
        stick diagram of unnormalised rotational Raman intensities
        *I_J* must correspond to strictly increasing *nu_J*
    we : float
        fundamental vibrational frequency
        *we* must be formal wavenumbers in cm-1
    wexe : float
        first-order anharmonic oscillator correction (Y20)
        *wexe* must be formal wavenumbers in cm-1
    alpha : float
        first-order anharmonic vibrotor correction (Y11)
        *alpha* must be formal wavenumbers in cm-1
    T_vib : float
        vibrational temperature
        *T_vib* must be temperature in kelvins
    gamma_v : function *<function __main__.<lambda>(v)>*
        polarisibilty matrix elements fractioned as a function of vibrational quanta *v*
        *gamma_v* must be a single-argument function and return a dimensionless number
    delta_J : float
        allowed transitions' change of *J* quanta
        by default, *delta_J* equals *2.0*
    v_max : integer
        maximal vibrational quanta to be partitioned
        by default, *v_max* equals *6*

    Returns
    -------
    nu_vJ : numpy.array
        rotovibrational Raman transitions' wavenumbers
        *nu_vJ* are formal wavenumbers in cm-1
        *nu_vJ* are strictly increasing
    I_vJ : numpy.array
        stick diagram of unnormalised rotovibrational Raman intensities
        *I_vJ* corresponds to strictly increasing *nu_vJ*
    '''


    v, x_v = Nv_diatomic(we, wexe, T_vib, v_max)
    gamma_part = x_v*gamma_v(v)**2

    Y11_J = -alpha*(v+0.5).reshape(len(v),1)*(J*(J+1)).reshape(len(J),1).T
    Y11_delta_J = -alpha*(v+0.5).reshape(len(v),1)*((J+delta_J)*(J+delta_J+1)).reshape(len(J),1).T

    displacement_Stokes = Y11_delta_J - Y11_J
    displacement_antiStokes = (Y11_J - Y11_delta_J)[::-1]
    displacement = np.concatenate([displacement_antiStokes, displacement_Stokes],axis=1)

    I_vJ_matrix = gamma_part.reshape(len(gamma_part),1)*I_J.reshape(len(I_J),1).T
    nu_vJ_matrix = nu_J + displacement

    nu_vJ_sorted = np.argsort(nu_vJ_matrix.reshape(-1,1)[:,0])

    nu_vJ = nu_vJ_matrix.reshape(-1,1)[:,0][nu_vJ_sorted]
    I_vJ = I_vJ_matrix.reshape(-1,1)[:,0][nu_vJ_sorted]

    return nu_vJ, I_vJ

def roRaman_spectrum(wn, nu, I, fwhm, profile='voigt'):
    '''Generates a theoretical rotational Raman spectrum based on its stick diagram


    Parameters
    ----------
    wn : numpy.array
        experimentally recorded wavenumber axis
        *wn* must be wavenumbers in cm-1
        *wn* must be strictly increasing
    nu : numpy.array
        array containing theoretical Raman transitions' wavenumbers
        *nu* must be formal wavenumbers in cm-1
        *nu* must be strictly increasing    
    I : numpy.array
        stick diagram of theoretical Raman intensities
        indexing of *I* must correspond to that of *nu*
    fwhm : float
        peak full width at half maximum
        *fwhm* must be formal wavenumbers in cm-1
    profile : string
        line fitting profile name
        *'gauss'*, *'lorentz'*, and *'voigt'* are allowed inputs
        by default, *profile* equals *'voigt'*


    Returns
    -------
    wn : numpy.array
        experimentally recorded wavenumber axis
        *wn* are formal wavenumbers in cm-1
        *wn* are strictly increasing
    I_wn : numpy.array
        theoretical rotational Raman spectrum
        *I_wn* corresponds to strictly increasing *wn*
    '''


    if type(fwhm) == float:
        fwhm = fwhm*np.ones(len(nu))

    else:
        pass

    I = I[(nu>min(wn))&(nu<max(wn))]    
    nu = nu[(nu>min(wn))&(nu<max(wn))]

    I_wn = np.zeros(len(wn))
    
    for ix in range(len(nu)):
        I_wn += profiles[profile](wn, nu[ix], I[ix], fwhm[ix])

    return wn, I_wn
