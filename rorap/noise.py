#!/usr/bin/python
# -*- coding: utf-8 -*-

#noise.py
#Vojta Laitl in DIFFER, 2023
#reproduced from PuPlasAn (https://github.com/laitvo/PuPlasAn, Petr Kubelík and Vojta Laitl in the Czech Academy of Sciences, 2022)
#this work is shared under an Apache 2.0 license code: https://github.com/laitvo/PuPlasAn/blob/main/LICENSE
#please reach out for a citation note before publishing results of this code


'''
Signal/noise module
-------------------

Contains functions for estimating the spectral data noise level.
'''

import warnings
#warnings.filterwarnings('ignore')
import numpy as np
from scipy.interpolate import UnivariateSpline
from .filters import smooth_ma


def std_ma(noisedata, wpn):
    '''Calculates the standard deviation of the difference between *noisedata*
    array and its moving average.

    Parameters
    ----------
    noisedata : numpy.array
        array containing net noise disposed of high-frequency signal
    wpn : int
        number of the adjacent values of *I_wn* being averaged
        *wpn* must be an odd integer

    Returns
    -------
    stdev : numpy.array
        standard deviation of the difference between *noisedata*
        array and its moving average (i.e., the standard deviation of the base line 
        corrected data with no signal)
    '''

    if noisedata.__class__ != np.ndarray:
        noisedata = np.array(noisedata)
    lendata = len(noisedata)
    if lendata < wpn*3:
        warnings.warn('len(noisedata) < wpn*3', UserWarning)
    base = smooth_ma(noisedata, wpn)
    stdev = np.sqrt(np.sum((base - noisedata)**2)/lendata)

    return stdev


def noise_estim_ma(I_wn, wpn, n, m=10):
    '''Estimates the *I_wn* data noise level by calculating the noise for *m* 
    short subarrays (of length equal to *n*) randomly selected
    from the *I_wn* array. The resulting noise level corresponds to the minimal
    value of the standard deviations calculated for all short subarrays.

    Parameters
    ----------
    I_wn : numpy.array
        noised intensity spectrum
    wpn : int
        number of the adjacent values of *I_wn* being averaged
        *wpn* must be an odd integer
    n : int
        length of the randomly selected subarray used for calculation
        of the local noise level
    m : int
        number of the randomly selected subarrays
        by default, *n* equals *10*

    Returns
    -------
    min(stds) : float
        estimated noise level of the *y* data
    '''

    stds = []
    mini = 0
    maxi = len(I_wn) - wpn 
    for i1 in range(m):
        rn = np.random.randint(mini, maxi)
        stds.append(std_ma(I_wn[rn:rn+n], wpn=wpn))

    return min(stds)


def noise_estim_ma_xdependent(wn, I_wn, wpn, n, partnum, m=10, k=1,
                              fulloutput=False):
    '''Estimates the *wn*-dependent *I_wn* data noise level.

    Parameters
    ----------
    wn : numpy.array
        array of wavenumbers to be fitted against
        *wn* must be wavenumbers in cm-1
        *wn* must be strictly increasing
    I_wn : numpy.array
        noised intensity spectrum
        *I_wn* must correspond to strictly increasing *wn*
    wpn : int
        number of the adjacent values of *I_wn* being averaged
        *wpn* must be an odd integer
    n : int
        length of the randomly selected subarray used for calculation
        of the local noise level
    partnum : int
        number of equidistant points for which the *x*-dependent
        noise is calculated
    m : int
        number of the randomly selected subarrays
        by default, *n* equals *10*
    k : int
        Degree of the smoothing spline
        *k* must be <= 5
        by defaul, *k* equals *1* (invoking a linear spline)
    fulloutput : bool
        boolean algebra object influencing the output (see the *Returns* section).
        by default, *fulloutput* equals *False*

    Returns
    -------
    If *fulloutput* is *True*, returns

    spl(wn) : numpy.array
        array containing the *wn*-dependent *I_wn* data noise level
        *spl(wn)* corresponds to strictly increasing *wn*
    spl : instance
        an instance of the *scipy.interpolate.fitpack2.LSQUnivariateSpline* class
        *spl* passed on *wn* data returns the numerical noise
    sdevx : list
        list of *wn*-values (positions) of the calculated noise
        length of *sdevx* is equal to *partnum*
    sdevy : list
        list of *wn*-dependent *I_wn* data noise corrresponding to *sdevx*  


    Else, returns

    spl(wn) : numpy.array
        array containing the *wn*-dependent *I_wn* data noise level
        *spl(wn)* corresponds to strictly increasing *wn*
    '''

    partlen = int(len(wn)/partnum)
    if partlen <= 1:
        raise Exception('"partnum" is too high ' +
                        'or length of "x" and "y" is to low')
    sdevx = []
    sdevy = []
    for i1 in range(partnum):
        ind1 = i1*partlen
        ind2 = i1*partlen + partlen
        sdevx.append(sum(wn[ind1:ind2])/(ind2-ind1))
        sdevy.append(noise_estim_ma(I_wn[ind1:ind2], wpn=wpn, n=n, m=m))
    sdevx = [wn[0]] + sdevx
    sdevy = [sdevy[0]] + sdevy
    sdevx.append(wn[-1])
    sdevy.append(sdevy[-1])
    spl = UnivariateSpline(sdevx, sdevy, k=k, s=0)
    if fulloutput:
        return spl(wn), spl, sdevx, sdevy
    else:
        return spl(wn)


