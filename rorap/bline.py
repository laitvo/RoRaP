#!/usr/bin/python
# -*- coding: utf-8 -*-

#bline.py
#Vojta Laitl in DIFFER, 2023
#reproduced from PuPlasAn (https://github.com/laitvo/PuPlasAn, Petr Kubelík and Vojta Laitl in the Czech Academy of Sciences, 2022)
#this work is shared under an Apache 2.0 license code: https://github.com/laitvo/PuPlasAn/blob/main/LICENSE
#please reach out for a citation note before publishing results of this code


'''
Baseline correction module
--------------------------

This module contains functions for determination the base line
of experimental data (e.g., UV--ViS spectrum). The base line is obtained
by iterative removing the high frequency signal and interpolating
the remaining points by the spline method.
'''

import warnings
import numpy as np
from scipy.interpolate import UnivariateSpline
from .filters import smooth_ma


def base(wn, I_wn, wpn, noise, rem=0.5, k=1, fulloutput=False, cbfun=None,
         cbfunargs=(), remrate=1.5):
    '''Calculates the baseline of the input intensity data *I_wn* using an iterative removing
    of the high frequency signal. The method is based on application of moving
    average smoothing.

    Parameters
    ----------
    wn : numpy.array
        array of wavenumbers to be fitted against
        *wn* must be wavenumbers in cm-1
        *wn* must be strictly increasing
    I_wn : numpy.array
        intensity spectrum
        *I_wn* must correspond to strictly increasing *wn*
    wpn : int
        number of the adjacent values of *I_wn* being averaged
        *wpn* must be an odd integer
    noise : float OR numpy.array
        Estimated noise level of the data in the *I_wn* array 
        if *noise* is an array, it must be of the same length as *wn*
    rem : float
        during the calculation the elements of *I_wn* containing the high
        frequency signal are removed iteratively. The iterations are stopped
        when the number of the remining elements in the *I_wn* is equal or lower
        then lenx*(1. - *rem*), where *lenx* is the original length
        of the *I_wn* array.
    k : int
        Degree of the smoothing spline
        *k* must be <= 5
        by defaul, *k* equals *1* (invoking a linear spline)
    fulloutput : bool
        boolean algebra object influencing the output (see the *Returns* section).
        by default, *fulloutput* equals *False*
    cbfun : callable OR None, optional *cbfun(b_inds, ma, \*cbfunargs)*
        a callback function executed at the end of each iteration
        (*cbfun(b_inds, ma, *cbfunargs)*),
        *b_inds* is an numpy.array containing the actual indices
        of the array *I_wn* which are used to construct the baseline,
        *ma* is an array of smoothed values of *I_wn* corresponding to the indexes
        in the *b_inds* array
    cbfunargs : tuple, optional
        Extra positional arguments passed to *cbfun*
    remrate : float
        A number in (1,inf] range 
        recommended values are between 1.05 and 3
        the higher is the value the more signal points
        are removed during one iteration
        by default, *remrate* equals *1.5*

    Returns
    -------
    If *fulloutput* is *True*, returns

    spl(wn) : numpy.array
        array containing the *I_wn* data base line
        *spl(wn)* corresponds to strictly increasing *wn*
    spl : instance
        an instance of the *scipy.interpolate.fitpack2.LSQUnivariateSpline* class
        *spl* passed on *wn* data returns the numerical base line
    b_inds : numpy.array
        array containing the indices of *I_wn* corresponding to the base line with the high frequency
        signal removed
        *b_inds* corresponds to strictly increasing *wn*        


    Else, returns

    spl(wn) : numpy.array
        array containing the *I_wn* data base line
        *spl(wn)* corresponds to strictly increasing *wn*
    '''

    lenx = len(wn)
    b_inds = np.arange(lenx)
    remi = [0]
    while len(remi) > 0 and len(b_inds) > lenx*(1. - rem):
        try:
            ma = smooth_ma(I_wn[b_inds], wpn)
            d = abs(ma - I_wn[b_inds])
            try:
                float(noise)
                remi = np.where((d > max(d)/remrate) & (d > noise))[0]
            except:
                remi = np.where((d > max(d)/remrate) & (d > noise[b_inds]))[0]
            if cbfun:
                cbfun(b_inds, ma, *cbfunargs)
            b_inds = np.delete(b_inds, remi, 0)
        except KeyboardInterrupt:
            break
    ma = smooth_ma(I_wn[b_inds], wpn)
    spl = UnivariateSpline(wn[b_inds], ma, k=k, s=0)
    if fulloutput:
        return spl(wn), spl, b_inds
    else:
        return spl(wn)



