#!/usr/bin/python
# -*- coding: utf-8 -*-

#pdetect.py
#Vojta Laitl in DIFFER, 2023
#reproduced from PuPlasAn (https://github.com/laitvo/PuPlasAn, Petr Kubelík and Vojta Laitl in the Czech Academy of Sciences, 2022)
#this work is shared under an Apache 2.0 license code: https://github.com/laitvo/PuPlasAn/blob/main/LICENSE
#please reach out for a citation note before publishing results of this code


'''
Peak detection module
---------------------

This module contains functions (especially the *detect* function)
for detection peaks in the experimental data (e.g., an UV--ViS spectrum).
'''

import numpy as np
import copy

def detectTop_inds(I_wn, ptype='up'):
    '''Selects the indices of *I_wn* where the optimum of a peak is detected.

    Parameters
    ----------
    I_wn : numpy.array
        (possibly noised) intensity spectrum
    ptype : str
        *"up"* (invoking the detection of local maxima) and *"down"* (invoking the detection of local minima) 
        are only accepted as inputs        
        by default, *ptype* equals *"up"*

    Returns
    -------
    intersec : numpy.array
        array containing the indexes of *I_wn* where the optimum of a peak is detected
    '''

    dy = np.diff(I_wn)
    dy1 = dy[:-1]
    dy2 = dy[1:]
    dymul = dy1 * dy2
    dysub = dy1 - dy2
    dymulinds = np.where(dymul < 0.0)[0]
    if ptype == 'up':
        dysubinds = np.where(dysub > 0.0)[0]
    elif ptype == 'down':
        dysubinds = np.where(dysub < 0.0)[0]
    else:
        raise Exception('invalid ptype (must be set to "up" or "down",' +
                        ' actual value == %s)' % str(ptype))
    intersec = np.intersect1d(dymulinds, dysubinds) + 1

    return intersec


def detectThr_inds(I_wn, thr, ptype='up'):
    '''Selects the indices of *I_wn* where the values are higher (*ptype* equals *"up"*)
    or lower (*ptype* equals *"down"*) than *thr*.

    Parameters
    ----------
    I_wn : numpy.array
        noised intensity spectrum
    thr : float OR numpy.array
        selection threshold (if array, it is a *wn*-dependent threshold)
    ptype : str
        *"up"* (selects the indices of values higher than *thr*) 
        and *"down"* (selects the indices of values higher than *thr*) 
        are only accepted as inputs        
        by default, *ptype* equals *"up"*

    Returns
    -------
    inds : numpy.array
        array containing the indices of *I_wn* where the values
        are higher (*ptype* equals *"upp"*) or lower (*ptype* equals *"down"*) than *thr*
    '''

    if ptype == 'up':
        inds = np.where(I_wn > thr)[0]
    elif ptype == 'down':
        inds = np.where(I_wn < thr)[0]
    else:
        raise Exception('invalid ptype (must be set to "up" or "down",' +
                        ' actual value == %s)' % str(ptype))

    return inds


def detect(wn, I_wn, noise, snr_thr, ptype='up', fulloutput=False):
    '''Detect peaks' positions and highs in the input data arrays.

    Parameters
    ----------
    wn : numpy.array
        experimentally recorded wavenumber axis
        *wn* must be wavenumbers in cm-1
        *wn* must be strictly increasing
    I_wn : numpy.array
        noised intensity spectrum
        *I_wn* must correspond to strictly increasing *wn*
    noise : float OR numpy.array
        Estimated noise level of the data in the *I_wn* array 
        if *noise* is an array, it must be of the same length as *wn*
    snr_thr : float
        the peak detection threshold
        is calculated as *noise* \* *snr_thr*
    ptype : str
        *"up"* (selects the indices of values higher than *thr*) 
        and *"down"* (selects the indices of values higher than *thr*) 
        are only accepted as inputs        
        by default, *ptype* equals *"up"*
    fulloutput : bool
        boolean algebra object influencing the output (see the *Returns* section).
        by default, *fulloutput* equals *False*

    Returns
    -------
    If *fulloutput* is *True*, returns

    wn[pinds] : numpy.array
        array containing containing the local extrema positions detected on the *wn* axis
        *wn[pinds]* are wavenumbers in cm-1
    I_wn[pinds] : numpy.array
        array containing the local extrema values detected
        *I_wn[pinds]* corresponds to *wn[pinds]*
    pinds : numpy.array
        array of indices corresponding to locating the local extrema in *wn[pinds]* and/or *I_wn[pinds]*


    Else, returns

    wn[pinds] : numpy.array
        array containing containing the local extrema positions detected on the *wn* axis
        *wn[pinds]* are wavenumbers in cm-1
    I_wn[pinds] : numpy.array
        array containing the local extrema values detected
        *I_wn[pinds]* corresponds to *wn[pinds]*
    '''

    thr = noise * snr_thr
    pinds = np.intersect1d(detectThr_inds(I_wn, thr, ptype=ptype),
                           detectTop_inds(I_wn, ptype=ptype))
    if fulloutput:
        return wn[pinds], I_wn[pinds], pinds
    else:
        return wn[pinds], I_wn[pinds]





