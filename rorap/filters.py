#!/usr/bin/python
# -*- coding: utf-8 -*-

#filters.py
#Vojta Laitl in DIFFER, 2023
#reproduced from PuPlasAn (https://github.com/laitvo/PuPlasAn, Petr Kubelík and Vojta Laitl in the Czech Academy of Sciences, 2022)
#this work is shared under an Apache 2.0 license code: https://github.com/laitvo/PuPlasAn/blob/main/LICENSE
#please reach out for a citation note before publishing results of this code

'''
Filters module
--------------------------

This module assembles a smoothed moving average filter.
'''


import copy
import warnings
import numpy as np


def smooth_ma(I_wn, wpn):
    '''Performs moving average smoothing.

    Parameters
    ----------
    I_wn : numpy.array
        noised intensity spectrum
    wpn : int
        number of the adjacent values averaged
        *wpn* must be an odd integer
        
        if *wpn* is even, 
            the *wpn* parameter is automatically set to *wpn* + 1,
            a UserWarning is raised

    Returns
    -------
    ys/wpn : numpy.array
        array containing smoothened data of *y*
    '''

    if wpn < 1:
        wpn = 1
        warnings.warn("wpn must be an odd positive number" +
                      "(wpn set to %d)" % (wpn),
                      UserWarning)
    if wpn % 2 == 0:
        wpn += 1
        warnings.warn("wpn must be an odd positive number" +
                      " (wpn set to %d)" % (wpn),
                      UserWarning)
    yaux = np.concatenate((np.ones(int((wpn-1)/2))*I_wn[0], np.asarray(copy.copy(I_wn)),
                           np.ones(int((wpn-1)/2))*I_wn[-1]))
    leny = len(I_wn)
    ys = np.zeros(len(I_wn))
    for i1 in range(wpn):
        ys += yaux[i1:i1+leny]

    return ys/wpn



