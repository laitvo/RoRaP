#!/usr/bin/python
# -*- coding: utf-8 -*-

#assign.py
#Vojta Laitl in DIFFER, 2023
#reproduced from PuPlasAn (https://github.com/laitvo/PuPlasAn, Petr Kubelík and Vojta Laitl in the Czech Academy of Sciences, 2022)
#this work is shared under an Apache 2.0 license code: https://github.com/laitvo/PuPlasAn/blob/main/LICENSE
#please reach out for a citation note before publishing results of this code


'''
Assignment module
--------------------------

This module is responsible for mutual assignment between experimental and theoretically predicted indices in data arrays (e.g., in those comprised by spectral records). Iterative minimisation is employed.
'''


def assign(px1, px2, dx_max):
    '''Assigns the closest values form *px1* and *px2*.

    Parameters
    ----------
    px1: list or numpy.arrays containing floats, 1-D each
        list containing experimental peak positions
    px2 : list of numpy.arrays containing floats, 1-D each
        list containing theoretical peak positions

    Returns
    -------
    inds : list of tuples
        list of tuples with assigned indexes corresponding to the values
        *inds* correspond to indices of *px1* and *px2* (e.g., [(5,9), (6,12)...])
    '''

    inds = []

    for i1, px1i in enumerate(px1):

        select_aux = []

        for i2, px2i in enumerate(px2):
            d = abs(px1i - px2i) 

            if d < dx_max:
                select_aux.append((i1, i2, d))

        if len(select_aux) > 0:

            select_aux.sort(key=lambda x: x[2])
            inds.append((select_aux[0][0], select_aux[0][1]))

    return inds
