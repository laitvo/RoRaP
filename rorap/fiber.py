#!/usr/bin/python
# -*- coding: utf-8 -*-

#raman.py
#Vojta Laitl in DIFFER, 2023


'''
Fiber calibrating module

Functions contained here are responsible for reading out the fiber bundle's PI-MAX image and converting it onto experimental Raman spectra.
'''
#requires having `spe2py' installed (https://pypi.org/project/spe2py/); `pip install spe2py'

import numpy as np
import pylab as pl
pl.ion()
import spe2py as sp
import spe_loader as sload

def fiber_load(experiment):
    '''Loads experimental spectra from specified files and subtracts experimental backgroud.

    Parameters
    ----------
    experiment : list of strings
        list of strings describing (relative) paths to experimental data files
        *experiment* must point onto possibly replicated *.SPE* files
        the last string of *experiment* must be experimental background file

    Returns
    -------
    data : numpy.array
        array containing the raw experimental image given by averaging any replicas and subtracting experimental background

    '''
    
    files = sload.load_from_files(experiment)
    data = files[0].data[0][0]
    
    for i in range(1,len(files)-1):
        data += files[i].data[0][0]
        
    data = data/(len(files)-1) - files[-1].data[0][0]            
    
    return data
    

def fiber_spectra(data, pos_0, lambda_0, fiber_pos, binning, dispersion, cutoff_pixels):
    '''Estimates cutoff wavelengths of a fiber bundle and draws experimental spectra for each fiber.
    
    Parameters
    ----------
    data : numpy.array
        array containing the raw experimental image given by averaging any replicas and subtracting experimental background
    pos_0 : <function __main__.<lambda>(y)>
        horizontal position of fibers collecting a reference peak of *lambda_0* wavelength
        pos_0 must be a single-argument function of the vertical position
    lambda_0 : float
        reference peak wavelength recorded at *pos_0*(y)
        *lambda_0* must be wavelength in nanometres
    fiber_pos : list of lists
        list containing vertical coordinate limits of all fibers and their horizontal cutoff pixels for an unbinned sensor
        an example format reads *[[[0,17],384],[[18,37],393],[[...,...],...], ...]* = *[[[v11,v12],h1],[[v21,v22],h2],[[...,...],...], ...]*
        the horizontal cutoff must correspond to the same reference image as *pos_0*(y)
    dispersion : float
        linear dispersion of the system
        *dispersion* must be given in nm/pixel
    cutoff_pixels : int
        cutoff distance of the central fibre's spectrum
        *cutoff_pixels* are given from the fiber's edge to the first point of Raman signal observation

    Returns
    -------
    fiber_lambda : numpy.array
        array containing the cutoff wavelengths of the bundle
        *fiber_lambda* are given in nanometres
    fiber_spectra : list of numpy.array
        list of 2-D numpy.arrays containing individual fibers' spectra
        each spectrum is given in a row format of [*Raman shift (cm-1)*, *signal (arb. u.)*]
    '''
    
    fiber_lambda = []
    fiber_spectra = []
    
    ref_fiber = int(len(fiber_pos)/2.)
    ref_pos = fiber_pos[ref_fiber][1]
    ref_spe = np.sum(data[int(fiber_pos[ref_fiber][0][0]/binning[1]):int(fiber_pos[ref_fiber][0][1]/binning[1])],axis=0)
    
    for i in range(len(fiber_pos)):
        
        mean_pos = int(np.mean(fiber_pos[i][0]))
        lambda_i = lambda_0 + (fiber_pos[i][1]-pos_0(mean_pos))*dispersion
        fiber_lambda = np.append(fiber_lambda, lambda_i)
        pixels = cutoff_pixels+int((fiber_pos[i][1]-ref_pos)/binning[0])
        wl = lambda_i + (dispersion/binning[0])*(np.arange(len(data[0]))-pixels)[pixels:]
        wn = (1e07/lambda_0)-(1e07/wl)
        sig = np.sum(data[int(fiber_pos[i][0][0]/binning[1]):int(fiber_pos[i][0][1]/binning[1])],axis=0)[pixels:]
        spectrum = np.array([wn,sig]).T
        fiber_spectra.append(spectrum)
    
    return fiber_lambda, fiber_spectra
