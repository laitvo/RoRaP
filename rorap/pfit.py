#!/usr/bin/python
# -*- coding: utf-8 -*-

#bline.py
#Vojta Laitl in DIFFER, 2023
#reproduced from PuPlasAn (https://github.com/laitvo/PuPlasAn, Petr Kubelík and Vojta Laitl in the Czech Academy of Sciences, 2022)
#this work is shared under an Apache 2.0 license code: https://github.com/laitvo/PuPlasAn/blob/main/LICENSE
#please reach out for a citation note before publishing results of this code


import numpy as np
from scipy.special import wofz

'''
Peak profile fitting module
---------------------------

This module contains functions for peak profile fitting. The peak fitting
is performed by *rsopt* function which is based on random rearch algorithm
and is able to fit a model composed by multiple (possibly overlapping)
peaks represented by an instance of the :class:`rorap.pfit.PeakModel` class.
This function is applicable also for single peak fitting.
'''

def rsopt(obfun, bounds, msteps=50, fsteps=100, tsteps=500, focus=0.5,
          ftol=[3, 1.e-5], obfun_args=(), obfun_kwargs={}, callback=None,
          callback_args=(), callback_kwargs={}):
    '''Casts a randomised algorithm onto peak profile searching

    Parameters
    ----------
    obfun : instance
        instance resulting from the objective function to be minimised onto a peak profile
    bounds : list
        list containing respectively [*lower*, ⋅] and [⋅, *upper*] bounds to a peak profile
    msteps : int
        the number of minimisation steps to be taken within an iteration
        by default, *msteps* equals *50*
    fsteps : int
        the number of steps taken to focus the searching algorithm
        by default, *fsteps* equals *100*
    tsteps : int
        the number of testing/trial steps to the algorithm
        *tsteps* must be > *fsteps*
        
        if *tsteps* < *fsteps*,
            a UserException is raised

        by default, *tsteps* equals *500*

    ftol : list of type [int, float]
        list containing respectively [*initial maximum iterations*, ⋅] and [⋅, *tolerance threshold*] characteristics
        by default, *ftol* equals *[3, 1e-05]*
    obfun_args : tuple
        additional arguments to the *obfun* objective function
        by default, *obfun_args* equals *()*
    obfun_kwargs : dict
        additional keyword arguments to the *obfun* objective function
        by default, *obfun_kwargs* equals *{}*
    callback : instance
        arbitrary callback function
        by default, callback is *NoneType*'d
    callback_args : tuple
        additional arguments to the *callback* function
        by default, *callback_args* equals *()*
    callback_kwargs : dict
        additional keyword arguments to the *callback* function
        by default, *callback_kwargs* equals *{}*

    Returns
    -------
    best_pars : numpy.array
        array containing the best fitting peak parameters
    fold : float
        converge threshold achieved
    '''

    if fsteps >= tsteps:
        raise Exception('tsteps must be greater than fsteps ' +
                        '(current values: fsteps = %d, ' +
                        'tsteps = %d)' % (fsteps, tsteps))
    iter_bounds = bounds[:]
    mcounter = 0
    fcounter = 0
    tcounter = 0
    ftolcounter = 0
    fold = float('inf')
    while tcounter <= tsteps:
        mcounter += 1
        fcounter += 1
        tcounter += 1
        pars = []
        for (ibmin, ibmax), (bmin, bmax) in zip(iter_bounds, bounds):
            maxx = min(bmax, ibmax)
            minx = max(bmin, ibmin)
            #(b - a) * np.random.random_sample() + a; b>a 
            pars.append((maxx - minx) * np.random.random_sample() + minx)
        fnew = obfun(pars, *obfun_args, **obfun_kwargs)
        if fnew < fold:
            if abs(fnew - fold) < ftol[1]:
                ftolcounter += 1
            fold = fnew
            best_pars = pars[:]
            fcounter = 0
            tcounter = 0
            mcounter = 0
            if callback:
                callback(best_pars, *callback_args)
            if ftolcounter > ftol[0]:
                break #convergence
        if mcounter >= msteps:#move
            iter_bounds_new = []
            for (ibmin, ibmax), (bmin, bmax), bp in zip(iter_bounds, bounds, best_pars):
                r = ((ibmax - ibmin))/2.
                maxb = min(bmax, bp + r)
                minb = max(bmin, bp - r)
                iter_bounds_new.append((minb, maxb))
            iter_bounds = iter_bounds_new[:]
            mcounter = 0
        if fcounter >= fsteps:#focus
            iter_bounds_new = []
            for (ibmin, ibmax), (bmin, bmax), bp in zip(iter_bounds, bounds, best_pars):
                r = ((ibmax - ibmin)*focus)/2.
                maxb = min(bmax, bp + r)
                minb = max(bmin, bp - r)
                iter_bounds_new.append((minb, maxb))
            iter_bounds = iter_bounds_new[:]
            fcounter = 0
    return best_pars, fold


def area_estim(height, fwhm):
    '''Estimates the peak area by a rectangular function (i.e., by multiplying the peak height by its FWHM).

    Parameters
    ----------
    height : float OR numpy.array
        peak's high
    fwhm : float OR numpy.array
        peak's FWHN
        if numpy.array, the length of *fwhm* must correspond to that of *height*
        

    Returns
    -------
    area : float OR numpy.array
        peak area estimated as *height* \* *fwhm*
    '''

    area = height*fwhm

    return area


def detect_peak_groups(px, d):
    '''Generates groups of peaks which are so close each to other that they
    can be overlapped.

    Parameters
    ----------
    px : numpy.array
        array of the peak positions (obtained by a peak picking method)
    d : float OR callable

        if float, it is the distance threshold (i.e., peaks which are closer to each
        other are grouped)

        if callable, it is a function with one input
        parameter (the peak position as *float*) which returns the distance
        threshold expressed as a function of the independent variable
        corresponding to *px*

    Returns
    -------
    groups : list of lists of tuples
        list describing the grouping of the peaks accordind their mutual
        distances
        the format is as follows:

        [[(i0, v0), (i1, v1)...], [(in, vn), (in+1, vn+1)...]...], where each
        nested list represent a group of mutually close peaks and each tuple
        in these lists contains index of an element of the input parameter
        *px* and the corresponding value.
    '''


    px.sort()
    groups = [[(0, px[0])]]
    if hasattr(d, '__call__'):
        for i1, pxi in enumerate(px[1:], 1):
            #appends next peak to the last group
            if (pxi - groups[-1][-1][1]) < d(pxi):
                #creates a new group and insert the next peak
                groups[-1].append((i1, pxi))
            else:
                groups.append([(i1, pxi)])
    else:
        for i1, pxi in enumerate(px[1:], 1):
            #appends next peak to the last group
            if (pxi - groups[-1][-1][1]) < d:
                #creates a new group and insert the next peak
                groups[-1].append((i1, pxi))
            else:
                groups.append([(i1, pxi)])
    return groups


def gauss(x, mu, a, fwhm):
    '''Calculates the Gaussian profile function from the input parameters.

    Parameters
    ----------
    x : numpy.array
        array of independent variables to be fitted against
        *x* must be strictly increasing
    mu : float
        peak center position
    a : float
        peak area
    fwhm : float
        peak full width at half maximum
        *fwhm* must be formal wavenumbers in cm-1

    Returns
    -------
    G_x : numpy.array
        array containing the Gaussian peak profile
        corresponding to the independent variable *x*
    '''

    sigma = fwhm/(2.*np.sqrt(2.*np.log(2.)))
    G_x = ((1./(sigma * np.sqrt(2.*np.pi))) *
            np.exp(-0.5 * ((x - mu)/sigma)**2)) * a

    return G_x


def lorentz(x, mu, a, fwhm):
    '''Calculates the Lorentzian profile function from the input parameters.

    Parameters
    ----------
    x : numpy.array
        array of independent variables to be fitted against
        *x* must be strictly increasing
    mu : float
        peak center position
    a : float
        peak area
    fwhm : float
        peak full width at half maximum
        *fwhm* must be formal wavenumbers in cm-1

    Returns
    -------
    L_x : numpy.array
        array containing the Lorentzian peak profile
        corresponding to the independent variable *x*
    '''

    gamma = fwhm/2.
    L_x = ((1./np.pi) * ((0.5*gamma)/((x - mu)**2 + (0.5*gamma)**2))) * a

    return L_x


def voigt(x, mu, a, fwhm):
    '''Applies the Faddeeva function to estimate the Voigt profile function from the input parameters.

    Parameters
    ----------
    x : numpy.array
        array of independent variables to be fitted against
        *x* must be strictly increasing
    mu : float
        peak center position
    a : float
        peak area
    fwhm : float
        peak full width at half maximum
        *fwhm* must be formal wavenumbers in cm-1

    Returns
    -------
    V_x : numpy.array
        array containing the Voigt peak profile
        corresponding to the independent variable *x*
    '''

    sigma = fwhm/(2.*np.sqrt(2.*np.log(2.)))
    gamma = fwhm/2.

    V_x = a * np.real(wofz((x - mu + 1j*gamma)/sigma/np.sqrt(2.))) / sigma /np.sqrt(2.*np.pi)
    
    return V_x

def pseudovoigt(x, mu, a, fwhm, s=0.2):
    '''Calculates the pseudoVoigt (weighted sum of Gaussian and Lorentzian
       profiles) profile function from the input parameters.

    Parameters
    ----------
    x : numpy.array
        array of independent variables to be fitted against
        *x* must be strictly increasing
    mu : float
        peak center position
    a : float
        peak area
        by default, *a* equals *1.0* (normalised profile)
    fwhm : float
        peak full width at half maximum
        *fwhm* must be formal wavenumbers in cm-1
    s : float
        peak shape parameter
        values from the interval of [0,1] are only accepted as inputs
        if *s* equals *0.*, the profile is pure Gaussian
        if *s* equals *1.*, the profile is pure Lorentzian

    Returns
    -------
    P_x : numpy.array
        array containing the pseudoVoigt peak profile
        corresponding to the independent variable *x*
    '''


    g = gauss(x, mu, 1., fwhm)
    l = lorentz(x, mu, 1., fwhm)

    P_x = ((1. - s)*g + s*l) * a

    return P_x


class PeakModel:
    '''Provides methods to model single/multi peak profiles.'''

    profiles = {'gauss': gauss,
                'lorentz': lorentz,
                'voigt': voigt,
                'pseudovoigt': pseudovoigt}

    def __init__(self, mu, area, fwhm, pars=(), profile='voigt'):
        '''
        Parameters
        ----------
        mu : list OR tuple of floats
            positions of the centers of the peaks included in the profile model
        area : list OR tuple of floats
            areas of the peaks included in the profile model
            *area* must correspond to *mu*
        fwhm : list OR tuple of floats
            full widths in half maxima of the peaks included
            in the profile model
            *fwhm* must correspond to *mu*
        pars : list of lists OR tuples of floats
            other single peak profile parameters (each nested tuple
            corresponds to a specific parameter of all peaks of the model)
            used by the selected line profile function (e.g. the shape
            parameter used by the *pseudovoigt* function)
        profile : str
            a string corresponding to one of single peak profile functions
            ("gauss", "lorentz", "voigt", or "pseudovoigt")
            this profile is used for all peaks within the model
            by default, *profile* equals *voigt*
        '''

        self.profile = profile
        self.mu = mu
        self.area = area
        self.fwhm = fwhm
        self.pars = pars
        if any([len(mu) != len(area), len(mu) != len(fwhm)]):
            raise Exception('the length of mu, area and fwhm must be equal')
        self.pnum = len(mu)

    def get_flat_pars(self):
        '''Returns internal peak model parameters in a single flat list
        (used as an input partameter of the *calc_fit_error* method)

        Returns
        -------
        pars : list of floats
            model parameters in a single flat list with the following format:
            [par_i_peak_j, par_i+1_peak_j, par_i+2_peak_j...
            par_i_peak_j+1, par_i+1_peak_j+1... par_n_peak_m]
            where "par_i_peak_j" refers to the ith parameter
            of the jth peak of the profile model
        '''

        if self.pars:
            return ([el for tup in zip(self.mu, self.area, self.fwhm,
                                      *self.pars) for el in tup])
        else:
            return ([el for tup in zip(self.mu, self.area,
                                      self.fwhm) for el in tup])

    def set_flat_pars(self, pars):
        '''Sets the internal peak parameters of the profile model

        Parameters
        ----------
        pars : list of floats
            model parameters in a single flat list with the following format:
            [par_i_peak_j, par_i+1_peak_j, par_i+2_peak_j...
            par_i_peak_j+1, par_i+1_peak_j+1... par_n_peak_m]
            "par_i_peak_j" refers to the i-th parameter of the j-th peak of the profile model
            the peak number as well as the numper of the parameters of each peak 
            must be in agreement with the internal peak model
        '''

        parn = len(pars)/self.pnum
        aux = list(zip(*[pars[i:i+int(parn)] for i in range(0, len(pars), int(parn))]))
        self.mu = aux[0]
        self.area = aux[1]
        self.fwhm = aux[2]
        if len(aux) > 3:
            self.pars = aux[3:]
        else:
            self.pars = ()

    def gen_profile(self, x, fulloutput=False):
        '''Returns the profile of multiple (possibly overlapped) peaks.
        The resulting profile is calculated as a sum of several
        peak-like profiles.

        Parameters
        ----------
        x : numpy.array
            array of independent variables to be fitted against
            *x* must be strictly increasing
        fulloutput : bool
            boolean algebra object influencing the output (see the *Returns* section).
            by default, *fulloutput* equals *False*

        fulloutput : bool
            if True: returns array containing the resulting multi-peak profile
            as well as the profiles of the individual functions
            (e.g. gaussians)
            else: returns array containing the resulting multi-peak profile

        Returns
        -------
        If *fulloutput* is *True*, returns

        y : numpy.array
            array containing the resulting multi-peak profile
            corresponding to the independent variable *x*
        parts : list
            list of individuals peak profiles'


        Else, returns

        y : numpy.array
            array containing the resulting multi-peak profile
            corresponding to the independent variable *x*
        '''


        y = np.zeros(len(x))
        parts = []
        if self.pars:
            pars = zip(self.mu, self.area, self.fwhm, *self.pars)
        else:
            pars = zip(self.mu, self.area, self.fwhm)
        for parsi in pars:
            prof = PeakModel.profiles[self.profile](x, *parsi)
            y += prof
            if fulloutput:
                parts.append(prof)
        if fulloutput:
            return y, parts
        return y

    def calc_fit_error(self, pars_flat, datax=[], datay=[]):
        '''Calculates the objective function value used by
        a fitting function

        Parameters
        ----------
        pars_flat : list OR tuple of floats
            profile model parameters in the following format:
            [par_i_peak_j, par_i+1_peak_j, par_i+2_peak_j...
            par_i_peak_j+1, par_i+1_peak_j+1... par_n_peak_m]
            "par_i_peak_j" refers to the i-th parameter of the j-th peak of the profile model
            the peak number as well as the numper of the parameters of each peak 
            must be in agreement with the internal peak model 
            this list can be generated by *get_flat_pars* method 
        datax : numpy.array
            array of independent variable values
            *datax* must be strictly increasing
        datay : numpy.array
            array of dependent variable values
            *datay* must correspond to strictly increasing *datax*

        Returns
        -------
        err : float
            the difference between the model profile and the *datay* parameter,
            the value is calculated es follows:
            np.sqrt(sum((self.gen_profile(datax) - datay)**2))

        ::Warning::

           by calling this method, the internal model parameters are rewriten
           according to the input parameter *pars_flat*
        '''


        self.set_flat_pars(pars_flat)
        err = np.sqrt(sum((self.gen_profile(datax) - datay)**2)/
                       float(len(datax)))

        return err


def peak_model(px, py, spe, fwhm, pl_range= [200., 212., -0.07, 0.6], step=12., profile='voigt'):
    '''Builds a fit of experimental spectrum based on stochastic modeling of small areas' optima

    Parameters
    ----------
    px : numpy.array
        array of the peak positions (obtained by a peak picking method)
    py : numpy.array
        array containing integral intensities of the detected peaks
        *py* must correspond to *px*
    spe : numpy.array
        array containing the experimental spectrum as sorted onto two columns of
        [*energy*, *signal/intensity*]
        *energy* must be given in units consistent with *px*
    fwhm : float
        peak full width at half maximum
        *fwhm* must be formal wavenumbers in cm-1
    pl_range : list
        list defining the initial spectral area to search on and additional constants as defined above
        by default, *pl_range* equals *[200., 212., -0.07, 0.6]*
    step : float
        increment step to be taken by the fitting procedure
        *step* must be given in units consistent with *px*
        by default, *step* equals *12.*
    profile : str
        a string corresponding to one of single peak profile functions
        ("gauss", "lorentz", "voigt", or "pseudovoigt")
        this profile is used for all peaks within the model
        by default, *profile* equals *voigt*


    Returns
    -------
    
    fit_spect : numpy.array
        y-axis of the fitted/simulated spectrum
        fit_spect corresponds to the *energy* axis of *spe*
    '''      

    fwhm_estim = fwhm
    fit_spect = []

    while pl_range[1]<max(px):
    
        ind1 = np.where(spe[:,0] > pl_range[0])[0][0]
        ind2 = np.where(spe[:,0] > pl_range[1])[0][0]
        indg1 = np.where(px > pl_range[0])[0][0]
        indg2 = np.where(px > pl_range[1])[0][0]
        groups = detect_peak_groups(px[indg1:indg2], 0.05)
        fit_spect0 = np.zeros(len(spe[ind1:ind2,0]))
            
        for gi in groups:
            
            ind1_fit = np.where(spe[:,0] > gi[0][1] - 0.1)[0][0]
            ind2_fit = np.where(spe[:,0] > gi[-1][1] + 0.1)[0][0]
            bounds = []
            mu = []
            area = []
            fwhm = []
                
            for pki in gi:
                
                a_estim = area_estim(py[pki[0]], fwhm_estim)
                mu.append(pki[1])
                area.append(a_estim)
                fwhm.append(fwhm_estim)
                bounds.extend([(pki[1] - 0.02, pki[1] + 0.02),
                    (a_estim/10., a_estim*10.),
                    (fwhm_estim/5., fwhm_estim*3.)])
            peak = PeakModel(mu, area, fwhm, pars=(), profile=profile)
            best_pars, fold = rsopt(peak.calc_fit_error, bounds, msteps=20, fsteps=500,
                            tsteps=5000, focus=0.6, ftol=[3, 1.e-6],
                            obfun_args=(spe[ind1_fit:ind2_fit,0],
                                        spe[ind1_fit:ind2_fit,1]),
                            obfun_kwargs={}, callback=None, callback_args=(),
                            callback_kwargs={})
            fit_spect0 += peak.gen_profile(spe[ind1:ind2,0])
                
        fit_spect = np.insert(fit_spect,len(fit_spect),fit_spect0)
        pl_range[0] += step
        pl_range[1] += step

        try:
            
            fit_spect = np.insert(fit_spect,len(fit_spect),np.zeros(len(spe[:,0])-len(fit_spect)))

        except:
        
            fit_spect = fit_spect[:len(spe[:,0])]        

    return fit_spect
