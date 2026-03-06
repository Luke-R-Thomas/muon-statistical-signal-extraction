#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 19:35:51 2026

@author: lukethomas
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq

# ====================================VALUE CHECK FUNCTION=========================================


def value_check(variable,value,min_limit,max_limit):
    """
    Function to check whether the tested variable is within the expected range given, 
    returns error messages if None or is not within the range 
    
    Parameters
    ----------
    variable: 
        The name of the data value being tested
    value:
        The calcuated value associated with the variable
    min_limit:
        The given min value 
    max_limit:
        The given max value
    
    Returns
    -------
    ValueError:
        If variable value is None
    ValueError:
        If the variable value is not within the expected range 
        
    Notes
    -----
    This function is used to check and output the given error, not to raise the error itself 
    """
    
    
    if value is None:
        return ValueError(f"{variable} is None")
    
    
    valid = min_limit <= value <= max_limit
    if not valid:
    
        return ValueError(f"{variable} = {value} is not within the expected range of {min_limit} and {max_limit}")  
    
# ====================================FITTING GUESS FUNCTIONS=========================================


def B0_calculation_FFT(A,centres,B_min = 0.1e-5,B_max = 0.03, gamma = 851.616e6):
    """
    Estimates an initial magnetic field guess B0 using an FFT of the early-time asymmetry
    
    - Uses the first initial_data_limit (4µs) of data which is less uncertian, removes the mean and perfromes FFT. 
    - FFT allows to find max frequency within the expected frequency ranges and convert to the inital B0 guess, using B0 = 2 * np.pi * f_peak / gamma

    
    Parameters
    ----------
    A: 
        Asymmetry per bin
    centres: 
        Array of bin centre times for each bin edge    
    B_min, B_max: 
        Given range of magnetic field (T) used to restrict frequency
    gamma:
        Gyromagnetic ratio for a positive muon (rad s^-1 T^-1) 
    
   
    Returns
    -------
    B0: 
        Initial guess for magnetic field (T)
    
    Notes
    -----
    FFT is perfromed on A(t) data in the early time period as uncertianties grow larger with time
    
    Frequency is restricted using the limits ensuring that the only real values are returned, 
    if B0 is not found within the range it reutrns an error
    """
   
    T = 2 * np.pi / (gamma * B_max)
    bin_width = T / 20

    # restricts the data to the first 4µs
    initial_data_limit = 4
    
    idx = np.abs(centres - initial_data_limit).argmin()
    A_initial = A[:idx+1]    
    
    # subtracts the mean from A(t)
    A_mean = np.mean(A_initial)
    x = A_initial - A_mean
    
    # calculates the upper and lower frequency limit
    f_min = (gamma * B_min) / (2*np.pi)
    f_max = (gamma * B_max) / (2*np.pi)
    
    # FFT to find frequency
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=bin_width)
    mag = np.abs(X)
    mag[0] = 0
    
    # applies the frequency restriction
    frequency_restriction = (freqs >= f_min) & (freqs <= f_max)
    
    idx_peak = np.argmax(mag[frequency_restriction])
    f_peak = freqs[frequency_restriction][idx_peak]
    
    # convert to B0
    B0 = 2 * np.pi * f_peak / gamma
    
    return B0


def amplitude_estimate(A,centres,B_max = 0.03, gamma = 851.616):
    
    """
    Estimated the initial oscillation amplitde using an intial time frame (2µs)
    - B0 from B0_calculation_FFT used to calculate period and samples_per_period of data
    - Guassian fucntion smooths data
    - Find peas used to calcuate the peak and troughs, first peak manually set to t(0) and corresponding next trough found
    - amplitude calcuted using: (peak - trough)/2.
    
    Parameters
    ----------
    A: 
        Asymmetry per bin
    centres: 
        Array of bin centre times for each bin edge
    B_max: 
        Given upper limit of magnetic field (T) 
    gamma:
         Gyromagnetic ratio for a positive muon (rad s^-1 T^-1) 
    
    Returns
    -------
    amplitude:
        Amplitude calcuated using the first peak and trough 
        
    Notes
    -----
    Assumes centres is in µs and so gamma is taken to be 851.616, powers cancel 
    
    Find peaks fails to identfity edges of data as peaks, in this case the data begins at the first peak and decays then on
    the first data must manually be set as a peak otherwise the peak after has already had decay, 
    leading to a underestimation in amplitude
       
    """

    # recalculates bin width used in centres 
    T = 2 * np.pi / (gamma * B_max)
    bin_width = T / 20

    # calculates T period of wave using predicted B0
    B0 = B0_calculation_FFT(A,centres)
    T0 = 2 * np.pi / (gamma * B0)

    # restricts the data to the first 2µs
    initial_data_limit = centres <= 2
    y0 = A[initial_data_limit]
    
    # smooths asymmetry data using guassian filter 
    y0 = gaussian_filter1d(y0, sigma=2)
    
    # calculates peak_distance for find_peaks using samples_per_period (allowing 0.35 per T)
    samples_per_period = T0 / bin_width
    peak_distance = 0.35 * samples_per_period 
  
    
    # calculates peak_prom the required displacemnt from the range of data to count as a peak (10%)
    asymmetry_range = np.max(y0) - np.min(y0)
    peak_prom = 0.1 * asymmetry_range  
    
    # calculates all peaks and troughs within the early time period data
    peaks, _ = find_peaks(y0, prominence=peak_prom,distance=peak_distance)
    troughs, _ = find_peaks(-y0, prominence=peak_prom,distance=peak_distance)
    
    
    # if the frist position on the data is > the next two then its set by defualt as the first peak 
    if (y0[0] > y0[1]) and (y0[0] > y0[2]):
        p_idx = 0
    else:
        p_idx = peaks[0]    
        
    # finds the next trough after 
    after_trough = troughs[troughs > p_idx]
    t_idx = after_trough[0]  
    
    # asymmetry values of peak and trough using location idx
    peak0 = y0[p_idx]
    trough0 = y0[t_idx]
    
    # amplitude calculation
    amplitude = (peak0 - trough0) / 2
    
    return amplitude


def beta0_calculation_brentq(amplitude,beta_min = 0.5,beta_max = 1.5):
    
    """
    Calculates an inital beta0 guess using the estimated amplitude calculated in amplitude_estimate
   
    Uses brentq root finder, beta0 is found in the range given between beta_min and beta_max
   
    Parameters
    ----------
    amplitude: 
        Initial amplitude calcuated using the first peak and trough 
        
    beta_min, beta_max: 
         Given range of detector angle (rads) used to restrict beta0
        
    Returns
    -------
    beta0: 
        Initial guess for beta (radians)
    """
    
    # function f (sin(beta) / beta = 3A) that can be called when solving usining brentq
    def f(beta):
        return (np.sin(beta) / beta) - 3*amplitude
    
    
    beta0 = brentq(f,beta_min,beta_max)
    
    return beta0


def P0_calculation(A,centres): 
    """
    Estimate initial fit parameters (P0) in an array, used for curve fitting

    Combines:
    - B0 from FFT peak detection,
    - beta0 from amplitude found from early-time peak/trough
    - tau0 a fixed initial damping consntant 
       
    Parameters
    ----------
    A:
        Asymmetry values per bin.
    centres: 
        Array of bin centre times for each bin edge

    
    Returns
    -------
    P0 : list
        Initial guesses [B0, beta0, tau0] for the damped asymmetry fit.
    
    
    Returns
    -------
    P0: 
        Initial guess paramaters [B0, beta0, tau0], where B0 and beta0 are calucted and tau0 is a cosntant
    
    Notes
    -----
    FFT is perfromed on A(t) data in the early time period as uncertianties grow larger with time

    Frequency is restricted using the limits ensuring that the only real values are returned, 
    if B0 is not found within the range it reutrns an error

    beta0 is found from the early time period amplitude using bakcground physics (sin(beta) / beta = 3A)
    
    Solved using scipy brentq in the given region of (0.5 - 1.5 rads)
    
    tau0 can be a constant at the midpoint of range of limits (6e-6µs) as has little affect on position of curve fit
     
     """

    B0 = B0_calculation_FFT(A,centres)

    amplitude = amplitude_estimate(A,centres)

    beta0 = beta0_calculation_brentq(amplitude)
        
    tau0 = 6e-6
    
    raised_errors = []

    error = value_check('B0',B0,0.5e-3,30e-3)
    if error:
        raised_errors.append(error)

    
    error = value_check('beta',beta0,0.5,1.5)
    if error:
        raised_errors.append(error)
        
    error = value_check('10keV_tau_damp',tau0,2e-6,10e-6)
    if error:
        raised_errors.append(error)
            
    
    if raised_errors:
        raise ValueError(f"Guess function value check failed: {raised_errors}")


    # final P0 array    
    P0 = [B0,beta0,tau0]
    
    return P0
 
    
