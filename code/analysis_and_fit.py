#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 19:27:50 2026

@author: lukethomas
"""
import numpy as np
from scipy.optimize import curve_fit

from file_formatting_and_preprocess import energy_channel_separation,bin_analysis
from guess_parameters import P0_calculation

# ====================================CURVE FIT FUNCTIONS========================================

def damped_asymmetry_function(time_us,B,beta,tau_damp,gamma = 851.616E6):  
    """
    Damped asymetry function f(x,B,beta,tau,gama)
    
    Parameters
    ----------
    time_us: 
        Time values in microseconds (µs)
    B: 
        Magnetic field (T)
    beta: 
        Detector angle (rads)
    tau_damp:
        Damping time constant (s)
    gamma:
        Gyromagnetic ratio for a positive muon (rad s^-1 T^-1) - Default is 851.616e6
        
    
    Returns
    -------
    A : 
        Model asymmetry for given time
    
    Notes
    -----
    - Assumes time_us is in µs and converts to seconds 
    """

    # Converts time from micro to seconds 
    time_s = time_us * 1e-6
    
    calc1 = (np.cos(gamma * B * time_s) * np.sin(beta)) / beta
    calc2 = np.exp(-time_s/tau_damp)
    
    A = (1/3) * calc1 * calc2
    
    return A


def B_field_function(energy,a,b,c):
    """
    B field function f(x,energy,a,b,b)
    Function that represents how the magnetic field chnages quadratically with energy
    
    Parameters
    ----------
    energy: 
        Time values in microseconds (µs)
    a,b,c:
        Coeefficents of quadratic
    
    Returns
    -------
    B: 
        Magnetic Field for given input energy (T)

    """
    
    B = (a*energy**2) + (b * energy) + c
    
    return B

# ====================================CALCULATION FUNCTIONS======================================




def asymmetry_calculation(left_array,right_array,bin_edges):
    """
    Calculates asymmetry A(t) and its uncertainty A_unc from the left and right time data

    Parameters
    ----------
    left_arr:
        Array of time values (µs) at a given energy, incident on the left detector
    right_arr:
        Array of time values (µs) at a given energy, incident on the right detector
    bin_edges:
        Histogram bin edges calculated from bin_analysis function
    
    
    Returns
    -------
    A_fit:
        Asymetry per bin, calculated using A = (L - R) / (L + R)
    A_unc:
        Uncertainty in A per bin, calculated using 2*sqrt( (L * R) / (L + R)^3 )
    counts_left:
        Histogram counts per bin for the left data
    counts_right: 
        Histogram counts per bin for the right data
    
    Notes
    -----
    Histogram bin edges calculated using the full combined data:
        - Same bin edges for both data sets
        - Final bins can extend beyond left or right data, can give bins with one side empty
        - Results in A +/- 1 and A_unc = 0
        - Bins with these conditions are therfore filtered / dropped later
    
    np.divide ensures any /0 errors are replaced with nan type: prevent calcualtion errors
    """

    # histogram count data per bin

    counts_left, _ = np.histogram(left_array, bins=bin_edges)
    counts_right, _ = np.histogram(right_array, bins=bin_edges)

    # asymetry calculation

    counts_total = counts_left + counts_right
    counts_product = counts_left * counts_right

    A_fit = np.divide((counts_left - counts_right), counts_total, out=np.full_like(counts_total,np.nan,dtype = float), where=counts_total !=0 )

    # asymetry uncertianty calculation

    calc1 = np.divide(counts_product, (counts_total**3),out=np.full_like(counts_product,np.nan,dtype = float), where=counts_total !=0 )
    
    A_unc = 2 * np.sqrt(calc1)
    
    return A_fit , A_unc, counts_left, counts_right


def B_field_calculation(energy_data):
    """
    Using a single energrys data, fits a damped asymmetry model to evaluate B, beta and tau values 
    Constants computed using a range of functions:
    - Splits the data into L / R data sets of detection times
    - Calcualtes bin edges, centres and A(t) / A_unc(t)
    - Applies a filter where A or A_unc are Nan or where A_unc = 0; also applied to centres for similarity 
      (can arrise from the asymmetry of left andf right counts data and applying the same bin edges array)
    - Calculates an intial P0 paramater array, of [B0,beta0,tua0]
    - Applies a curve fit using damped_asymmetry_function to extract values, accounts for uncertianty weighting in A
    
    Parameters
    ----------
    energy_data:
        Unsorted energy data of times (µs) for both left and right detector at a given energy

    Returns
    -------
    B_fit : 
        Fitted magnetic field (T)
    B_err: 
        Uncertainty in B_fit 
    beta_fit: 
        Fitted detector angle (rads)
    beta_err: 
        Uncertainty in beta_fit
    tau_damp_fit: 
        Fitted damping time constant (s)
    tau_damp_err: 
        Uncertainty in tau_damp_fit
        
    Notes
    -----
    Uncertainties come from the covariance matrix
    
    Filtering ensure errors that can arrise from the asymmetry of L / R right counts data and applying the same bin edges array
    
    Initially tried flooring A_unc = 0 values to the min unc - was dragging B_fit down

    """
    
    # cleaning data and A(t)/A_unc(t) calculations
    left_arr, right_arr = energy_channel_separation(energy_data)
    
    energy_bin_edges, centres = bin_analysis(energy_data)

    A_energy, A_unc, _,_ = asymmetry_calculation(left_arr,right_arr,energy_bin_edges)

    # filtering out values where:
          
        # A = NaN when counts_total == 0 (empty bin)
        # A_unc = NaN when counts_total == 0
        # A_unc = 0 when counts_product == 0 but counts_total > 0 
  
    filter_condition = np.isfinite(A_energy) & np.isfinite(A_unc) & (A_unc > 0)
    
    filtered_A_energy = A_energy[filter_condition]
    filtered_A_unc = A_unc[filter_condition]
    filtered_centres = centres[filter_condition]

    # calculates P0 guess arr

    p0_arr = P0_calculation(filtered_A_energy,filtered_centres)
    
    # applies a curve fit to damped_asymmetry_function 

    popt, pcov = curve_fit(damped_asymmetry_function, filtered_centres, filtered_A_energy, p0=p0_arr,sigma=filtered_A_unc,absolute_sigma=True,bounds=([0.0, 0.5,2.0e-6], [0.03, 1.5,10.0e-6]))
    
    # sets values from popt and uncertainites from covariance matrix 
    B_fit, beta_fit, tau_damp_fit = popt
    B_err, beta_err, tau_damp_err = np.sqrt(np.diag(pcov))
    
    return B_fit, B_err, beta_fit, beta_err, tau_damp_fit, tau_damp_err

def field_profile_coeff_calculation(energies_arr,B_energy_arr,B_energy_unc_arr):
    """
    From B(E) and uncertaintes for energies in energies_arr, fits a quadratic (B(E) = aE^2 + bE + c) to extraxt coefficients
    
    Parameters
    ----------   
    energies_arr: 
        Energies (keV), set to [5, 10, 15, 20, 25]
    B_energy_arr: 
        B values for each energy 
    B_energy_unc_arr: 
        Uncertainties of B for each energy 

    Returns
    -------
    coeffs: 
        A tuple of the quadratic coefficients (a, b, c) in B(E) = aE^2 + bE + c
    coeff_errs: 
        A tuple of uncertainties of coefficients (a_err, b_err, c_err) 
        
    """
    
    
    # applies a curve fit to B_field_function 

    popt, pcov = curve_fit(B_field_function, energies_arr, B_energy_arr, sigma=B_energy_unc_arr,absolute_sigma=True)

    # sets values from popt and uncertainites from covariance matrix 
    a, b, c = popt
    a_err, b_err,c_err = np.sqrt(np.diag(pcov))
    
    return (a, b, c),(a_err, b_err,c_err)

