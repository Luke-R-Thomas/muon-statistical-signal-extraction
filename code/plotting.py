#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:50:08 2026

@author: lukethomas
"""


import numpy as np
import matplotlib.pyplot as plt
from analysis_and_fit import damped_asymmetry_function, B_field_function  

# =================================PLOTTING FUNCTIONS============================================

def histogram_plots(left_arr, right_arr,bin_edges):
    """
    Plot left and right time histograms using the same bin edges
    
    Parameters
    ----------
    left_arr, right_arr: 
        Event times (µs) for left and right detectors, converted into counts
    bin_edges: 
        Histogram bin edges 
    
    """
   
    # Left histogram
    plt.figure()
    plt.hist(left_arr, bins=bin_edges, color = 'k')
    plt.title("Left Channel Histogram")
    plt.xlabel("Time (μs)")
    plt.ylabel("Counts")
    plt.xlim(left=0)
    plt.show()
    
    # Right histogram
    plt.figure()
    plt.hist(right_arr, bins=bin_edges,color = 'k' )
    plt.title("Right Channel Histogram")
    plt.xlabel("Time (μs)")
    plt.ylabel("Counts")
    plt.xlim(left=0)
    plt.show()
    
    return  None
   

def asymmetry_plot(A_fit,A_unc,B_fit,B_err,beta_fit,beta_err,tau_fit,tau_err,centres):

    """
    Plot asymmetry A(t) with error bars and the fitted damped model curve over it
    
    Parameters
    ----------
    A_fit: 
        Asymmetry values 
    A_unc: 
        Uncertainties of A_fit 
    centres: 
        Bin-centre times (µs)
    B_fit, beta_fit, tau_damp_fit: 
        Calculated values for damped_asymmetry_function
    
    """
    
    stats_asymmetry = asymmetry_rounded_stats(B_fit,B_err,beta_fit,beta_err,tau_fit,tau_err)
        
    time_fit = np.linspace(centres.min(), centres.max(), 100)

    fit_curve = damped_asymmetry_function(time_fit,B_fit,beta_fit,tau_fit)

 
    # asymmetry Plot with model fit curve using extrapolated B and beta 
    plt.figure()
    
    plt.errorbar(centres, A_fit, yerr=A_unc, fmt='ko' ,ecolor='k', ms=2, capsize=2,label = 'Fitted Data')
    
    plt.plot(time_fit,fit_curve,color = '#d80073',lw='2',label = 'Measured Asymmetry')
  
    plt.xlim(left=0,right = 10)

    plt.title("Measured and Fitted Asymmetry 10KeV")
    plt.ylabel("Asymmetry Ratio")
    plt.xlabel("Time (μs)")
    
    # generates the text box for model curve data 
    bbox = dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1,alpha=0.11) 
    plt.text(0.03, 0.05, stats_asymmetry, transform=plt.gca().transAxes,fontsize=10, bbox=bbox)
    
    
    plt.legend()
    plt.show()
    
    return None


def quadratics_plot(B_energy_arr,B_energy_unc_arr,a,a_err,b,b_err,c,c_err):
    
    """
    Plots magnetic field (B) against energy with error bars and the model quadratic fit B(E)=aE^2+bE+c
    
    Parameters
    ----------
    B_energy_arr: 
        Calculated B values for each energy
    B_energy_unc_arr: 
        Uncertainties of each B value 
    a, b, c: 
        Quadratic coefficients

    """
    stats_coeff = coeff_rounded_stats(a,a_err,b,b_err,c,c_err)
    
    energy_arr = [5,10,15,20,25]
        
       
    x_arr = np.linspace(5, 25,100)

    y_arr = B_field_function(x_arr,a,b,c)


    # Field Profile with model fit curve using extrapolated a,b and c
    plt.figure()
    
    plt.errorbar(energy_arr, B_energy_arr, yerr=B_energy_unc_arr, fmt='ko' ,ecolor='k', ms=5, capsize=2,label = 'Measured B-Field')
    plt.xticks(energy_arr)    
    plt.xlim(min(energy_arr), max(energy_arr))
    
    plt.plot(x_arr,y_arr,color = '#d80073',label = 'Fitted Field')
    plt.title("Field Profile with Energy")
    plt.ylabel("Magnetic Field (B)")
    plt.xlabel("Implantation energy (KeV)")
    
    # generates the text box for coeff data 
    bbox = dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1,alpha=0.11)

    plt.text(0.97, 0.05, stats_coeff, transform=plt.gca().transAxes,fontsize=10, bbox=bbox,ha='right',multialignment="left")
    
    
    plt.legend()
    plt.show()
    
    return None

# =================================ROUNDING FUNCTIONS============================================


def rounder(value,uncertainty):
    """
    Rounds a value calcuted and its uncertianty in format for plotting
    
    Parameters
    ----------
    value: 
        The value of the data variable being rounded

    uncertainty:
        The associated uncertainty in value
    
    Returns
    -------
    rounded:
        The rounded value
    rounded_err:
        The rounded values uncertainty to 1dp
        
    """
    
    # rounds unc to 1sf 
    
    rounded_err = str(f'{uncertainty:.1g}')
    
    split = rounded_err.split('.')
    
    rounded_err = float(f'{uncertainty:.1g}')
    
    
    if len(split) == 1:
        idx = 0
    
    else:
        idx = len(split[1])
    
    # rounds value to the same dp as the unc
    
    rounded = float(f'{value:.{idx}f}')

    
    return rounded,rounded_err

def coeff_rounded_stats(a,a_err,b,b_err,c,c_err):
    """
    Formats the data for the Field Profile with Energy plot text box 
    
    Parameters
    ----------
    coeffs: 
        A tuple of the quadratic coefficients (a, b, c) in B(E) = aE^2 + bE + c
    coeff_errs: 
        A tuple of uncertainties of coefficients (a_err, b_err, c_err) 

    Returns
    -------
    stats:
        coeff stats, in Matplotlib format

    """
    # turns value into µ
    a *= 1e6
    a_err *= 1e6
    b *= 1e6
    b_err *= 1e6
    c *= 1e6
    c_err *= 1e6

    a_r, a_err_r = rounder(a, a_err)
    b_r,b_err_r = rounder(b,b_err)
    c_r,c_err_r = rounder(c,c_err)
    
    stats = (
        rf"$a = {a_r}\ \pm\ {a_err_r}\ \mu\,\mathrm{{T\,keV^{{-2}}}}$" "\n"
        rf"$b = {b_r}\ \pm\ {b_err_r}\ \mu\,\mathrm{{T\,keV^{{-1}}}}$" "\n"
        rf"$c = {c_r}\ \pm\ {c_err_r}\ \mu\,\mathrm{{T}}$")

    return stats


def asymmetry_rounded_stats(B,B_err,beta,beta_err,tau,tau_err):  
    """
    Formats the data for the Measured and Fitted Asymmerty plot text box 
    
    Parameters
    ----------
    values: 
        B, beta and tau values used to generate the model curve
    value_errs: 
        Uncertainty values for B, beta and tau

    Returns
    -------
    stats:
        Model fit curve stats, in Matplotlib format
    """
    
    # converts B into mT
    B *= 1e3
    B_err *= 1e3          
    # converts tau into µs
    tau *= 1e6
    tau_err *= 1e6  
  
    B_r, B_err_r = rounder(B,B_err)
    beta_r,beta_err_r = rounder(beta,beta_err)
    tau_r,tau_err_r = rounder(tau,tau_err)

    stats = (
        rf"$B = {B_r}\ \pm\ {B_err_r}\ \mathrm{{mT}}$" "\n"
        rf"$\beta = {beta_r}\ \pm\ {beta_err_r}\ \mathrm{{rad}}$" "\n"
        rf"$\tau_{{\mathrm{{damp}}}} = {tau_r}\ \pm\ {tau_err_r}\ \mu s$")
    
    return stats
