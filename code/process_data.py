#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:50:34 2026

@author: lukethomas
"""

from file_formatting_and_preprocess import read_data,energy_channel_separation,bin_analysis
from analysis_and_fit import asymmetry_calculation, B_field_calculation, field_profile_coeff_calculation  
from plotting import histogram_plots, asymmetry_plot, quadratics_plot 
from guess_parameters import value_check 

# ====================================FINAL CALL FUNCTION AND CHECKING=========================================

def pipeline_10KeV(filename):
    """
    Runs the 10keV analysis pipeline for the muon detector dataset

    1. Read 10 keV data using read_data
    2. Split detector channels into left/right arrays using energy_channel_separation.
    3. Compute bin edges and bin centres using bin_analysis.
    4. Compute asymmetry and uncertainties using asymmetry_calculation.
    5. Fit the damped-asymmetry model using B_field_calculation to obtain:
       - B (T), beta (rad), tau_damp (s), and their uncertainties.
    6. Run robustness/range checks on fitted parameters using value_check.

    Parameters
    ----------
    filename :
        Path to the data file.

    Returns
    -------
    results_10kev : 
        Dictionary of fitted parameters at 10 keV:
        - 10keV_B (T), 10keV_B_error (T)
        - beta (rad), beta_error (rad)
        - 10keV_tau_damp (s), 10keV_tau_damp_error (s)

    plotting_data_10kev : 
        Dictionary containing arrays/values required for plotting:
        - L_data, R_data
        - bin_edges, centres
        - A, A_error
        - B, B_error
        - beta, beta_error
        - tau, tau_error

    Raises
    ------
    FileNotFoundError:
        If `filename` cannot be opened (raised inside read_data).
    ValueError:
        If the file cannot be parsed (e.g. missing required header/columns for 10 keV)
        (raised inside read_data / earlier helper functions).
    ValueError:
        If any fitted parameter fails the expected range checks:
        - 10keV_B not in (0.5e-3, 30e-3) T
        - beta not in (0.5, 1.5) rad
        - 10keV_tau_damp not in (2e-6, 10e-6) s
    """
    
    # 10keV data processing
    print("FILENAME:", repr(filename))
    data_10 = read_data(filename,10)

    left_arr, right_arr = energy_channel_separation(data_10)
    bin_edges, centres = bin_analysis(data_10)

    A_fit, A_unc,counts_left, counts_right = asymmetry_calculation(left_arr,right_arr,bin_edges)
    B_fit, B_err, beta_fit, beta_err, tau_fit, tau_err= B_field_calculation(data_10)
 
    # 10KeV data range checks
 
    raised_errors = []
        
    error = value_check('10keV_B',B_fit,0.5e-3,30e-3)
    if error:
        raised_errors.append(error)

    error = value_check('beta',beta_fit,0.5,1.5)
    if error:
        raised_errors.append(error)
        
    error = value_check('10keV_tau_damp',tau_fit,2e-6,10e-6)
    if error:
        raised_errors.append(error)
        
    if raised_errors:
        raise ValueError(f"Multi energy value check failed: {raised_errors}")
    
    results_10kev = {
        "10keV_B": B_fit,
        "10keV_B_error": B_err,
        "beta": beta_fit,
        "beta_error": beta_err,
        "10keV_tau_damp": tau_fit,
        "10keV_tau_damp_error": tau_err
        }
    
    plotting_data_10kev = {
        "L_data": left_arr,
        "R_data": right_arr,
        
        "bin_edges": bin_edges,
        "centres": centres,
       
        "A": A_fit,
        "A_error": A_unc,
        "B": B_fit,
        "B_error": B_err,
        "beta": beta_fit,
        "beta_error": beta_err,
        "tau": tau_fit,
        "tau_error": tau_err}

    return results_10kev, plotting_data_10kev


def multi_energy_pipeline(filename,energies_arr = [5, 10, 15, 20, 25]):
    
    """
    Calculates B(E) and uncertainty for energies in energies_arr and fits a quadratic (B(E) = aE^2 + bE + c) to extraxt coefficients
    
    Parameters
    ----------
    filename: 
        Data file path
    energies_arr: 
        Energies (keV), set to [5, 10, 15, 20, 25]

    Returns
    -------
    coeffs: 
        A tuple of the quadratic coefficients (a, b, c) in B(E) = aE^2 + bE + c
    coeff_errs: 
        A tuple of uncertainties of coefficients (a_err, b_err, c_err) 
    Returns
    -------
    results_multi : 
        Dictionary containing quadratic coefficients and uncertainties:
        - B(Energy)_coeffs: tuple (a, b, c)
        - B(Energy)_coeffs_errors: tuple (a_err, b_err, c_err)

        Coefficient units (typical):
        - a : T / keV^2
        - b : T / keV
        - c : T

    plotting_multi_E : 
        Plotting payload containing:
        - 'a', 'a_err', 'b', 'b_err', 'c', 'c_err'
        - B_energy_arr : list of B values (T) in the same order as energies_arr
        - B_energy_unc_arr : list of B uncertainties (T)

    Raises
    ------
    ValueError:
        There are less than three working energy values 
    ValueError:
        If any of the energies associated B values is not within the expected range (0.5e-3,30e-3)
      
    Notes
    -----
    The quadratic fit is weighted using uncertianites in B

    """
    
    B_energy_arr = []
    B_energy_unc_arr = []
    raised_errors = []
    
    if len(energies_arr) < 3:
        raise ValueError("For quadratic curve fit, need atleast 3 energy levels to extraxt coefficients")
    
    
    # loops over energies [5, 10, 15, 20, 25] and calcuates B for each using B_field_calculation
    for E in energies_arr:

        E_data = read_data(filename,E)
    
        B_fit, B_err, beta_fit, beta_err, tau_fit, tau_err= B_field_calculation(E_data)
        
            
        error = value_check(f'B_energy_coeffs_{E}KeV',B_fit,0.5e-3,30e-3)
        if error:
            raised_errors.append(error)
        
        
        # builds an array of B and B_unc values
        B_energy_arr.append(B_fit)
        B_energy_unc_arr.append(B_err)
        
    if raised_errors:
        raise ValueError(f"Multi energy value check failed: {raised_errors}")
        

    (a,b,c),(a_err,b_err,c_err) = field_profile_coeff_calculation(energies_arr,B_energy_arr,B_energy_unc_arr) 
    
    
    
    results_multi={"B(Energy)_coeffs":(a,b,c), #tuple of a,b,c for quadratic,linear and constant terms
                                                  #for fitting B dependence on energy
                                                  #(T/keV^2,T/keV,T)
              "B(Energy)_coeffs_errors":(a_err,b_err,c_err), # Errors in above in same order with the same units
              }
    
    plotting_multi_E = {"a": a,
                        "a_err": a_err,
                        "b":b,
                        "b_err": b_err,
                        "c": c,
                        "c_err": c_err,
                        "B_energy_arr": B_energy_arr,
                        "B_energy_unc_arr": B_energy_unc_arr}
      
    return results_multi,plotting_multi_E


def ProcessData(filename):
    """
    Final call function that computes a **complete analysis** for the muon detector dataset.

    This function acts as the top-level pipeline and combines:
    1) t10 keV analysis (full asymmetry + damped-fit + plots)
    2)  multi-energy analysis (B(E) extraction + quadratic fit + plot)

    Main process
    ------------
    1. 10 keV pipeline:
       - Splits channels, bins the data, computes asymmetry A(t),
         and fits the damped-asymmetry model to extract:
         - B (T), beta (rad), tau_damp (s) + uncertainties
       - Produces:
         - left/right histograms
         - asymmetry data vs fitted model at 10 keV

    2. Multi-energy pipeline:
       - Extracts B and uncertainty for multiple energies
       - Fits a weighted quadratic field profile:
         B(E) = a E^2 + b E + c
       - Produces the B vs E (field profile) plot with the quadratic fit

    3. Robustness checks
       - File integrity / required columns per energy are handled primarily by 'read_data'
       - Fitted parameter sanity checks are handled inside the sub-pipelines using 'value_check'

    Parameters
    ----------
    filename:
        Path to the data file.

    Returns
    -------
    results: 
        Dictionary containing:
        - 10 keV fit results:
          '10keV_B', '10keV_B_error', 'beta', 'beta_error',
          '10keV_tau_damp', '10keV_tau_damp_error'
          
        - quadratic coefficients describing B(E):
          'B(Energy)_coeffs' = (a, b, c)
          'B(Energy)_coeffs_errors' = (a_err, b_err, c_err)

    Raises
    ------
    FileNotFoundError:
        If 'filename' cannot be opened (raised inside 'read_data')
    ValueError:
        If the file format is invalid or required columns/data are missing for any processed energy
        (raised inside 'read_data')
    ValueError:
        If any parameter range checks fail inside 'pipeline_10KeV' or 'multi_energy_pipeline'


    Notes
    -----
    - Each pipeline has its own docstring detialing what it specifically does and any limitations present 
    - Some robustness checks are through the code and mitigating error, rather than raising errors (eg: divide by 0 errors, or uneven bin an analysis)
    
    """
    
    results = {
        "10keV_B": None,
        "10keV_B_error": None,
        "beta": None,
        "beta_error": None,
        "10keV_tau_damp": None,
        "10keV_tau_damp_error": None,
        "B(Energy)_coeffs": (None, None, None),
        "B(Energy)_coeffs_errors": (None, None, None),
    }

    # ---- 10 keV ----
    results_10kev, plot10 = pipeline_10KeV(filename)
    results.update(results_10kev)


    
    # left and right histograms
    histogram_plots(plot10["L_data"], plot10["R_data"], plot10["bin_edges"])
   
    # asymmetry model and data plot
    asymmetry_plot(
        plot10["A"], plot10["A_error"],
        plot10["B"], plot10["B_error"],
        plot10["beta"], plot10["beta_error"],
        plot10["tau"], plot10["tau_error"],
        plot10["centres"])


    # ---- multi-energy ----
    results_multi, plotM = multi_energy_pipeline(filename)
    results.update(results_multi)

    # multi-energy plot
    quadratics_plot(
        plotM["B_energy_arr"], plotM["B_energy_unc_arr"],
        plotM["a"], plotM["a_err"],
        plotM["b"], plotM["b_err"],
        plotM["c"], plotM["c_err"],
    )

    return results

    










