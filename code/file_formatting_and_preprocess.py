#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:58:47 2026

@author: lukethomas
"""
import numpy as np
import pandas as pd 


# ====================================FILE LOAD FUNCTIONS=========================================

def read_data(filename,E = 10):
    """
    Load and return the time and channel data for a given implantation energy or default of 10KeV
    Can deal with Variable Metadata.
    
    Parameters
    ----------
    filename: 
        Data file path
    E: 
        Energy (keV)
    
    Returns
    -------
    energy_data: 
        Data for the selected energy
    
    
    Raises
    ------
    FileNotFoundError:
        If filename cant be opened 
    ValueError:
        If row cannot be found starting with 'Time'
    ValueError:
        If either time or channel columns are missing for the current working energy (E)
    ValueError:
        If channel values are not exclusively 1/2 or if either channel data is missing


    Notes
    -----
    - Channel 1 =  the left detector and channel = 2 as the right detector
    
    """
    
    try:
        f = open(filename,'r')
    except:
        raise FileNotFoundError(f"Missing file: {filename}")
     
     
  
    # loops through rows looking for the header of data and outputs row number it begins on

    header_line = None
    for i, line in enumerate(f):
        if line.startswith("Time"):
            header_line = i
            break
    f.close()

    # returns value error if time cant be found
    if header_line == None:
        raise ValueError("File format error: could not find a row starting with 'Time'") 
    
            
    # reads file at given header line - skipping meta data
    file = pd.read_csv(filename ,sep="\t", header= header_line)

    time_col = f"Time (us) @ {E}keV"
    channel_col = f"Channel @ {E}keV"
    

    # checks if working energy (E) hos both channel and time columns
    if time_col not in file.columns:
        raise ValueError(f"Missing columns for {E} KeV: {time_col} ")
        
       
    if channel_col not in file.columns:
        raise ValueError(f"Missing columns for {E} KeV: {channel_col} ")
    
    
    unique_channels = set(file[channel_col])
    
    # checks if the channel data for energy (E) has only 1,2 (left and right)
    if not unique_channels.issubset({1,2}):
        raise ValueError(f"{channel_col} has detected more than 2 channels (left and right data)")
        
    # checks if the channel data for energy (E) has both 1,2 (left and right)  
    if unique_channels != {1, 2}:
        raise ValueError("v: needs left and right data")
     

    energy_data = file[[time_col, channel_col]].copy()

    
    return energy_data

def bin_analysis(energy_array,B = 0.03, gamma = 851.616):
    """
    Bin analysis fucntion that calcuated all the neccisary bin data: 
        creates bin edges for energy specifc data and locates the centres of these ranges 
   
    The bin width = T/20,where T is the muon precession period calculated from 
    the upper limit of the magnetic field B, using T = 2*pi / (gamma*B)
    
    Parameters
    ----------
    energy_array: 
        Data for the selected energy, 'time_us' is in microseconds
    B: 
        Magnetic field (T) 
    gamma: 
        Gyromagnetic ratio for a positive muon (rad s^-1 T^-1) 

    Returns
    -------
    bin_edges: 
        Array of histogram bin edges across the full range of time in energy_array
    centres: 
        Array of bin centre times for each bin edge

    Notes
    -----
    This allows bin data to be case specifc to each each energy data and controlling the resoltion in T
    
    Bin edges are built from the full time range of the dataset.
    
    Assumes time_us is in µs and so gamma is taken to be 851.616, powers cancel 
    
    """
    # time period using Bmax (upper limit of B)
   
    T = 2 * np.pi / (gamma * B)
    
    # bind width using 20 bins per T
    
    bin_width = T / 20

    # bin edges calculated over entire data set provided divided into the bin widths
    
    bin_edges = np.arange(np.min(energy_array["time_us"]),np.max(energy_array["time_us"]) + bin_width,bin_width)

    # centre position of each bin
    
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_edges,centres


def energy_channel_separation(energy_data):
    """
    Splits the input detector data into left (channel 1) and right (channel 2) time arrays for a given energy
   
    Parameters
    ----------
    energy_data:
        Unsorted energy data of times (µs) for both left and right detector at a given energy

    Returns
    -------
    left_arr:
        Array of time values (µs) at a given energy, incident on the left detector
    right_arr:
        Array of time values (µs) at a given energy, incident on the right detector

   
    """
    
    # renames columns to time_u and channel
    
    energy_data.columns = ["time_us", "channel"]
    
    # filters L and R channels
    
    energy_data_left = energy_data[energy_data['channel'] == 1]
    energy_data_right = energy_data[energy_data['channel'] == 2]
    
    left_arr, right_arr = energy_data_left['time_us'],energy_data_right['time_us']
          
    return left_arr, right_arr