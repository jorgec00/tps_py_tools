#!/usr/bin/env python
# coding: utf-8

# # PF7111A Performance Reporting
# **See:** Erb, Russell E. “Pitot-Statics and the Standard Atmosphere, 4th Ed.” Edwards AFB, CA: USAF Test Pilot School, July 2020.
# 
# **Authors:** Jorge Cervantes
# 
# **Date:13 April 2025**
# 

# ------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from dataclasses import dataclass
from modules.std_atm import *
import pandas as pd
from modules.helpers import *
from modules.LJ_library import LJ_hand_LASC_flight_data
from modules.plotters import plot_energy_height_mach, plot_Ps
import os
import matplotlib.pyplot as plt

def LASC_95():
    """ Generate 10K Ps Curves for the given power setting """
    # Specify data file name
    filename = os.path.join('Performance','data_folder', 'LA_95_10K_5-2-1_IADS.csv')

    """Data import and pre-processing"""
    # Import flight data from IADS/hand collected data
    flight_data = LJ_hand_LASC_flight_data(filename)

    # Process data
    LA_data = flight_data.process_level_accel()
    print(LA_data.Mic)

    # Load hand-faired data for Energy height and Mach number
    hand_data = pd.read_excel(os.path.join('Performance','data_folder','LA_95_10K_5-2-1_hand_SAT.xlsx'))

    # Plot
    (fig, ax) = plot_energy_height_mach(LA_data.time, LA_data.energy_height, LA_data.Mic, hand_data)
    
    # TODO: Calculate specfic excess power from hand-faired data curves
    hand_Ps = np.gradient(hand_data['Eh (ft)'], hand_data['Time (s)'])
    print(hand_data['Eh (ft)'])
    print(hand_Ps)

    # Plot Ps on its own plot
    (fig, ax) = plot_Ps(hand_data, hand_Ps)

    # Plot the requirement
    # Create a standard atmosphere object
    atm = StandardAtmosphere()

    # Calculate true airspeed
    Mr = 0.525
    Psr = 83 # requirement
    # Add requirement to plot
    ax.plot(Mr, Psr, 'ko', label='Requirement')
    

    # Import and process sawtooth climb data
    filename = os.path.join('Performance', 'data_folder','SC_summary.csv')
    # Import flight data from data summary
    SC_data = pd.read_csv(filename)
    # Select only the SC_data for 10000 ft
    SC_data = SC_data[SC_data['Altitude (ft)'] == 10000]
    # Use temperature to then convert to Mach nnumber
    # Speed of sound at this temperature
    SC_data['a'] = (atm.constants.SPECIFIC_HEAT_RATIO * atm.constants.GAS_CONSTANT * (SC_data['Temperature (C)'] + 273.15))**0.5
    SC_data['Mach'] = SC_data['True Airspeed (ft/s)'] / SC_data['a']

    # Add SC data to plot
    ax.plot(SC_data['Mach'], SC_data['Ps (ft/s)'], 'kx', label='SC data')

    plt.show()

if __name__ == "__main__":
    LASC_95()