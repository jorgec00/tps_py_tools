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

def LASC():
    """ Main Function """
    # Specify data file name
    filename = os.path.join('Performance','5-2-25-1', 'LA_95_10K_5-2-1.csv')

    """Data import and pre-processing"""
    # Import flight data from IADS/hand collected data
    flight_data = LJ_hand_LASC_flight_data(filename)

    # Process data
    LA_data = flight_data.process_level_accel()

    # Add hand-faired data curves to energy height and Mach number plot
    hand_data = pd.read_csv(os.path.join('Performance','5-2-25-1','LA_95_10K_hand_curve.csv'))

    # Plot
    (fig, ax) = plot_energy_height_mach(LA_data.time, LA_data.energy_height, LA_data.Mic, hand_data)
    
    # TODO: Calculate specfic excess power from hand-faired data curves
    hand_Ps = np.gradient(hand_data['Eh (ft)'], hand_data['Time (s)'])
    print(hand_Ps)

    # TODO: Import and process SAWTOOTH CLIMB data pairs
    filenames = [os.path.join('Performance', '5-2-25-1','SC_95_20K_5-2-1_92.csv'), 
                 os.path.join('Performance', '5-2-25-1','SC_95_20K_5-2-1_275.csv')]
     
    # Import flight data from IADS/hand collected data
    SC_Ps = np.empty(len(filenames), dtype=np.float64)
    Mic = np.empty(len(filenames), dtype=np.float64)
    for i, filename in enumerate(filenames):
        flight_data = LJ_hand_LASC_flight_data(filename)
        # Process data
        processed_flight_data = flight_data.process_SC()
        SC_Ps[i] = processed_flight_data.Ps
        Mic[i] = processed_flight_data.Mic
    # Average the two SC data sets for a final Ps
    SC_data = dict()
    SC_data['Ps'] = np.average(SC_Ps, axis=0)
    SC_data['Mic'] = np.average(Mic, axis=0)

    # Plot Ps on its own plot
    (fig, ax) = plot_Ps(hand_data, hand_Ps, SC_data)

    
    # TODO: Plot sawtooth climb data and specific excess power and specification
    '''
    
    '''


if __name__ == "__main__":
    LASC()