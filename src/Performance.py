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
from modules.plotters import plot_static_position_error_analysis
import os
import matplotlib.pyplot as plt

class LJ_IADS_flight_data:
    def __init__(self, filename: str):
        """
        Import IADS data from a CSV file.
        
        Args:
            str (filename): Path to the CSV file containing IADS data.
            
        Returns:
            - Time
            - Instrument Corrected Airspeed
            - Instrument Corrected Altitude

        """
        data = pd.read_csv(filename)
        self.time = data['Time'].to_numpy(np.float64)
        self.Vic = data['Instrument Corrected Airspeed'].to_numpy(np.float64)
        self.Hic = data['Instrument Corrected Altitude'].to_numpy(np.float64)

def performance():
    """ Main Function """
    # Import flight data
    filename = os.path.join('Performance', 'data.csv')
    flight_data = LJ_IADS_flight_data(filename)

    Tic = -10 + 273.15  # Celsius to Kelvin

    # Create a standard atmosphere object
    atm = StandardAtmosphere()

    # Calculate true airspeed
    qcic_psl = calculate_calibrated_pressure_differential_ratio(flight_data.Vic)
    qcic_ps = qcic_psl / atm.delta(flight_data.Hic)
    ps = atm.delta(flight_data.Hic) * atm.constants.PRESSURE_SEA_LEVEL
    qcic = qcic_ps * ps
    Mic = calculate_mach(qcic, ps)
    Vt = calculate_true_airspeed(Mic, Tic/atm.constants.TEMPERATURE_SEA_LEVEL)

    # Calculate energy height over time
    energy_height = flight_data.alt + Vt**2 / 2

    

