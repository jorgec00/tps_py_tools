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
from modules.plotters import plot_energy_height_mach
import os

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
        self.time = data['Time']
        seconds = np.array([x.split(':')[-1] for x in self.time], dtype=np.float64)
        minutes = np.array([x.split(':')[-2] for x in self.time], dtype=np.float64)
        hours = np.array([x.split(':')[-3] for x in self.time], dtype=np.float64)
        self.time = hours * 3600 + minutes * 60 + seconds # convert time to seconds
        self.time = self.time - self.time[0]  # Start time at zero seconds
        self.Vic = data['CalibratedAirspeed'].to_numpy(np.float64) * 1.6878 # to fps
        self.Hic = data['Altitude'].to_numpy(np.float64) # feet

        # Remove any value where there were link hits or zeros
        rm_mask = (np.isnan(self.Vic) | (self.Vic == 0))
        self.time = self.time[~rm_mask]
        self.Vic = self.Vic[~rm_mask]
        self.Hic = self.Hic[~rm_mask]

def performance():
    """ Main Function """
    """ Specifications """
    # Denote sample rate and desired moving average time window
    sample_rate = 100  # Hz
    average_time = 0  # seconds
    average_window = int((average_time+0.01)*sample_rate); # time converted to samples, 0.01 added for centered moving average

    # Denote length of maneuver, in seconds
    maneuver_time = 120 # seconds
    maneuver_samples = int(maneuver_time * sample_rate)

    # Specify data file name
    filename = os.path.join('Performance', 'flight_data.csv')

    # Manevuer (options - level accel: "LA", check climb: "C", check descent: "D", level sustained turn: "T")
    maneuver = "LA"

    """Data import and pre-processing"""
    # Import flight data
    flight_data = LJ_IADS_flight_data(filename)

    # Calculate the centered moving average 
    flight_data.Vic = centered_moving_average(flight_data.Vic, average_window)
    flight_data.Hic = centered_moving_average(flight_data.Hic, average_window)

    # Recorded flight temp
    Tic = np.full_like(flight_data.Hic, -10 + 273.15)  # Celsius to Kelvin

    # Select only the maneuver data
    flight_data.Hic = flight_data.Hic[:maneuver_samples]
    flight_data.Vic = flight_data.Vic[:maneuver_samples]
    flight_data.time = flight_data.time[:maneuver_samples]
    Tic = Tic[:maneuver_samples]

    # Downsample for plotting
    flight_data.Hic = flight_data.Hic[::average_window]
    flight_data.Vic = flight_data.Vic[::average_window]
    flight_data.time = flight_data.time[::average_window]
    flight_data.Tic = Tic[::average_window]
    
    # Create a standard atmosphere object
    atm = StandardAtmosphere()

    """Process Level Accelleration Data"""
    if maneuver == "LA":
        # Calculate true airspeed
        qcic_psl = calculate_calibrated_pressure_differential_ratio(flight_data.Vic)
        qcic_ps = qcic_psl / atm.delta(flight_data.Hic)
        ps = atm.delta(flight_data.Hic) * atm.constants.PRESSURE_SEA_LEVEL
        qcic = qcic_ps * ps
        Mic = calculate_mach(qcic, ps)
        Vt = calculate_true_airspeed(Mic, flight_data.Tic/atm.constants.TEMP_SEA_LEVEL)

        # Calculate energy height over time
        energy_height = flight_data.Hic + Vt**2 / 2

        # Plot
        plot_energy_height_mach(flight_data.time, energy_height, Mic)

if __name__ == "__main__":
    performance()


