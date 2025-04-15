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

    def process_level_accel(self, speeds: np.array, Tic: np.float64) -> pd.DataFrame:
        """Process the IADS data to extract energy height and Mach number for level acceleration.
        
        Args:
            - speeds: calibrated airpseed desired to extract from DAS data (try to match handwritten data)
            - Tic: Recorded flight temperature (Kelvin)

        Returns:
            - Dataframe containing:
                -- Time since beginning of maneuver
                -- Energy height at each calibrated airspeed
                -- Mach number at each calibrated airspeed
                -- Specific excess power at each calibrated airspeed
        
        """
        # Find index of flight_data set for desired speeds and average the nearest +/- 10 samples
        df = pd.DataFrame()
        for speed in speeds:
            index = np.where(np.isclose(self.Vic, speed, atol=0.1))[0][0]
            df = pd.concat([df, pd.DataFrame([{
                'time': np.average(self.time[index-10:index+10]),
                'Vic': np.average(self.Vic[index-10:index+10]) * 1.6878,  # to fps
                'Hic': np.average(self.Hic[index-10:index+10]),
                'Tic': Tic,
            }])], ignore_index=True)

        # Create a standard atmosphere object
        atm = StandardAtmosphere()

        # Calculate true airspeed
        qcic_psl = calculate_calibrated_pressure_differential_ratio(df['Vic'])
        qcic_ps = qcic_psl / atm.delta(df['Hic'])
        ps = atm.delta(df['Hic']) * atm.constants.PRESSURE_SEA_LEVEL
        qcic = qcic_ps * ps
        Mic = calculate_mach(qcic, ps)
        Vt = calculate_true_airspeed(Mic, df['Tic']/atm.constants.TEMP_SEA_LEVEL)

        # Calculate energy height over time
        energy_height = df['Hic'] + Vt**2 / 2

        # Append Mic and energy height to dataframe
        df['Mic'] = Mic
        df['energy_height'] = energy_height

        return df

        

def performance():
    """ Main Function """
    # Specify data file name
    filename = os.path.join('Performance', 'flight_data.csv')

    """Data import and pre-processing"""
    # Import flight data
    flight_data = LJ_IADS_flight_data(filename)

    # Process level acelleration data
    speeds = np.array([257, 258, 259, 260, 261, 262, 263])
    # Recorded flight temp
    Tic = -10 + 273.15  # Celsius to Kelvin
    # Process data
    LA_data = flight_data.process_level_accel(speeds, Tic)
    # Extract energy height and Mach number
    energy_height = LA_data['energy_height']
    Mic = LA_data['Mic']
    # Extract time
    time = LA_data['time']

    # Plot
    plot_energy_height_mach(time, energy_height, Mic)

if __name__ == "__main__":
    performance()


