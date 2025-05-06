import numpy as np
from modules.std_atm import *
import pandas as pd
from modules.helpers import *

class LJ_IADS_flight_data:

    def __init__(self, filename: str):
        """
        Import IADS data from a CSV file.
        
        Args:
            str (filename): Path to the CSV file containing IADS data.
            
        Returns:
            Class instance with the following attributes:
            - Time
            - Instrument Corrected Airspeed
            - Instrument Corrected Altitude

        """
        data = pd.read_csv(filename)
        self.time = data['Time (s)']
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
    
class LJ_hand_LASC_flight_data:
    
    def __init__(self, filename: str):
        """
        Import flight data from a csv file created from test card, hand collected data.
        
        Args:
            - str (filename): Path to the CSV file containing hand collected data. The data should be formatted in columns as follows:
                -- Time (mm:ss) from start of maneuver
                -- Calibrated Airspeed (kts)
                -- Indicated Altitude (feet)
            
        Returns:
            Class instance with the following attributes:
            - Time (in second from start of maneuver)
            - Calibrated Airspeed (fps)
            - Indicated Altitude (ft)
            - Temperature (Kelvin)

        """
        data = pd.read_csv(filename)
        # Seperate time by colon
        time_array = data['Time (s)'].str.split(':', expand=True)
        # Convert columns to float
        hh = time_array[0].astype(np.float64)
        mm = time_array[1].astype(np.float64)
        ss = time_array[2].astype(np.float64)
        # Calculate total seconds
        time = hh * 3600 + mm * 60 + ss
        # Calculate time from start of maneuver 
        self.time = np.array(time - time[0]) # Time in seconds from start of maneuver
        self.Vic = (data['Airspeed (KCAS)'].to_numpy(np.float64)) * 1.6878 # to fps (-4 is position corrected speed)
        self.Hic = data['Altitude (ft)'].to_numpy(np.float64) # feet
        self.Tic = data['Temperature (C)'].to_numpy(np.float64) + 273.15  # Celsius to Kelvin

    def process_level_accel(self):
        """Process the hand data to extract energy height and Mach number for level acceleration.
        
        Args:
            - None

        Returns:
            - Class instance containing the following;
                -- Time since beginning of maneuver
                -- Calibrated Airspeed (kts)
                -- Indicated Altitude (ft)
                -- Energy height at each calibrated airspeed
                -- Mach number at each calibrated airspeed
                -- Specific excess power at each calibrated airspeed
        
        """

        # Create a standard atmosphere object
        atm = StandardAtmosphere()

        # Calculate true airspeed
        qcic_psl = calculate_calibrated_pressure_differential_ratio(self.Vic)
        qcic_ps = qcic_psl / atm.delta(self.Hic)
        ps = atm.delta(self.Hic) * atm.constants.PRESSURE_SEA_LEVEL
        qcic = qcic_ps * ps
        Mic = calculate_mach(qcic, ps)
        Vt = calculate_true_airspeed(Mic, self.Tic/atm.constants.TEMP_SEA_LEVEL)

        # Calculate energy height over time
        energy_height = self.Hic + Vt**2 / (2 * atm.constants.GRAVITY_SEA_LEVEL)
        
        # Append Mic and energy height to dataframe
        self.Mic = Mic
        self.energy_height = energy_height
        

        return self

    def process_SC(self):
        """Process the hand data to extract Ps and Mach for sawtooth climb.
        
        Args:
            - None

        Returns:
            - Class instance containing the following:
                -- Ps at the average Mach number
                -- Average Mach number
        
        """

        # Create a standard atmosphere object
        atm = StandardAtmosphere()

        # Calculate true airspeed
        qcic_psl = calculate_calibrated_pressure_differential_ratio(self.Vic)
        qcic_ps = qcic_psl / atm.delta(self.Hic)
        ps = atm.delta(self.Hic) * atm.constants.PRESSURE_SEA_LEVEL
        qcic = qcic_ps * ps
        Mic = calculate_mach(qcic, ps)

        # Calculate Ps from dH/dt
        Ps = (self.Hic[-1] - self.Hic[0])/(self.time[-1] - self.time[0])
        
        
        # Append Mic and energy height to dataframe
        self.Mic = np.mean(Mic)
        self.Ps = Ps

        return self