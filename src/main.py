#!/usr/bin/env python
# coding: utf-8

# # PF7111A Performance Reporting
# **See:** Erb, Russell E. “Pitot-Statics and the Standard Atmosphere, 4th Ed.” Edwards AFB, CA: USAF Test Pilot School, July 2020.
# 
# **Authors:** Clark McGehee and Julian McCafferty
# *Modified by* Jorge Cervantes
# 
# **Date:** 5 March 2025
# 

# ------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from dataclasses import dataclass
from modules.std_atm import *
import pandas as pd
from modules.helpers import *
from modules.plotters import *
import os

#test

class TalonSortieData:
    def __init__(self, filepath: str):
        data = pd.read_csv(filepath)
        # Measured values
        self.time_s = data["time_s"].to_numpy(dtype=np.float64)
        self.velocity_north = data["vn_fps"].to_numpy(dtype=np.float64)
        self.velocity_east = data["ve_fps"].to_numpy(dtype=np.float64)
        self.velocity_down = data["vd_fps"].to_numpy(dtype=np.float64)
        self.geometric_height = data["z_ft"].to_numpy(dtype=np.float64)
        self.total_pressure = data["pt_psi"].to_numpy(dtype=np.float64)
        self.static_pressure = data["pa_psi"].to_numpy(dtype=np.float64)
        self.total_temperature = data["tt_k"].to_numpy(dtype=np.float64)
        self.angle_of_attack = data["aoa_rad"].to_numpy(dtype=np.float64)
        self.angle_of_sideslip = data["aos_rad"].to_numpy(dtype=np.float64)
        self.roll = data["phi_rad"].to_numpy(dtype=np.float64)
        self.pitch = data["theta_rad"].to_numpy(dtype=np.float64)
        self.yaw = data["psi_rad"].to_numpy(dtype=np.float64)
        # Derived values
        self.differential_pressure = self.total_pressure - self.static_pressure

class TFBData:
    def __init__(self, filepath: str):
        data = pd.read_excel(filepath)
        self.indicated_airspeed = data["Vic"].to_numpy(dtype=np.float64)
        self.indicated_altitude = data["Hic"].to_numpy(dtype=np.float64)
        self.grid_reading = data["grid_reading"].to_numpy(dtype=np.float64)
        self.grid_pressure_altitude = data["grid_pa"].to_numpy(dtype=np.float64)
        self.tower_temperature = data["tower_temp"].to_numpy(dtype=np.float64)+273.15
        self.indicated_temperature = data["T_ic"].to_numpy(dtype=np.float64)+273.15

class AirDataComputer:
    """
    Air data computer that processes pressure and temperature measurements
    to calculate airspeeds using a provided atmosphere model.
    """

    def __init__(self, atmosphere: AtmosphereModel):
        self.atmosphere = atmosphere

    def process_measurements(self,
                             geometric_altitude: np.ndarray,
                             total_pressure: np.ndarray,
                             static_pressure: np.ndarray,
                             total_temperature: np.ndarray) -> dict:
        """
        Process air data measurements to calculate flight conditions.
        """
        # Calculate indicated quantities
        qcic_ps = (total_pressure - static_pressure) / static_pressure
        mach_ic = calculate_mach(total_pressure - static_pressure, static_pressure)

        # Convert geometric to geopotential altitude
        geopotential_altitude = calculate_geopotential_altitude(geometric_altitude)
        # Get true ambient pressure from atmosphere model
        ambient_pressure = self.atmosphere.pressure(geopotential_altitude)

        # Calculate observed position error (Ps - Pa)/Ps
        observed_error = (static_pressure - ambient_pressure) / static_pressure

        # Calculate EAS using observed position error correction
        qc_pa_obs = (qcic_ps + 1) / (1 - observed_error) - 1
        mach_pc_obs = calculate_mach(qc_pa_obs * static_pressure, static_pressure)

        # Calculate ambient temperature from total temperature
        temperature = total_temperature / (1 + 0.2 * constants.TEMPERATURE_RECOVERY_FACTOR * mach_pc_obs * mach_pc_obs)

        # Get atmosphere properties and ratios
        temperature_ratio = self.atmosphere.theta(geopotential_altitude)
        pressure_ratio = self.atmosphere.delta(geopotential_altitude)
        density_ratio = self.atmosphere.sigma(geopotential_altitude)

        # Calculate pressure altitude
        pressure_altitude = calculate_pressure_altitude(pressure_ratio)

        # Calculate airspeeds
        eas_obs = calculate_equivalent_airspeed(mach_pc_obs, pressure_ratio)  # Using observed correction

        return {
            'mach_ic': mach_ic,
            'mach_pc_obs': mach_pc_obs,
            'eas_obs': eas_obs,  # EAS using observed position error
            'temperature': temperature,
            'pressure': static_pressure,
            'density': density_ratio * constants.DENSITY_SEA_LEVEL,
            'temperature_ratio': temperature_ratio,
            'pressure_ratio': pressure_ratio,
            'density_ratio': density_ratio,
            'pressure_altitude': pressure_altitude,
            'observed_error': observed_error
        }

class TFB_calculator:
    '''
    Calculator that converts tower flyby data to static position corrections for Mach, Pressure Altitude, and Calibrated Airspeed
    '''
    def __init__(self, atmosphere: AtmosphereModel):
        self.atmosphere = atmosphere

    def process_measurements(self, 
                             indicated_airspeed: np.ndarray,
                             indicated_altitude: np.ndarray,
                             grid_reading: np.ndarray,
                             grid_pressure_altitude: np.ndarray,
                             tower_temperature: np.ndarray,
                             indicated_temperature) -> dict:
        # Std Temp at tower pressure altitude
        T_std = self.atmosphere.temperature(grid_pressure_altitude)
        Hc = grid_pressure_altitude + TFB_constants.GRID_CONSTANT*grid_reading*T_std/tower_temperature

        # Altitude error correction
        dHpc = Hc - indicated_altitude

        # True ambient pressure
        Pa = self.atmosphere.pressure(Hc)
        # Static Pressure
        Ps = self.atmosphere.pressure(indicated_altitude)

        # Calibrated Differential Pressure (measured)
        qcic_psl = calculate_calibrated_pressure_differential_ratio(indicated_airspeed * 1.6878) #converted to kts
        # Differential Pressure (measured)
        qcic_ps = qcic_psl / pressure_ratio(indicated_altitude)

        #Position Error Ratio
        dPp_Ps = (Ps - Pa) / Ps

        # Differential Pressure Ratio (actual)
        qc_pa = (qcic_ps + 1) / (1 - dPp_Ps) - 1
        qc_psl = qc_pa * pressure_ratio(Hc)

        # Mach
        Mic = calculate_mach(qcic_ps * Ps, Ps)
        Mc = calculate_mach(qc_pa * Pa, Pa)

        # Mach correction
        dMpc = Mc - Mic

        # Airspeed
        Vic = calculate_calibrated_airspeed(qcic_psl) / 1.6878 # converted to kts
        Vc = calculate_calibrated_airspeed(qc_psl) / 1.6878 

        # Airspeed correction
        dVpc = Vc - Vic

        # Position Error Ratio
        dPp_qcic = dPp_Ps / qcic_ps

        #Calculate the temperature and mach parameters for temp recovery factor chart
        temp_param = indicated_temperature / tower_temperature - 1
        mach_param = 0.2*Mc**2
        
        # Calculate temperature recovery factor
        Kt, bias = np.polyfit(mach_param, temp_param, 1)
        # Calculate predictions using these values
        temp_pred = Kt * mach_param + bias

        print(f"Temperature Recovery Factor: {Kt}")
        print(f"Bias: {bias}")
    

        return {"dHpc": dHpc,
                "dMpc": dMpc,
                "dVpc": dVpc,
                "Mic": Mic,
                "Vic": Vic,
                "dPp_qcic": dPp_qcic,
                "temp_param": temp_param,
                "mach_param": mach_param,
                "temp_pred": temp_pred}
        

def main():
    # Initialize atmosphere models
    print("Initializing atmosphere models...")
    std_atm = StandardAtmosphere()

    # Load flight data
    print("\nLoading Tower Fly By Data...") #See sample excel spreadsheet for spreasheet format
    file_path = os.path.join("PF7111", "TFB.xlsx") #use path.join to avoid compatiblity issues between Linux/Windows
    sortie = TFBData(file_path)

    # Create a TFB calculator
    TFB_calc = TFB_calculator(std_atm)
    
    # Process data with standard atmosphere
    print("\nProcessing with standard atmosphere...")
    TFB_results = TFB_calc.process_measurements(
        sortie.indicated_airspeed,
        sortie.indicated_altitude,
        sortie.grid_reading,
        sortie.grid_pressure_altitude,
        sortie.tower_temperature,
        sortie.indicated_temperature
    )


    # Add results to TFB for plotting
    '''
    # Process data with test atmosphere
    print("Processing with test atmosphere...")
    test_results = test_adc.process_measurements(
        sortie.geometric_height,
        sortie.total_pressure * 144,  # Convert PSI to PSF
        sortie.static_pressure * 144,  # Convert PSI to PSF
        sortie.total_temperature
    )
    '''

    # Print summary statistics
    '''print("\nLast Tower Point (Standard Atmosphere):")
    print(f"Altimeter Position Correction: {TFB_results['dHpc'][-1]} ft")
    print(f"Airspeed Position Correction: {TFB_results['dVpc'][-1]}")
    print(f"Mach Position Correction: {TFB_results['dMpc'][-1]}")
    print(f"Indicated Mach: {TFB_results['Mic'][-1]}")
    print(f"Indicated Airspeed: {TFB_results['Vic'][-1]}")
    print(f"Position Correction Ratio: {TFB_results['dPp_qcic'][-1]}")'''

    # Plot position error analysis
    plot_static_position_error_analysis(
        TFB_results,
        std_atm
    )

if __name__ == "__main__":
    main()

