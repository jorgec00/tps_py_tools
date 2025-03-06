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
from modules.helpers import calculate_mach, calculate_equivalent_airspeed, TFB_constants, calculate_pressure_altitude
from modules.plotters import *

class TalonSortieData:
    def __init__(self, filepath: str):
        data = pd.read_csv(filepath)
        # Measured values
        self.time_s = data["time_s"].to_numpy()
        self.velocity_north = data["vn_fps"].to_numpy()
        self.velocity_east = data["ve_fps"].to_numpy()
        self.velocity_down = data["vd_fps"].to_numpy()
        self.geometric_height = data["z_ft"].to_numpy()
        self.total_pressure = data["pt_psi"].to_numpy()
        self.static_pressure = data["pa_psi"].to_numpy()
        self.total_temperature = data["tt_k"].to_numpy()
        self.angle_of_attack = data["aoa_rad"].to_numpy()
        self.angle_of_sideslip = data["aos_rad"].to_numpy()
        self.roll = data["phi_rad"].to_numpy()
        self.pitch = data["theta_rad"].to_numpy()
        self.yaw = data["psi_rad"].to_numpy()
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

        # Get true ambient pressure from atmosphere model
        ambient_pressure = self.atmosphere.pressure(geometric_altitude)

        # Calculate observed position error (Ps - Pa)/Ps
        observed_error = (static_pressure - ambient_pressure) / static_pressure

        # Calculate EAS using observed position error correction
        qc_pa_obs = (qcic_ps + 1) / (1 - observed_error) - 1
        mach_pc_obs = calculate_mach(qc_pa_obs * static_pressure, static_pressure)

        # Calculate ambient temperature from total temperature
        temperature = total_temperature / (1 + 0.2 * constants.TEMPERATURE_RECOVERY_FACTOR * mach_pc_obs * mach_pc_obs)

        # Get atmosphere properties and ratios
        temperature_ratio = self.atmosphere.theta(geometric_altitude)
        pressure_ratio = self.atmosphere.delta(geometric_altitude)
        density_ratio = self.atmosphere.sigma(geometric_altitude)

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
                             tower_temperature: np.ndarray) -> dict:
        # Std Temp at tower pressure altitude
        T_std = self.atmosphere.temperature(grid_pressure_altitude)
        Hc = grid_pressure_altitude + TFB_constants.GRID_CONSTANT*grid_reading*T_std/tower_temperature

        # Altitude error correction
        dHpc = Hc - indicated_altitude

        return {"dHpc": dHpc}
        

def main():
    # Initialize atmosphere models
    print("Initializing atmosphere models...")
    std_atm = StandardAtmosphere()

    # Load flight data
    print("\nLoading Toer Fly By Data...")
    sortie = TFBData("PF7111\TFB.xlsx")

    # Create a TFB calculator
    TFB_calc = TFB_calculator(std_atm)
    
    # Process data with standard atmosphere
    print("\nProcessing with standard atmosphere...")
    TFB_results = TFB_calc.process_measurements(
        sortie.indicated_airspeed,
        sortie.indicated_altitude,
        sortie.grid_reading,
        sortie.grid_pressure_altitude,
        sortie.tower_temperature
    )

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
    print("\nFlight Summary (Standard Atmosphere):")
    print(f"\nAverage Altimeter Position Correction: {TFB_results['dHpc'].mean():.0f} ft")
    '''print(f"Maximum Indicated Mach: {std_results['mach_ic'].max():.3f}")
    print(f"Maximum Corrected Mach - Reference Calibration: {std_results['mach_pc_cal'].max():.3f}")
    print(f"Maximum Corrected Mach - Observed Calibration: {std_results['mach_pc_obs'].max():.3f}")'''

    # Plot standard comparisons
    #plot_comparison(sortie.time_s, std_results, test_results)

    # Plot position error analysis
    ''''
    'plot_position_error_analysis(
        std_results['mach_ic'],
        sortie.static_pressure * 144,
        test_atm,
        sortie.geometric_height,
        cal
    )'
    '''

    # Plot EAS comparisons
    '''print("\nGenerating EAS comparison plots...")
    plot_eas_comparison(sortie.time_s, std_results)
    plot_eas_mach_characteristic(std_results)'''

if __name__ == "__main__":
    main()

