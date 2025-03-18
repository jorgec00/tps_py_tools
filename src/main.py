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
from modules.plotters import plot_static_position_error_analysis
import os
import matplotlib.pyplot as plt
import sys


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

class load_TFB_Data:
    ''' Extract tower flyby data from excel file'''
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
    Air data computer that processes indicated airspeed and indicated altitude with a 
    static position correction model to process corrections to flight parameters.
    """

    def __init__(self, atmosphere: AtmosphereModel):
        self.atmosphere = atmosphere

    def process_measurements(self,
                             indicated_airspeed: np.ndarray,
                             indicated_altitude: np.ndarray,
                             model: dict
                             ) -> dict:
        """
        Process air data measurements to calculate flight conditions.
        """
        # Calculate calibrated pressure differential ration
        qcic_psl = calculate_calibrated_pressure_differential_ratio(indicated_airspeed * 1.6878)

        # Calculate differential pressure ratio
        qcic_ps = qcic_psl / pressure_ratio(indicated_altitude)

        # Calculate static pressure
        Ps = self.atmosphere.pressure(indicated_altitude)

        # Indicated Mach
        Mic = calculate_mach(qcic_ps * Ps, Ps)

        ## Interpolate the mode data with indicated mach
        dPp_qcic = np.interp(Mic, model["Mic"], model["dPp_qcic"])

        # Plot for comparison
        plt.plot(Mic, dPp_qcic, 'ks', label="dPp_qcic")
        plt.plot(model["Mic"], model["dPp_qcic"], 'r*--', label="Hand-Faired Curve")

        # Calculate error ratio
        dPp_Ps = dPp_qcic * qcic_ps

        # Calculate true differential pressure ratio
        qc_pa = (qcic_ps + 1) / (1 - dPp_Ps) - 1
        
        # Calculate true ambient pressure
        Pa = Ps * (1 - dPp_Ps)
        
        # Calculate true calibrated differential pressure
        qc_psl = qc_pa * Pa / constants.PRESSURE_SEA_LEVEL

        # Calculate Mach Error Correction
        Mc = calculate_mach(qc_pa * Pa, Pa)
        dMpc = Mc - Mic

        # Calculate calibrated airspeed error correction
        Vic = calculate_calibrated_airspeed(qcic_psl) / 1.6878 # converted to kts
        Vc = calculate_calibrated_airspeed(qc_psl) / 1.6878 # converted to kts
        dVpc = Vc - Vic

        # Calculate altitude position correction
        Hic = calculate_pressure_altitude(Ps / constants.PRESSURE_SEA_LEVEL) # in feet
        Hc = calculate_pressure_altitude(Pa / constants.PRESSURE_SEA_LEVEL) # in feet
        dHpc = Hc - Hic

        # Return
        return {
            "dMpc": dMpc,
            "dHpc": dHpc,
            "dVpc": dVpc,
            "Mic": Mic,
            "Vic": Vic,
            "dPp_qcic": dPp_qcic
        }

    def generate_model_data(self,
                             altitude: np.float64,
                             model: dict,
                             ) -> dict:
        """
        Process air data measurements to calculate flight conditions.
        """
        # Mach
        Mic = model["Mic"]

        # Calculate static pressure
        Ps = self.atmosphere.pressure(altitude)

        # Calculate qcic_ps
        qcic_ps = calculate_pressure_differential_ratio(Mic)

        # Calculate calibrated pressure differential ration
        qcic_psl = qcic_ps * pressure_ratio(altitude)

        ## dPp_qcic
        dPp_qcic = model["dPp_qcic"]

        # Calculate error ratio
        dPp_Ps = dPp_qcic * qcic_ps

        # Calculate true differential pressure ratio
        qc_pa = (qcic_ps + 1) / (1 - dPp_Ps) - 1
        
        # Calculate true ambient pressure
        Pa = Ps * (1 - dPp_Ps)
        
        # Calculate true calibrated differential pressure
        qc_psl = qc_pa * Pa / constants.PRESSURE_SEA_LEVEL

        # Calculate Mach Error Correction
        Mc = calculate_mach(qc_pa * Pa, Pa)
        dMpc = Mc - Mic

        # Calculate calibrated airspeed error correction
        Vic = calculate_calibrated_airspeed(qcic_psl) / 1.6878 # converted to kts
        Vc = calculate_calibrated_airspeed(qc_psl) / 1.6878 # converted to kts
        dVpc = Vc - Vic

        # Calculate altitude position correction
        Hic = calculate_pressure_altitude(Ps / constants.PRESSURE_SEA_LEVEL) # in feet
        Hc = calculate_pressure_altitude(Pa / constants.PRESSURE_SEA_LEVEL) # in feet
        dHpc = Hc - Hic

        # Return
        return {
            "dMpc": dMpc,
            "dHpc": dHpc,
            "dVpc": dVpc,
            "Mic": Mic,
            "Vic": Vic,
            "dPp_qcic": dPp_qcic
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

    print(sys.executable)

    # Load RFB flight data from excel file (after DAS processing OR from Test Card)
    print("\nLoading Tower Fly By Data...") #See sample excel spreadsheet for spreasheet format

    #use path.join to avoid compatiblity issues between Linux/Windows
    file_path = os.path.join("PF7111", "TFB_20250307_378_DAS.xlsx") 
    data = load_TFB_Data(file_path)

    # Create a TFB calculator
    TFB_calc = TFB_calculator(std_atm)
    
    # Process data with standard atmosphere
    print("\nProcessing with standard atmosphere...")
    TFB_results = TFB_calc.process_measurements(
        data.indicated_airspeed,
        data.indicated_altitude,
        data.grid_reading,
        data.grid_pressure_altitude,
        data.tower_temperature,
        data.indicated_temperature
    )
    '''
    # Save mach and dPp_qcic in excel handfaired estimates
    print("\nSaving results to excel file...")
    results_df = pd.DataFrame({
        "Mic": TFB_results["Mic"],
        "dPp_qcic": TFB_results["dPp_qcic"]
    })
    results_df.to_excel(os.path.join("PF7111", "TFB_20250307_378_results.xlsx"), index=False)
    '''

    # Load hand faired position error curve (comment out if no curve found yet)
    print("\nLoading hand faired curve data...")
    #use path.join to avoid compatiblity issues between Linux/Windows
    model_data = pd.read_excel(os.path.join("PF7111", "hand_faired_curve.xlsx"))
    error_model = dict({
        "Mic": model_data["Mic"].to_numpy(dtype=np.float64),
        "dPp_qcic": model_data["dPp_qcic"].to_numpy(dtype=np.float64),
    })

    print(f"\nData for hand faired curve:")
    print(f"Minimum Indicated Mach: {np.min(TFB_results['Mic'])}")
    print(f"Maximum Indicated Mach: {np.max(TFB_results['Mic'])}")

    # Calculate all paramters using the ADC and the new model
    ADC = AirDataComputer(std_atm)
    #model_data = ADC.process_measurements(data.indicated_airspeed, data.indicated_altitude, model=error_model)
    model_data_2300 = ADC.generate_model_data(np.float64(2300), model=error_model)
    model_data_10K = ADC.generate_model_data(np.float64(10000), model=error_model)
    model_data_20k = ADC.generate_model_data(np.float64(20000), model=error_model)
    model_data = pd.DataFrame({
        "2300": model_data_2300,
        "10000": model_data_10K,
        "20000": model_data_20k,
    })

    # Plot position error analysis
    plot_static_position_error_analysis(
        TFB_results,
        std_atm,
        model_data=model_data
    )

    print(f"\nData for hand faired curve:")
    print(f"Minimum Indicated Mach: {np.min(TFB_results['Mic'])}")
    print(f"Maximum Indicated Mach: {np.max(TFB_results['Mic'])}")

    print(sys.executable)

if __name__ == "__main__":
    main()

