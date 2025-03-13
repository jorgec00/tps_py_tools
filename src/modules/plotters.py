import numpy as np
import matplotlib.pyplot as plt
from .std_atm import AtmosphereModel
import os
from typing import Tuple
from scipy.stats import t

# Update default font size and weight
plt.rcParams.update({
    'axes.labelsize': 18,        # Font size for x- and y-labels
    'axes.labelweight': 'bold',  # Font weight for x- and y-labels
    'axes.titleweight': 'bold',  # Font weight for figure title
    'axes.grid': True,
    'xtick.labelsize': 16,        # Font size for x-tick labels
    'ytick.labelsize': 16,        # Font size for y-tick labels
    # (Leave tick params as defaults)
})

class MIL_P_26292C:
    '''Define the MILL-P-26292C standard lines for plotting'''
    def __init__(self):
        pass

    def curve_a(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Define "Curve A" 
        
            Args:
                - None
                
            Returns:
                - mach: mach number points for x axes plot
                - curve_top: top of curve point for y axis plot corresponding to each mach
                - curve_bottom: bottom of curve for y axis plot corresponding to each mach
        '''
        # Mach Array
        mach = np.array([0.3, 0.4, 0.5, 0.6, 0.7 , 0.8, 0.9, 1., 1.1, 1.2])        
        curve_top = np.array([0.02, 0.02, 0.02, 0.017, 0.012, 0.008, 0.005, 0.003, 0.002, 0.002])
        curve_bottom = np.array([-0.015, -0.015, -0.015, -0.015, -0.012, -0.008, -0.005, -0.003, -0.002, -0.002])

        return mach, curve_top, curve_bottom
    
    def curve_b(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Define "Curve B without a noseboom system" 
        
            Args:
                - None
                
            Returns:
                - mach: mach number points for x axes plot
                - curve_top: top of curve point for y axis plot corresponding to each mach
                - curve_bottom: bottom of curve for y axis plot corresponding to each mach
        '''
        # Mach Array
        mach = np.array([0.3, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0, 1.0355, 1.15])        
        curve_top = np.array([0.02, 0.02, 0.0205, 0.0215, 0.02269, 0.0246, 0.02707, 0.0305, 0.03649, 0.0445, 0.01, 0.01])
        curve_bottom = -np.array([0.015, 0.015, 0.01539, 0.01637, 0.01775, 0.01952, 0.022, 0.025, 0.03073, 0.04, 0.01, 0.01])

        return mach, curve_top, curve_bottom
    
    def curve_b_NB(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Define "Curve B with a noseboom system" 
        
            Args:
                - None
                
            Returns:
                - mach: mach number points for x axes plot
                - curve_top: top of curve point for y axis plot corresponding to each mach
                - curve_bottom: bottom of curve for y axis plot corresponding to each mach
        '''
        # Mach Array
        mach = np.array([0.3, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0, 1.0355, 1.045, 1.15])        
        curve_top = np.array([0.02, 0.02, 0.0205, 0.0215, 0.02269, 0.0246, 0.02707, 0.0305, 0.03649, 0.0445, 0.01, 0.004, 0.004])
        curve_bottom = -np.array([0.015, 0.015, 0.01539, 0.01637, 0.01775, 0.01952, 0.022, 0.025, 0.03073, 0.04, 0.01, 0.004, 0.004])

        return mach, curve_top, curve_bottom

def plot_static_position_error_analysis(results: dict, std_atm: AtmosphereModel):
    """Define function for plotting"""
    def plotter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str):
        # Create figure for Mach vs dMic
        fig, ax = plt.subplots(figsize=(12, 8))

        # Position error comparison plot
        ax.plot(x, y, 'ks', label='Tower Flyby')
        ax.set_xlabel(xlabel, family='sans-serif')
        ax.set_ylabel(ylabel, family='sans-serif')
        ax.minorticks_on()
        #ax.set_ylim(-np.abs(np.min(y*1.2)), np.max(y)*1.2)
        ax.legend(loc='best', fontsize=16)

        fig.tight_layout()

        return fig, ax

    """Extract data"""
    dMpc = results["dMpc"]
    dHpc = results["dHpc"]
    dVpc = results["dVpc"]
    Mic = results["Mic"]
    Vic = results["Vic"]
    dPp_qcic = results["dPp_qcic"]
    temp_param = results["temp_param"]
    mach_param = results["mach_param"]
    temp_param_pred = results["temp_pred"]

    """Plot Mach position correction vs instrument corrected Mach"""
    fig, ax = plotter(Mic, dMpc, r"Instrument Corrected Mach, M$_{\mathbf{ic}}$", r"Mach Position Correction, ${\mathbf{\Delta M_{pc}}}$")    
    fig.savefig(os.path.join("PF7111","plots","dMpc_vs_Mic.png"))

    """Plot Altitude position correction vs instrument corrected Airspeed"""
    fig, ax = plotter(Vic, dHpc, r"Instrument Corrected Airspeed, ${\mathbf{V_{ic}}}$ (knots)", r"Altitude Position Correction, ${\mathbf{\Delta H_{pc}}}$ (feet)")
    fig.savefig(os.path.join("PF7111","plots", "dHpc_vs_Vic.png"))

    """Plot Altitude position correction vs instrument corrected Airspeed"""
    fig, ax = plotter(Vic, dVpc, r"Instrument Corrected Airspeed, ${\mathbf{V_{ic}}}$ (knots)", r"Airspeed Position Correction, ${\mathbf{\Delta V_{pc}}}$ (knots)")
    fig.savefig(os.path.join("PF7111","plots", "dVpc_vs_Vic.png"))

    """Plot position correction ratio vs instrument corrected Airspeed"""
    fig, ax = plotter(Mic, dPp_qcic, r"Instrument Corrected Mach Number, ${\mathbf{M_{ic}}}$", r"Static Position Error Pressure Coefficient, ${\mathbf{\Delta P_{p} / q_{cic}}}$")
    fig.savefig(os.path.join("PF7111","plots", "dPp_qcic_vs_Vic.png"))
    
    """Plot position correction ratio vs instrument corrected airspeed, overaly MIL-P-26292C Mil Spec"""
    # Initiate MIL-P-26292C
    mil_spec = MIL_P_26292C()
    mach, curve_top, curve_bottom = mil_spec.curve_a()

    # Plot!
    plt.figure()
    plt.plot(mach, curve_top, 'k--')
    plt.plot(mach, curve_bottom, 'k--')

    # Curve B
    mach, curve_top, curve_bottom = mil_spec.curve_b()
    plt.plot(mach, curve_top, 'k')
    plt.plot(mach, curve_bottom, 'k')

    # Curve B with Noseboom
    mach, curve_top, curve_bottom = mil_spec.curve_b_NB()
    plt.plot(mach, curve_top, 'k--')
    plt.plot(mach, curve_bottom, 'k--')

    """Plot temp parameter vs mach parameter"""
    fig, ax = plotter(mach_param, temp_param, r"Mach Parameter, ${\mathbf{M_{ic}^2/5}}$", r"Temperature Parameter, ${\mathbf{T_{ic} / T_{a} - 1}}$")
    ax.plot(mach_param, temp_param_pred, 'k', linewidth=0.5)
    # Calculate 95% prediction interval from Student T
    interval = t.interval(0.95, len(temp_param)-1)
    temp_lower = temp_param_pred + interval[0]*np.std(temp_param_pred - temp_param)
    temp_upper = temp_param_pred + interval[1]*np.std(temp_param_pred - temp_param)
    # Overlay
    ax.plot(mach_param, temp_lower, 'k--', dashes=(10,30))
    ax.plot(mach_param, temp_upper, 'k--', dashes=(10,30))

    fig.savefig(os.path.join("PF7111","plots", "temp_mach.png"))

    