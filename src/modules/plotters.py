import numpy as np
import matplotlib.pyplot as plt
from .std_atm import AtmosphereModel
import os
from typing import Tuple

# Update default font size and weight
plt.rcParams.update({
    'axes.labelsize': 12,        # Font size for x- and y-labels
    'axes.labelweight': 'bold',  # Font weight for x- and y-labels
    'axes.titlesize': 14,        # Font size for figure title
    'axes.titleweight': 'bold',  # Font weight for figure title

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

        return mach, curve_top


def plot_static_position_error_analysis(results: dict, std_atm: AtmosphereModel):
    """Define function for plotting"""
    def plotter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str):
        # Create figure for Mach vs dMic
        fig, ax = plt.subplots(figsize=(9, 6))

        # Position error comparison plot
        ax.plot(x, y, 'ks', label='Tower Flyby')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.minorticks_on()
        #ax.set_ylim(-np.abs(np.min(y*1.2)), np.max(y)*1.2)
        fig.legend()
        ax.grid(True)

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
    temp_pred = results["temp_pred"]

    """Plot Mach position correction vs instrument corrected Mach"""
    fig, ax = plotter(Mic, dMpc, r"Instrument Corrected Mach, M$_{ic}$", r"Mach Position Correction, $\Delta$ $M_{pc}$")    
    fig.savefig(os.path.join("PF7111","plots","dMpc_vs_Mic.png"))

    """Plot Altitude position correction vs instrument corrected Airspeed"""
    fig, ax = plotter(Vic, dHpc, r"Instrument Corrected Airspeed, $V_{ic}$ (knots)", r"Altitude Position Correction, $\Delta$ $H_{pc}$ (feet)")
    fig.savefig(os.path.join("PF7111","plots", "dHpc_vs_Vic.png"))


    """Plot Altitude position correction vs instrument corrected Airspeed"""
    fig, ax = plotter(Vic, dVpc, r"Instrument Corrected Airspeed, $V_{ic}$ (knots)", r"Airspeed Position Correction, $\Delta$ $V_{pc}$ (knots)")
    fig.savefig(os.path.join("PF7111","plots", "dVpc_vs_Vic.png"))


    """Plot position correction ratio vs instrument corrected Airspeed"""
    fig, ax = plotter(Mic, dPp_qcic, r"Instrument Corrected Mach Number, $M_{ic}$", r"Static Position Error Pressure Coefficient, $\Delta$ $P_{p} / q_{cic}$")
    fig.savefig(os.path.join("PF7111","plots", "dPp_qcic_vs_Vic.png"))
    
    """Plot position correction ratio vs instrument corrected airspeed, overaly MIL-P-26292C Mil Spec"""
    # Initiate MIL-P-26292C
    mil_spec = MIL_P_26292C()
    mach, curve_top = mil_spec.curve_a()

    # Plot!
    plt.figure()
    plt.plot(mach, curve_top, 'k')
    


    """Plot temp parameter vs mach parameter"""
    fig, ax = plotter(mach_param, temp_param, r"Mach Parameter, $M_{ic}^2/5$", r"Temperature Parameter, $T_{ic} / T_{a} - 1$")
    ax.plot(mach_param, temp_pred, 'k', linewidth=0.5)
    fig.savefig(os.path.join("PF7111","plots", "temp_mach.png"))

    plt.show()