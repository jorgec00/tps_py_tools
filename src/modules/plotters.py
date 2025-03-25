import numpy as np
import matplotlib.pyplot as plt
from .std_atm import AtmosphereModel
import os
from typing import Tuple
from scipy.stats import t
import pandas as pd

# Update default font size and weight
plt.rcParams.update({
    'axes.labelsize': 18,        # Font size for x- and y-labels
    'axes.labelweight': 'bold',  # Font weight for x- and y-labels
    'axes.titleweight': 'bold',  # Font weight for figure title
    'axes.grid': True,
    'xtick.labelsize': 15,        # Font size for x-tick labels
    'ytick.labelsize': 15,        # Font size for y-tick labels
    'lines.linewidth': 2.5,
    'figure.constrained_layout.use': True
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

def plot_static_position_error_analysis(results: dict, std_atm: AtmosphereModel, model_data: pd.DataFrame  = None) -> None:
    """Define function for plotting"""
    def plotter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str,  spec: bool = False):
        # Create figure for Mach vs dMic
        fig, ax = plt.subplots(figsize=(12, 8))

        # Position error comparison plot
        if spec:
            ax.plot(x, y, 'k', label='Data', linewidth=2)
        else:
            ax.plot(x, y, 'ks')
        ax.set_xlabel(xlabel, family='sans-serif')
        ax.set_ylabel(ylabel, family='sans-serif')
        ax.minorticks_on()
        #ax.set_ylim(-np.abs(np.min(y*1.2)), np.max(y)*1.2)
        #ax.legend(loc='best', fontsize=16)

        fig.tight_layout()

        return fig, ax

    """Extract TFB data"""
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
    fig1, ax1 = plotter(Mic, dMpc, r"Instrument Corrected Mach, M$_{\mathbf{ic}}$", r"Mach Position Correction, ${\mathbf{\Delta M_{pc}}}$")    

    """Plot Altitude position correction vs instrument corrected Airspeed"""
    fig2, ax2 = plotter(Vic, dHpc, r"Instrument Corrected Airspeed, ${\mathbf{V_{ic}}}$ (knots)", r"Altitude Position Correction, ${\mathbf{\Delta H_{pc}}}$ (feet)")

    """Plot Altitude position correction vs instrument corrected Airspeed"""
    fig3, ax3 = plotter(Vic, dVpc, r"Instrument Corrected Airspeed, ${\mathbf{V_{ic}}}$ (knots)", r"Airspeed Position Correction, ${\mathbf{\Delta V_{pc}}}$ (knots)")

    """Plot position correction ratio vs instrument corrected Airspeed"""
    fig4, ax4 = plotter(Mic, dPp_qcic, r"Instrument Corrected Mach Number, ${\mathbf{M_{ic}}}$", r"Static Position Error Pressure Coefficient, ${\mathbf{\Delta P_{p} / q_{cic}}}$")
    

    """Plot temp parameter vs mach parameter"""
    fig5, ax5 = plotter(mach_param, temp_param, r"Mach Parameter, ${\mathbf{M_{ic}^2/5}}$", r"Temperature Parameter, ${\mathbf{T_{ic} / T_{a} - 1}}$")
    ax5.plot(mach_param, temp_param_pred, 'k', linewidth=1)
    # Calculate 95% prediction interval from Student T
    interval = t.interval(0.95, len(temp_param)-1)
    temp_lower = temp_param_pred + interval[0]*np.std(temp_param_pred - temp_param)
    temp_upper = temp_param_pred + interval[1]*np.std(temp_param_pred - temp_param)
    # Overlay
    ax5.plot(mach_param, temp_lower, 'k--', dashes=(15,30))
    ax5.plot(mach_param, temp_upper, 'k--', dashes=(15,30))
    # plot the Kt = 1 and intercept zero curve for comparison
    ax5.plot(mach_param, mach_param, 'k', linewidth=1, label=r"$\mathbf{K_{t} = 1}$")
    ax5.set_xlim([-0.01, 0.18])
    ax5.set_ylim([-0.01, 0.18])


    # Prepare the handfaired curve data
    if model_data is not None:
        # Extract hand faired curve data
        """Extract hand faired data curves for Mach (should the same at all altitudes)"""
        model_Mic = model_data["2300"]["Mic"]
        model_dPp_qcic = model_data["2300"]["dPp_qcic"]

        """Plot position correction ratio vs instrument corrected airspeed, overaly MIL-P-26292C Mil Spec"""
        # Plot
        fig6, ax6 = plotter(model_Mic, 
                            model_dPp_qcic, 
                            r"Instrument Corrected Mach Number, ${\mathbf{M_{ic}}}$", 
                            r"Static Position Error Pressure Coefficient, ${\mathbf{\Delta P_{p} / q_{cic}}}$", spec=True)
        
        # Initiate MIL-P-26292C
        mil_spec = MIL_P_26292C()
        mach, curve_top, curve_bottom = mil_spec.curve_a()
        ax6.plot(mach, curve_top, 'k--', linewidth=0.5)
        ax6.plot(mach, curve_bottom, 'k--', linewidth=0.5)

        # Curve B
        mach, curve_top, curve_bottom = mil_spec.curve_b()
        ax6.plot(mach, curve_top, 'k', linewidth=0.5)
        ax6.plot(mach, curve_bottom, 'k', linewidth=0.5)

        # Curve B with Noseboom
        mach, curve_top, curve_bottom = mil_spec.curve_b_NB()
        ax6.plot(mach, curve_top, 'k--', linewidth=0.5)
        ax6.plot(mach, curve_bottom, 'k--', linewidth=0.5)

        for md in model_data:
            model_dMPc = model_data[md]["dMpc"]
            model_dHpc = model_data[md]["dHpc"]
            model_dVpc = model_data[md]["dVpc"]
            model_Mic = model_data[md]["Mic"]
            model_Vic = model_data[md]["Vic"]
            model_dPp_qcic = model_data[md]["dPp_qcic"]

            # Sort all the data in order from smallest to larges Vic
            sort_idx = np.argsort(model_Vic)
            model_dMPc = model_dMPc[sort_idx]
            model_dHpc = model_dHpc[sort_idx]
            model_dVpc = model_dVpc[sort_idx]
            model_Mic = model_Mic[sort_idx]
            model_Vic = model_Vic[sort_idx]
            model_dPp_qcic = model_dPp_qcic[sort_idx]

            """" Find 95% confidence interval """
            # Interpolate the model to the points from the data
            dPp_qcic_interp = np.interp(Mic, model_Mic, model_dPp_qcic)
            interval = t.interval(0.95, len(Mic)-1)
            dPp_lower = dPp_qcic_interp  + interval[0]*np.std(dPp_qcic_interp - dPp_qcic)
            dPp_upper = dPp_qcic_interp  + interval[1]*np.std(dPp_qcic_interp - dPp_qcic)

            # Plot
            # Sort for plotting
            sort_idx = np.argsort(Mic)
            ax4.plot(Mic[sort_idx], dPp_lower[sort_idx], 'k--', dashes=(10,30))
            ax4.plot(Mic[sort_idx], dPp_upper[sort_idx], 'k--', dashes=(10,30))


            # Add data to Mach correction plot
            ax1.plot(model_Mic, model_dMPc, 'k')

            # Add data to Altitude position correction plot
            ax2.plot(model_Vic, model_dHpc, 'k')

            # Add data to Airspeed position correction plot
            ax3.plot(model_Vic, model_dVpc, 'k')

            # Add data to position error comparison plot
            ax4.plot(model_Mic, model_dPp_qcic, 'k')

        fig6.savefig(os.path.join("PF7111","plots", "dPp_qcic_vs_Mic_MIL_SPEC.png"))

    fig1.savefig(os.path.join("PF7111","plots","dMpc_vs_Mic.png"))
    fig2.savefig(os.path.join("PF7111","plots", "dHpc_vs_Vic.png"))
    fig3.savefig(os.path.join("PF7111","plots", "dVpc_vs_Vic.png"))
    fig4.savefig(os.path.join("PF7111","plots", "dPp_qcic_vs_Mic.png"))
    fig5.savefig(os.path.join("PF7111","plots", "temp_mach.png"))
    

    plt.show()

        