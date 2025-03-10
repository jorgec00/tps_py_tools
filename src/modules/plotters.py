import numpy as np
import matplotlib.pyplot as plt
from .std_atm import AtmosphereModel

# Update default font size and weight
plt.rcParams.update({
    'axes.labelsize': 12,        # Font size for x- and y-labels
    'axes.labelweight': 'bold',  # Font weight for x- and y-labels
    'axes.titlesize': 14,        # Font size for figure title
    'axes.titleweight': 'bold',  # Font weight for figure title

    # (Leave tick params as defaults)
})

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
    fig.savefig("dMpc_vs_Mic.png")

    """Plot Altitude position correction vs instrument corrected Airspeed"""
    fig, ax = plotter(Vic, dHpc, r"Instrument Corrected Airspeed, $V_{ic}$ (knots)", r"Altitude Position Correction, $\Delta$ $H_{pc}$ (feet)")
    fig.savefig("dHpc_vs_Vic.png")


    """Plot Altitude position correction vs instrument corrected Airspeed"""
    fig, ax = plotter(Vic, dVpc, r"Instrument Corrected Airspeed, $V_{ic}$ (knots)", r"Airspeed Position Correction, $\Delta$ $V_{pc}$ (knots)")
    fig.savefig("dVpc_vs_Vic.png")


    """Plot position correction ratio vs instrument corrected Airspeed"""
    fig, ax = plotter(Mic, dPp_qcic, r"Instrument Corrected Mach Number, $M_{ic}$", r"Static Position Error Pressure Coefficient, $\Delta$ $P_{p} / q_{cic}$")
    fig.savefig("dPp_qcic_vs_Vic.png")


    """Plot temp parameter vs mach parameter"""
    fig, ax = plotter(mach_param, temp_param, r"Mach Parameter, $M_{ic}^2/5$", r"Temperature Parameter, $T_{ic} / T_{a} - 1$")
    ax.plot(mach_param, temp_pred, 'k', linewidth=0.5)
    fig.savefig("temp_mach.png")

    plt.show()