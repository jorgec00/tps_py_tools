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

def plot_eas_comparison(time: np.ndarray, results: dict):
    """Plot comparison of EAS calculated using observed vs calibrated position errors."""
    # Convert to knots for plotting
    fps_to_knots = 0.592484

    eas_obs = results['eas_obs'] * fps_to_knots
    eas_cal = results['eas_cal'] * fps_to_knots
    eas_error = eas_cal - eas_obs  # Difference between calibration and observed

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # EAS comparison plot
    ax1.plot(time, eas_obs, 'b.', label='Observed Position Error')
    ax1.plot(time, eas_cal, 'r-', label='Calibrated Position Error')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Equivalent Airspeed (knots)')
    ax1.set_title('Equivalent Airspeed Comparison')
    ax1.legend()
    ax1.grid(True)

    # Error plot
    ax2.plot(time, eas_error, 'g.')
    ax2.axhline(y=0, color='k', linestyle='--')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('EAS Error (knots)')
    ax2.set_title('EAS Error (Calibration - Observed)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_eas_mach_characteristic(results: dict):
    """Plot EAS error vs Mach number comparing observed to calibrated corrections."""
    # Convert to knots for plotting
    fps_to_knots = 0.592484

    eas_obs = results['eas_obs'] * fps_to_knots
    eas_cal = results['eas_cal'] * fps_to_knots
    eas_error = eas_cal - eas_obs
    mach_ic = results['mach_ic']

    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(mach_ic, eas_error, 'b.')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Indicated Mach Number')
    plt.ylabel('EAS Error (Calibration - Observed) (knots)')
    plt.title('Speed Calibration Error Characteristic')
    plt.grid(True)
    plt.show()


def plot_comparison(time, std_results, test_results):
    """Create comparison plots of key parameters."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Mach number comparison
    ax1.plot(time, std_results['mach_ic'], 'b-', label='Standard')
    ax1.plot(time, test_results['mach_ic'], 'r--', label='Test')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mach')
    ax1.legend()
    ax1.grid(True)

    # True airspeed comparison
    ax2.plot(time, std_results['tas'], 'b-', label='Standard')
    ax2.plot(time, test_results['tas'], 'r--', label='Test')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('True Airspeed (ft/s)')
    ax2.legend()
    ax2.grid(True)

    # Pressure altitude comparison
    ax3.plot(time, std_results['pressure_altitude'], 'b-', label='Standard')
    ax3.plot(time, test_results['pressure_altitude'], 'r--', label='Test')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Pressure Altitude (ft)')
    ax3.legend()
    ax3.grid(True)

    # Temperature ratio comparison
    ax4.plot(time, std_results['temperature_ratio'], 'b-', label='Standard')
    ax4.plot(time, test_results['temperature_ratio'], 'r--', label='Test')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Temperature Ratio')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


def plot_static_position_error_analysis(results: dict, std_atm: AtmosphereModel):
    """Extract data"""
    dMpc = results["dMpc"]
    dHpc = results["dHpc"]
    dVpc = results["dVpc"]
    Mic = results["Mic"]
    Vic = results["Vic"]
    dPp_qcic = results["dPp_qcic"]
    temp_param = results["temp_param"]
    mach_param = results["mach_param"]

    """Plot Mach position correction vs instrument corrected Mach"""
    # Create figure for Mach vs dMic
    plt.figure()

    # Position error comparison plot
    plt.plot(Mic, dMpc, 'ks', label='Tower Flyby')
    plt.xlabel(r"Instrument Corrected Mach, M$_{ic}$")
    plt.ylabel(r"Mach Position Correction, $\Delta$ $M_{pc}$")
    plt.minorticks_on()
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig("dMpc_vs_Mic.png")

    """Plot Altitude position correction vs instrument corrected Airspeed"""
    # Create figure for Mach vs dMic
    plt.figure()

    # Position error comparison plot
    plt.plot(Vic, dHpc, 'ks', label='Tower Flyby')
    plt.xlabel(r"Instrument Corrected Airspeed, $V_{ic}$ (knots)")
    plt.ylabel(r"Altitude Position Correction, $\Delta$ $H_{ic}$ (feet)")
    plt.minorticks_on()
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.savefig("dHpc_vs_Vic.png")

    """Plot Altitude position correction vs instrument corrected Airspeed"""
    # Create figure for Mach vs dMic
    plt.figure()

    # Position error comparison plot
    plt.plot(Vic, dVpc, 'ks', label='Tower Flyby')
    plt.xlabel(r"Instrument Corrected Airspeed, $V_{ic}$ (knots)")
    plt.ylabel(r"Airspeed Position Correction, $\Delta$ $V_{pc}$ (knots)")
    plt.minorticks_on()
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.savefig("dVpc_vs_Vic.png")

    """Plot position correction ratio vs instrument corrected Airspeed"""
    # Create figure for Mach vs dMic
    plt.figure()

    # Position error comparison plot
    plt.plot(Mic, dPp_qcic, 'ks', label='Tower Flyby')
    plt.xlabel(r"Instrument Corrected Mach Number, $M_{ic}$")
    plt.ylabel(r"Static Position Error Pressure Coefficient, $\Delta$ $P_{p} / q_{cic}$")
    plt.xlim(0, 1.4)
    ylim = np.max([np.abs(np.min(dPp_qcic*(1.1))), np.max(dPp_qcic*1.1)])
    plt.ylim(-ylim, ylim)
    plt.minorticks_on()
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.savefig("dPp_qcic_vs_Vic.png")

    """Plot temp parameter vs mach parameter"""
    # Create figure for Mach vs dMic
    plt.figure()

    # Position error comparison plot
    plt.plot(mach_param, temp_param, 'ks', label='Tower Flyby')
    plt.xlabel(r"Mach Parameter, $M_{ic}^2/5$")
    plt.ylabel(r"Temperature Parameter, $T_{ic} / T_{a} - 1$")
    plt.minorticks_on()
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.savefig("temp_mach.png")