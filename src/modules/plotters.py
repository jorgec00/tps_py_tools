import numpy as np
import matplotlib as plt

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


def plot_position_error_analysis(mach_ic, static_pressure, atm, altitude, cal):
    """Plot position error analysis including reference comparison."""
    # Get true ambient pressure from atmosphere model
    ambient_pressure = atm.pressure(altitude)

    # Calculate observed position error (Ps - Pa)/Ps
    observed_error = (static_pressure - ambient_pressure) / static_pressure

    # Get reference error from calibration
    reference_error = cal.get_error(mach_ic)

    # Calculate residuals
    residuals = observed_error - reference_error

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Position error comparison plot
    ax1.plot(mach_ic, observed_error, 'b.', label='Observed')
    ax1.plot(mach_ic, reference_error, 'r-', label='Reference')
    ax1.set_xlabel('Indicated Mach Number, M$_{ic}$')
    ax1.set_ylabel(r'$\Delta P_p/P_s = (P_s - P_a)/P_s$')
    ax1.set_title('Static Source Position Error Characteristic')
    ax1.legend()
    ax1.grid(True)

    # Residual plot
    ax2.plot(mach_ic, residuals, 'g.')
    ax2.axhline(y=0, color='k', linestyle='--')
    ax2.set_xlabel('Indicated Mach Number, M$_{ic}$')
    ax2.set_ylabel('Residual Error')
    ax2.set_title('Error Residuals (Observed - Reference)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
