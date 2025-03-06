import numpy as np
from .std_atm import constants

class TFB_constants:
    GRID_CONSTANT: float = 31.4 #ft/div

def calculate_geopotential_altitude(geometric_altitude: np.ndarray) -> np.ndarray:
    """
    Convert geometric altitude to geopotential altitude.

    Geopotential altitude accounts for the variation of gravity with height
    and is used in standard atmosphere calculations. The conversion uses
    the formula:
        H = (R_E * h) / (R_E + h)
    where:
        H = geopotential altitude
        h = geometric altitude
        R_E = Earth radius

    Args:
        geometric_altitude: Array of geometric altitudes in feet

    Returns:
        Array of geopotential altitudes in feet
    """
    return (constants.EARTH_RADIUS * geometric_altitude) / (constants.EARTH_RADIUS + geometric_altitude)



def calculate_pressure_altitude(pressure_ratio: np.ndarray) -> np.ndarray:
    """
    Calculate pressure altitude for any layer based on pressure ratio.

    Args:
        pressure_ratio: Array of pressure ratios δ = p/p0 (dimensionless)

    Returns:
        Array of pressure altitudes (ft)
    """

    def calculate_tropospheric_pressure_altitude(pressure_ratio: np.ndarray) -> np.ndarray:
        """
        Calculate pressure altitude in the troposphere (0 to 36,089 ft).

        Args:
            pressure_ratio: Array of pressure ratios δ = p/p0 (dimensionless)

        Returns:
            Array of pressure altitudes (ft)
        """
        return (1 - pressure_ratio ** (1 / constants.TROPOSPHERE_EXP)) / constants.TROPOSPHERE_LAPSE

    def calculate_stratospheric_pressure_altitude(pressure_ratio: np.ndarray) -> np.ndarray:
        """
        Calculate pressure altitude in the stratosphere (36,089 to 65,617 ft).

        Args:
            pressure_ratio: Array of pressure ratios δ = p/p0 (dimensionless)

        Returns:
            Array of pressure altitudes (ft)
        """
        return (np.log(
            pressure_ratio / constants.STRATOSPHERE_PRESSURE_COEFF) / constants.STRATOSPHERE_DECAY + constants.TRANSITION_ALTITUDE)

    # Initialize output array
    H_c = np.zeros_like(pressure_ratio)

    # Split calculation by atmospheric layer
    troposphere_mask = pressure_ratio > constants.TROPOSPHERE_PRESSURE_RATIO
    stratosphere_mask = ~troposphere_mask

    # Calculate for each layer
    H_c[troposphere_mask] = calculate_tropospheric_pressure_altitude(pressure_ratio[troposphere_mask])
    H_c[stratosphere_mask] = calculate_stratospheric_pressure_altitude(pressure_ratio[stratosphere_mask])

    return H_c


def root(f, a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Find root of f(x) = 0 using bisection method.

    Args:
        f: Function that returns function value
        a: Left bracket
        b: Right bracket
        tol: Tolerance for convergence
        max_iter: Maximum iterations

    Returns:
        Root value
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Root must be bracketed between a and b")

    i = 0
    while i < max_iter:
        # Calculate midpoint
        c = (a + b) / 2
        fc = f(c)

        # Check if we've found the root
        if abs(fc) < tol:
            return c

        # Update brackets
        if fc * fa < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        i += 1

    raise RuntimeError(f"Failed to converge after {max_iter} iterations")


def calculate_mach(qc: np.ndarray, pa: np.ndarray) -> np.ndarray:
    """
    Calculate Mach number from impact pressure (qc) and static pressure (pa).

    Uses root finding to solve for Mach number in both subsonic and supersonic regimes.

    Args:
        qc: Impact pressure (total - static) in consistent units
        pa: Ambient static pressure in same units as qc

    Returns:
        Array of Mach numbers (dimensionless)
    """

    def calculate_subsonic_mach(qc_pa: np.ndarray) -> np.ndarray:
        """Calculate subsonic Mach number from pressure ratio."""
        sub_mach = np.sqrt(5 * ((qc_pa + 1) ** (2 / 7) - 1))
        return sub_mach

    def subsonic_zero_func(qc_pa: float, mach: float) -> float:
        """Zero function for subsonic Mach calculation."""
        sub_mach_0 = mach - np.sqrt(5 * ((qc_pa + 1) ** (2 / 7) - 1))
        return sub_mach_0

    def supersonic_zero_func(qc_pa: float, mach: float) -> float:
        """Zero function for supersonic Mach calculation."""
        super_mach_0 = super_mach_0  =(mach - 0.881284 * np.sqrt(
            (qc_pa + 1) * (1 - 1 / (7 * mach ** 2)) ** (5 / 2)
        ))
        return super_mach_0

    def combined_zero_func(qc_pa: float, mach: float) -> float:
        """Combined zero function for both regimes."""
        if mach < 1:
            return subsonic_zero_func(qc_pa, mach)
        else:
            return supersonic_zero_func(qc_pa, mach)

    # Input validation
    if np.any(qc < 0) or np.any(pa <= 0):
        raise ValueError("Pressures must be positive")

    # Calculate pressure ratio
    qc_pa = qc / pa

    # Initialize with subsonic estimate
    mach_estimates = calculate_subsonic_mach(qc_pa)

    # Refine using root finding
    mach = np.zeros_like(qc_pa)
    for i, (qp, m0) in enumerate(zip(qc_pa, mach_estimates)):
        # Define target function for this pressure ratio
        def target(m):
            return combined_zero_func(qp, m)

        # Find root using provided bisection method
        # Use appropriate brackets based on initial estimate
        if m0 < 0.95:  # Clearly subsonic
            mach[i] = root(target, 0, 1)
        else:  # Potentially supersonic
            try:
                mach[i] = root(target, 0.95, 5.0)
            except ValueError:  # If no supersonic solution, try subsonic
                mach[i] = root(target, 0, 1)

    return mach


def calculate_true_airspeed(mach: np.ndarray, temperature_ratio: np.ndarray) -> np.ndarray:
    """
    Calculate true airspeed using Vt = M * a_0 * sqrt(θ)

    Args:
        mach: Array of Mach numbers (dimensionless)
        temperature_ratio: Array of temperature ratios θ = T/T0 (dimensionless)

    Returns:
        Array of true airspeeds (ft/s) with invalid values replaced by np.nan
    """
    # Input validation
    if np.any(~np.isfinite(mach)) or np.any(~np.isfinite(temperature_ratio)):
        print("Warning: Non-finite values found in Mach or temperature ratio inputs")

    if np.any(temperature_ratio <= 0):
        print(f"Warning: {np.sum(temperature_ratio <= 0)} negative or zero temperature ratios found")

    # Create output array
    tas = np.full_like(mach, np.nan)

    # Calculate only for valid inputs
    valid_mask = (np.isfinite(mach) & np.isfinite(temperature_ratio) & (temperature_ratio > 0))
    tas[valid_mask] = (mach[valid_mask] * constants.SPEED_OF_SOUND_SEA_LEVEL *
                       np.sqrt(temperature_ratio[valid_mask]))

    # Report percentage of valid calculations
    valid_percent = 100 * np.sum(valid_mask) / len(mach)
    if valid_percent < 100:
        print(f"Warning: TAS calculation valid for {valid_percent:.1f}% of points")

    return tas


def calculate_equivalent_airspeed(mach: np.ndarray, pressure_ratio: np.ndarray) -> np.ndarray:
    """
    Calculate equivalent airspeed using Ve = M * a_0 * sqrt(δ)

    Args:
        mach: Array of Mach numbers (dimensionless)
        pressure_ratio: Array of pressure ratios δ = p/p0 (dimensionless)

    Returns:
        Array of equivalent airspeeds (ft/s) with invalid values replaced by np.nan
    """
    # Input validation
    if np.any(~np.isfinite(mach)) or np.any(~np.isfinite(pressure_ratio)):
        print("Warning: Non-finite values found in Mach or pressure ratio inputs")

    if np.any(pressure_ratio <= 0):
        print(f"Warning: {np.sum(pressure_ratio <= 0)} negative or zero pressure ratios found")

    # Create output array
    eas = np.full_like(mach, np.nan)

    # Calculate only for valid inputs
    valid_mask = (np.isfinite(mach) & np.isfinite(pressure_ratio) & (pressure_ratio > 0))
    eas[valid_mask] = (mach[valid_mask] * constants.SPEED_OF_SOUND_SEA_LEVEL *
                       np.sqrt(pressure_ratio[valid_mask]))

    # Report percentage of valid calculations
    valid_percent = 100 * np.sum(valid_mask) / len(mach)
    if valid_percent < 100:
        print(f"Warning: EAS calculation valid for {valid_percent:.1f}% of points")

    return eas


