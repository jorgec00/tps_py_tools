import numpy as np
from typing import Protocol
import pandas as pd

# ## The Standard Atmosphere
# 
# **See:** Chapter 2, The Standard Atmosphere
# 
class AtmosphereConstants:
    """
    Standard atmospheric constants from Tables 2.1 and relevant values form pages 12 - 14.
    All values in English Engineering Units.
    """

    # Base sea-level constants
    GRAVITY_SEA_LEVEL: float = 32.1741  # ft/sec^2
    EARTH_RADIUS: float = 20902808.99  # ft
    TEMP_SEA_LEVEL: float = 288.15  # K
    PRESSURE_SEA_LEVEL: float = 2116.22  # lb/ft^2
    DENSITY_SEA_LEVEL: float = 0.00237688  # slug/ft^3

    # Gas properties
    SPECIFIC_HEAT_RATIO: float = 1.40  # unitless
    GAS_CONSTANT: float = 3089.8  # ft-lb/slug-K

    # Layer boundaries
    MIN_ALTITUDE: float = -16404.2  # ft
    TRANSITION_ALTITUDE: float = 36089.24  # ft
    MAX_ALTITUDE: float = 65616.8  # ft

    # Troposphere properties (0 to 36,089 ft)
    TROPOSPHERE_LAPSE: float = 6.87559e-6  # 1/ft
    TROPOSPHERE_EXP: float = 5.2559  # unitless

    # Stratosphere properties (36,089 to 65,617 ft)
    STRATOSPHERE_TEMP: float = 216.65  # K (constant)
    STRATOSPHERE_PRESSURE_COEFF: float = 0.223360  # unitless
    STRATOSPHERE_DENSITY_COEFF: float = 0.297075  # unitless
    STRATOSPHERE_DECAY: float = -4.80637e-5  # 1/ft

    # Layer pressure ratios at key altitudes
    TROPOSPHERE_PRESSURE_RATIO: float = 0.223361  # at 36,089 ft
    STRATOSPHERE_PRESSURE_RATIO: float = 0.054032  # at 65,617 ft

    # Temperature ratios
    TROPOSPHERE_TEMP_RATIO: float = 0.751865  # at 36,089 ft and above
        
    # Temperature recovery factor
    TEMPERATURE_RECOVERY_FACTOR: float = 0.98 # unitless

    @property
    def SPEED_OF_SOUND_SEA_LEVEL(self) -> float:
        """Speed of sound at sea level (ft/sec)."""
        return np.sqrt(self.SPECIFIC_HEAT_RATIO * self.GAS_CONSTANT * self.TEMP_SEA_LEVEL)


constants = AtmosphereConstants()

def temperature_ratio(geopotential_altitude: np.ndarray) -> np.ndarray:
    """
    Calculate the temperature ratio (T/T0) for given geopotential altitude.

    This function implements the standard atmosphere temperature model, which has:
    - Linear decrease in troposphere (0 to 36,089 ft)
    - Constant temperature in stratosphere (36,089 to 65,617 ft)

    Args:
        geopotential_altitude: Array of altitudes in feet

    Returns:
        Array of temperature ratios (dimensionless)

    Raises:
        ValueError: If altitude is outside valid range [-16,404 to 65,617 ft]
    """
    # Input validation
    if np.any(geopotential_altitude < constants.MIN_ALTITUDE) or np.any(geopotential_altitude > constants.MAX_ALTITUDE):
        raise ValueError(f"Altitude must be between {constants.MIN_ALTITUDE} "
                         f"and {constants.MAX_ALTITUDE} feet")

    # Initialize output array
    theta = np.zeros_like(geopotential_altitude)

    # Split calculation by atmospheric layer
    troposphere_mask = geopotential_altitude <= constants.TRANSITION_ALTITUDE
    stratosphere_mask = ~troposphere_mask

    # Troposphere calculation (linear temperature decrease) (Eq A78)
    theta[troposphere_mask] = (1 - constants.TROPOSPHERE_LAPSE * geopotential_altitude[troposphere_mask])

    # Stratosphere calculation (constant temperature) (Eq A86)
    theta[stratosphere_mask] = constants.TROPOSPHERE_TEMP_RATIO

    return theta


def pressure_ratio(geopotential_altitude: np.ndarray) -> np.ndarray:
    """
    Calculate the pressure ratio (p/p0) for given geopotential altitude.

    This function implements the standard atmosphere pressure model:
    - Power law decrease in troposphere (0 to 36,089 ft)
    - Exponential decrease in stratosphere (36,089 to 65,617 ft)

    Args:
        geopotential_altitude: Array of altitudes in feet

    Returns:
        Array of pressure ratios (dimensionless)

    Raises:
        ValueError: If altitude is outside valid range [-16,404 to 65,617 ft]
    """
    # Input validation
    if np.any(geopotential_altitude < constants.MIN_ALTITUDE) or np.any(geopotential_altitude > constants.MAX_ALTITUDE):
        raise ValueError(f"Altitude must be between {constants.MIN_ALTITUDE} "
                         f"and {constants.MAX_ALTITUDE} feet")

    # Initialize output array
    delta = np.zeros_like(geopotential_altitude)

    # Split calculation by atmospheric layer
    troposphere_mask = geopotential_altitude <= constants.TRANSITION_ALTITUDE
    stratosphere_mask = ~troposphere_mask

    # Troposphere calculation (power law) (Eq A79)
    delta[troposphere_mask] = (1 - constants.TROPOSPHERE_LAPSE * geopotential_altitude[
        troposphere_mask]) ** constants.TROPOSPHERE_EXP

    # Stratosphere calculation (exponential decay) (Eq A87)
    delta[stratosphere_mask] = constants.STRATOSPHERE_PRESSURE_COEFF * np.exp(
        constants.STRATOSPHERE_DECAY * (geopotential_altitude[stratosphere_mask] - constants.TRANSITION_ALTITUDE))

    return delta


def density_ratio(geopotential_altitude: np.ndarray) -> np.ndarray:
    """
    Calculate the density ratio (ρ/ρ0) for given geopotential altitude.

    This function implements the standard atmosphere density model:
    - Power law decrease in troposphere (0 to 36,089 ft)
    - Exponential decrease in stratosphere (36,089 to 65,617 ft)

    The density ratio can be derived from the temperature and pressure ratios
    using the perfect gas law relationship: σ = δ/θ

    Args:
        geopotential_altitude: Array of altitudes in feet

    Returns:
        Array of density ratios (dimensionless)

    Raises:
        ValueError: If altitude is outside valid range [-16,404 to 65,617 ft]
    """
    # Input validation
    if np.any(geopotential_altitude < constants.MIN_ALTITUDE) or np.any(geopotential_altitude > constants.MAX_ALTITUDE):
        raise ValueError(f"Altitude must be between {constants.MIN_ALTITUDE} "
                         f"and {constants.MAX_ALTITUDE} feet")

    # Initialize output array
    sigma = np.zeros_like(geopotential_altitude)

    # Split calculation by atmospheric layer
    troposphere_mask = geopotential_altitude <= constants.TRANSITION_ALTITUDE
    stratosphere_mask = ~troposphere_mask

    # Troposphere calculation (power law) (Equation A80)
    sigma[troposphere_mask] = (1 - constants.TROPOSPHERE_LAPSE * geopotential_altitude[troposphere_mask]) ** (
            constants.TROPOSPHERE_EXP - 1)

    # Stratosphere calculation (exponential decay) (Equation A88)
    sigma[stratosphere_mask] = constants.STRATOSPHERE_DENSITY_COEFF * np.exp(
        constants.STRATOSPHERE_DECAY * (geopotential_altitude[stratosphere_mask] - constants.TRANSITION_ALTITUDE))

    return sigma

# ## Classes
# 
# These utility classes are provided for you to help build the atmosphere models and load the flight data.
class AtmosphereModel(Protocol):
    """
    Protocol defining interface for atmosphere models.

    All atmosphere models must implement these methods to provide
    standard atmospheric properties given a geopotential altitude.
    """

    def theta(self, z: np.ndarray) -> np.ndarray:
        """Temperature ratio (T/T0)."""
        ...

    def delta(self, z: np.ndarray) -> np.ndarray:
        """Pressure ratio (p/p0)."""
        ...

    def sigma(self, z: np.ndarray) -> np.ndarray:
        """Density ratio (ρ/ρ0)."""
        ...

    def temperature(self, z: np.ndarray) -> np.ndarray:
        """Return ambient temperature in Kelvin."""
        ...

    def pressure(self, z: np.ndarray) -> np.ndarray:
        """Return ambient pressure in lb/ft²."""
        ...

    def density(self, z: np.ndarray) -> np.ndarray:
        """Return ambient density in slug/ft³."""
        ...

    def speed_of_sound(self, z: np.ndarray) -> np.ndarray:
        """Return speed of sound in ft/s."""
        ...

class StandardAtmosphere:
    """
    Standard atmosphere model implementation based on US Standard Atmosphere 1976.
    """

    def __init__(self):
        self.constants = constants

    @staticmethod
    def theta(H: np.ndarray) -> np.ndarray:
        """Temperature ratio (T/T0)."""
        return temperature_ratio(H)

    @staticmethod
    def delta(H: np.ndarray) -> np.ndarray:
        """Pressure ratio (p/p0)."""
        return pressure_ratio(H)

    @staticmethod
    def sigma(H: np.ndarray) -> np.ndarray:
        """Density ratio (ρ/ρ0)."""
        return density_ratio(H)

    def temperature(self, H: np.ndarray) -> np.ndarray:
        """Return ambient temperature in Kelvin."""
        return self.theta(H) * constants.TEMP_SEA_LEVEL

    def pressure(self, H: np.ndarray) -> np.ndarray:
        """Return ambient pressure in lb/ft²."""
        return self.delta(H) * constants.PRESSURE_SEA_LEVEL

    def density(self, H: np.ndarray) -> np.ndarray:
        """Return ambient density in slug/ft³."""
        return self.sigma(H) * constants.DENSITY_SEA_LEVEL

    def speed_of_sound(self, H: np.ndarray) -> np.ndarray:
        """Return speed of sound in ft/s."""
        temperature = self.temperature(H)
        return np.sqrt(constants.SPECIFIC_HEAT_RATIO * constants.GAS_CONSTANT * temperature)


class TestAtmosphere:
    """
    Atmosphere model based on test data interpolation.
    """

    def __init__(self, filepath: str):
        data = pd.read_csv(filepath)
        self.geometric_height = data["z_ft"].to_numpy()
        self.ambient_pressure = data["pa_psi"].to_numpy()
        self.ambient_temperature = data["t_k"].to_numpy()

    def theta(self, z: np.ndarray) -> np.ndarray:
        """Temperature ratio (T/T0)."""
        return self.temperature(z) / constants.TEMP_SEA_LEVEL

    def delta(self, z: np.ndarray) -> np.ndarray:
        """Pressure ratio (p/p0)."""
        return self.pressure(z) / constants.PRESSURE_SEA_LEVEL

    def sigma(self, z: np.ndarray) -> np.ndarray:
        """Density ratio (ρ/ρ0) calculated from delta/theta."""
        return self.delta(z) / self.theta(z)

    def temperature(self, z: np.ndarray) -> np.ndarray:
        """Return ambient temperature in Kelvin."""
        return np.interp(z, self.geometric_height, self.ambient_temperature)

    def pressure(self, z: np.ndarray) -> np.ndarray:
        """Return ambient pressure in lb/ft²."""
        return np.interp(z, self.geometric_height, self.ambient_pressure) * 144  # PSI to psf

    def density(self, z: np.ndarray) -> np.ndarray:
        """Return ambient density in slug/ft³."""
        return self.sigma(z) * constants.DENSITY_SEA_LEVEL

    def speed_of_sound(self, z: np.ndarray) -> np.ndarray:
        """Return speed of sound in ft/s."""
        temperature = self.temperature(z)
        return np.sqrt(constants.SPECIFIC_HEAT_RATIO * constants.GAS_CONSTANT * temperature)