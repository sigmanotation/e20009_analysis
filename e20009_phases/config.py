from dataclasses import dataclass
from pathlib import Path
from spyral import INVALID_PATH


@dataclass
class ICParameters:
    """Parameters for IC trace signal analysis. Also includes downscale beam trace

    Attributes
    ----------
    baseline_window_scale: float
        The scale factor for the basline correction algorithm
    peak_separation: float
        The peak separation parameter used in scipy.signal.find_peaks
    peak_prominence: float
        The peak prominence parameter used in scipy.signal.find_peaks
    peak_max_width: float
        The maximum peak width parameter used in scipy.signal.find_peaks
    peak_threshold: float
        The minimum amplitude of a valid peak
    low_accept: int
        Minimum centroid value of a peak to be considered valid.
    high_accept: int
        Maximum centroid value of a peak to be considered valid.
    """

    baseline_window_scale: float
    peak_separation: float
    peak_prominence: float
    peak_max_width: float
    peak_threshold: float
    low_accept: int
    high_accept: int


@dataclass
class DetectorParameters:
    """Parameters describing the detector configuration

    Attributes
    ----------
    magnetic_field: float
        The magnitude of the magnetic field in Tesla
    electric_field: float
        The magnitude of the electric field in V/m
    detector_length: float
        The detector length in mm
    beam_region_radius: float
        The beam region radius in mm
    drift_velocity_path: str
        Path to file containing window and micromegas edges in time buckets
        for each run to be analyzed.
    get_frequency: float
        The GET DAQ sampling frequency in MHz. Typically 3.125 or 6.25
    garfield_file_path: str
        Path to a Garfield simulation file containing electron drift corrections

    """

    magnetic_field: float  # Tesla
    electric_field: float  # V/m
    detector_length: float  # mm
    beam_region_radius: float  # mm
    drift_velocity_path: Path
    get_frequency: float  # MHz
    garfield_file_path: Path
    do_garfield_correction: bool


@dataclass
class PadParameters:
    """Parameters describing the pad map paths

    Attributes
    ----------
    pad_geometry_path: Path
        Path to the csv file containing the pad geometry
    pad_time_path: Path
        Path to the csv file containing the pad time corrections
    pad_electronics_path: Path
        Path to the csv file containing the pad electronics ids
    pad_scale_path: Path
        Path to file that tells whether a pad is big or small
    """

    is_default: bool
    is_default_legacy: bool
    pad_geometry_path: Path
    pad_time_path: Path
    pad_electronics_path: Path
    pad_scale_path: Path


@dataclass
class SolverParameters:
    """Parameters for physics solving

    Attributes
    ----------
    gas_data_path: Path
        Path to a spyral-utils GasTarget file
    gain_match_factors_path: Path
        Path to CSV file containing gain match factors to normalize PID for each run
        to chosen subrange of runs.
    particle_id_filename: Path
        Name of a particle ID cut file
    ic_min_val: float
        Low value the desired beam region of the ion chamber spectrum
    ic_max_value: float
        High value the desired beam region of the ion chamber spectrum
    n_time_steps: int
        The number of timesteps used in the ODE solver
    interp_ke_min: float
        The minimum value of kinetic energy used in the interpolation scheme in MeV
    interp_ke_max: float
        The maximum value of kinetic energy used in the interpolation scheme in MeV
    interp_ke_bins: int
        The number of kinetic energy bins used in the interpolation scheme
    interp_polar_min: float
        The minimum value of polar angle used in the interpolation scheme in degrees
    interp_polar_max: float
        The maximum value of polar angle used in the interpolation scheme in degrees
    interp_polar_bins: int
        The number of polar angle bins used in the interpolation scheme
    """

    gas_data_path: Path
    gain_match_factors_path: Path
    particle_id_filename: Path
    ic_min_val: float
    ic_max_val: float
    n_time_steps: int
    interp_ke_min: float
    interp_ke_max: float
    interp_ke_bins: int
    interp_polar_min: float
    interp_polar_max: float
    interp_polar_bins: int
