from dataclasses import dataclass
from pathlib import Path

"""
Changes from attpc_spyral package base code (circa July 29, 2024):
    - FribParameters class removed and renamed to ICParameters. All ic parameters of
      that class were removed and replaced with low_accept and high_accept.
    - DetectorParameters class had window and micromegas timebucket parameters removed
      and instead are replaced by the single parameter drift_velocity_path.
    - SolverParameters class has new parameter gain_match_factors_path.
"""


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
        Low value of the desired beam region of the ion chamber spectrum (inclusive)
    ic_max_value: float
        High value of the desired beam region of the ion chamber spectrum (exclusive)
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
