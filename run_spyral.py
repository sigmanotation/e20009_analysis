from spyral import (
    Pipeline,
    start_pipeline,
    GetParameters,
    ClusterParameters,
    EstimateParameters,
)

from e20009_phases.PointcloudLegacyPhase import PointcloudLegacyPhase
from e20009_phases.ClusterPhase import ClusterPhase
from e20009_phases.EstimationPhase import EstimationPhase
from e20009_phases.InterpSolverPhase import InterpSolverPhase
from e20009_phases.config import (
    ICParameters,
    DetectorParameters,
    PadParameters,
    SolverParameters,
)

from pathlib import Path
import multiprocessing

#########################################################################################################
# Create workspace
workspace_path = Path("/Volumes/e20009/e20009_analysis")
trace_path = Path("/Volumes/e20009/h5")

# Make directory to store beam events
beam_events_folder = workspace_path / "beam_events"
if not beam_events_folder.exists():
    beam_events_folder.mkdir()

run_min = 344
run_max = 346
n_processes = 3

#########################################################################################################
# Define configuration
pad_params = PadParameters(
    is_default=False,
    is_default_legacy=True,
    pad_geometry_path=Path(
        "/Users/attpc/Desktop/e20009_analysis/e20009_analysis/e20009_parameters/pad_geometry_legacy.csv"
    ),
    pad_time_path=Path(
        "/Users/attpc/Desktop/e20009_analysis/e20009_analysis/e20009_parameters/pad_time_correction.csv"
    ),
    pad_electronics_path=Path(
        "/Users/attpc/Desktop/e20009_analysis/e20009_analysis/e20009_parameters/pad_electronics_legacy.csv"
    ),
    pad_scale_path=Path(
        "/Users/attpc/Desktop/e20009_analysis/e20009_analysis/e20009_parameters/pad_scale.csv"
    ),
)

get_params = GetParameters(
    baseline_window_scale=20.0,
    peak_separation=5.0,
    peak_prominence=20.0,
    peak_max_width=100.0,
    peak_threshold=30.0,
)

ic_params = ICParameters(
    baseline_window_scale=100.0,
    peak_separation=5.0,
    peak_prominence=30.0,
    peak_max_width=20.0,
    peak_threshold=300.0,
    low_accept=60,
    high_accept=411,
)

det_params = DetectorParameters(
    magnetic_field=3.0,
    electric_field=60000.0,
    detector_length=1000.0,
    beam_region_radius=20.0,
    drift_velocity_path=Path(
        "/Users/attpc/Desktop/e20009_analysis/e20009_analysis/e20009_parameters/drift_velocity.csv"
    ),
    get_frequency=3.125,
    garfield_file_path=Path(
        "/Users/attpc/Desktop/e20009_analysis/e20009_analysis/e20009_parameters/e20009_efield_correction.txt"
    ),
    do_garfield_correction=True,
)

cluster_params = ClusterParameters(
    min_cloud_size=50,
    min_points=3,
    min_size_scale_factor=0.05,
    min_size_lower_cutoff=10,
    cluster_selection_epsilon=10.0,
    circle_overlap_ratio=0.5,
    fractional_charge_threshold=0.8,
    outlier_scale_factor=0.05,
)

estimate_params = EstimateParameters(
    min_total_trajectory_points=20, smoothing_factor=100.0
)

solver_params = SolverParameters(
    gas_data_path="/Users/attpc/Desktop/e20009_analysis/e20009_analysis/e20009_parameters/e20009_target.json",
    gain_match_factors_path="/Users/attpc/Desktop/e20009_analysis/e20009_analysis/e20009_parameters/gain_match_factors.csv",
    particle_id_filename="/Users/attpc/Desktop/e20009_analysis/e20009_analysis/e20009_parameters/pid.json",
    ic_min_val=450.0,
    ic_max_val=850.0,
    n_time_steps=1000,
    interp_ke_min=0.1,
    interp_ke_max=40.0,
    interp_ke_bins=200,
    interp_polar_min=0.1,
    interp_polar_max=179.9,
    interp_polar_bins=340,
)

#########################################################################################################
# Construct pipeline
pipe = Pipeline(
    [
        PointcloudLegacyPhase(
            get_params,
            ic_params,
            det_params,
            pad_params,
        ),
        ClusterPhase(
            cluster_params,
            det_params,
        ),
        EstimationPhase(estimate_params, det_params),
        InterpSolverPhase(solver_params, det_params),
    ],
    [False, False, False, True],
    workspace_path,
    trace_path,
)


def main():
    start_pipeline(pipe, run_min, run_max, n_processes)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
