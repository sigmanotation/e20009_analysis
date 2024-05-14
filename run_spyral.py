from spyral import (
    Pipeline,
    start_pipeline,
    GetParameters
)

from e20009_analysis.e20009_phases.PointcloudPhase import PointcloudLegacyPhase
from e20009_analysis.e20009_phases.config import (
    ICParameters,
    DetectorParameters,
    PadParameters,
)

from pathlib import Path
import multiprocessing

workspace_path = Path("D:\\e20009_analysis")
trace_path = Path("D:\\h5")

# Make directory to store beam events
beam_events_folder = workspace_path / "beam_events"
if not beam_events_folder.exists():
    beam_events_folder.mkdir()

run_min = 344
run_max = 345
n_processes = 10

pad_params = PadParameters(
    is_default=False,
    is_default_legacy=True,
    pad_geometry_path=Path("C:\\Users\\zachs\Desktop\\attpc_spyral\\e20009_analysis\\e20009_parameters\\padxy_legacy.csv"),
    pad_time_path=Path("C:\\Users\\zachs\\Desktop\\attpc_spyral\\e20009_analysis\\e20009_parameters\\pad_time_correction.csv"),
    pad_electronics_path=Path("C:\\Users\\zachs\\Desktop\\attpc_spyral\\e20009_analysis\\e20009_parameters\\pad_electronics_legacy.csv"),
    pad_scale_path=Path("C:\\Users\\zachs\\Desktop\\attpc_spyral\\e20009_analysis\\e20009_parameters\\gain_match_factors.csv"),
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
    peak_threshold=300.0
)

det_params = DetectorParameters(
    magnetic_field=3.0,
    electric_field=60000.0,
    detector_length=1000.0,
    drift_velocity_path=Path("C:\\Users\\zachs\\Desktop\\attpc_spyral\\e20009_analysis\\e20009_parameters\\drift_velocity.csv"),
    get_frequency=3.125,
    garfield_file_path=Path("C:\\Users\\zachs\\Desktop\\attpc_spyral\\e20009_analysis\\e20009_parameters\\electrons.txt"),
    do_garfield_correction=False,
)


pipe = Pipeline(
    [
        PointcloudLegacyPhase(
            get_params,
            ic_params,
            det_params,
            pad_params,
        )
    ],
    [True],
    workspace_path,
    trace_path,
)


def main():
    start_pipeline(pipe, run_min, run_max, n_processes)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
