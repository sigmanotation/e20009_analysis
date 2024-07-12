from spyral.core.run_stacks import form_run_string

from pathlib import Path
from numba import njit

import numpy as np
import h5py as h5

NUM_CHANNELS = 10240

# Configuration parameters
workspace_path = Path("D:\\test_pulser")
trace_path = Path("D:\\pulser_h5")
run = 378


def tc_calculator(workspace_path: Path, traces_path: Path, run: int):
    """
    Calculates the micromegas and window time buckets of beam events in a run to find its drift velocity using downscale beam events.
    The error of both these edges is found as well. The results are output in a csv file in the workspace.

    The PointcloudLegacyPhase must be run on the data before the drift velocity is calculated.

    Arguments
    ---------
    workspace_path: Path
        Path to workspace where attpc_spyral results are stored.
    traces_path: Path
        Path to where HDF5 files are stored.
    run_min: int
        Minimum run to calculate the drift velocity of.
    run_max: int
        Maximum run to calculate the drift velocity of.
    """

    point_path = (
        Path(workspace_path) / "PointCloudLegacy" / f"{form_run_string(run)}.h5"
    )
    try:
        point_file = h5.File(point_path, "r")
    except Exception:
        print(f"Point cloud file not found for run {run}!")
        return 
    
    # Make results array
    sum_pads = np.zeros((NUM_CHANNELS))
    hits_per_pad = np.zeros((NUM_CHANNELS))

    # Result storage for run
    sum_pads = np.zeros(NUM_CHANNELS)
    hits_per_pad = np.zeros(NUM_CHANNELS, dtype=np.int64)

    cloud_group: h5.Group = point_file["cloud"]
    min_event: int = cloud_group.attrs["min_event"]
    max_event: int = cloud_group.attrs["max_event"]

    # Go through each event
    for idx in range(min_event, max_event + 1):

        cloud_data: h5.Dataset | None = None
        try:
            cloud_data = cloud_group[f"cloud_{idx}"]  # type: ignore
        except Exception:
            continue

        if cloud_data is None:
            continue

        pc: np.ndarray = cloud_data[:].copy()
        pads_event, hits_event = pad_times(pc)

        sum_pads += pads_event
        hits_per_pad += hits_event

    # factors = np.divide(sum_pads, hits_per_pad, where=hits_per_pad != 0)
    # mask = hits_per_pad == 0
    # ma_factors = np.ma.array(factors, mask=mask)
    # factors_avg = np.ma.average(ma_factors)
    # answer = factors_avg - factors

    mask = hits_per_pad > 0
    factors = np.divide(sum_pads, hits_per_pad, where=mask)
    factors_avg = np.average(factors[mask])
    answer = np.subtract(factors_avg, factors, where=mask)

    np.savetxt("C:\\Users\\zachs\\Desktop\\e20009_analysis\\e20009_analysis\\e20009_parameters\\welp.csv", answer, fmt="%.4f")


@njit
def pad_times(pc: np.ndarray):

    # Result storage for event
    pads_event = np.zeros(NUM_CHANNELS)
    hits_event = np.zeros(NUM_CHANNELS, dtype=np.int64)

    for point in pc:

        pad_id = int(point[5])
        tb = point[6]

        # THIS SHOULDNT BE NEEDED, ASK GORDON
        if tb == 0:
            continue

        if (
            pads_event[pad_id] == 0
        ):  # Array is initialized with 0, so we always want to replace the first time a pad is encountered
            pads_event[pad_id] = tb
            hits_event[pad_id] += 1

        elif (
            tb < pads_event[pad_id]
        ):  # Only take the smallest point from each trace in an event
            pads_event[pad_id] = tb

    return pads_event, hits_event


def main():
    tc_calculator(workspace_path, trace_path, run)


if __name__ == "__main__":
    main()
