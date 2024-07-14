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
    Determines the time correction of each pad using pulser runs. This function analyzes the 
    point clouds produced from running pulser runs through Spyral. For each pad in a pulser 
    run, its earliest point from each event is added together. This number is divided by number 
    of terms in the sum to find the average. The time correction factor is in time buckets. 

    WARNING: The PointcloudLegacyPhase must be run on the data before the time correction 
    factors are found. Also, two things must be turned off in the PointCloudLegacyPhase.
    First, turn the condition off that if a pad has more than x points it is not added to the 
    point cloud. Second, remove the time correction factor from being applied to the time
    bucket of the point cloud (because this function will find it).

    Parameters
    ----------
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

    # Result storage
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

    mask = hits_per_pad != 0
    factors = np.divide(sum_pads, hits_per_pad, where=mask)
    factors_avg = np.mean(factors[mask])
    answer = np.where(mask, factors_avg-factors, 0)

    np.savetxt("C:\\Users\\zachs\\Desktop\\e20009_analysis\\e20009_analysis\\e20009_parameters\\welp.csv", answer, fmt="%.4f")


@njit
def pad_times(pc: np.ndarray):
    """ 
    Auxiliary function to tc_calculator. It finds the first point of each pad
    in a given event, recording its time bucket.
    
    This function encompasses a large for loop through all the points of an 
    event's point cloud and is jitted to speed up the time it takes to run 
    tc_calculator.

    Parameters
    ----------
    pc: np.ndarray
        Point cloud of an event produced by Spyral

    Returns
    -------
    pads_event: np.ndarray
        NUM_CHANNELSx1 array of each pad's earliest point in
        the point cloud. The pad number is given by the index.

    hits_event: np.ndarray
        NUM_CHANNELSx1 array indicating if a pad saw any points.
        It is either 1 or 0.
    """

    # Result storage for event
    pads_event = np.zeros(NUM_CHANNELS)
    hits_event = np.zeros(NUM_CHANNELS, dtype=np.int64)

    for point in pc:

        pad_id = int(point[5])
        tb = point[6]

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
