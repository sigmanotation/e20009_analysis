from spyral.core.run_stacks import form_run_string

from pathlib import Path
from numba import njit

import numpy as np
import h5py as h5

NUM_CHANNELS = 10240
THRESHOLD = 0.1

# Configuration parameters
workspace_path = Path("/Volumes/e20009/test_pulser")
trace_path = Path("/Volumes/e20009/pulser_h5")
run = 382


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

    cloud_group: h5.Group = point_file["cloud"]
    min_event: int = cloud_group.attrs["min_event"]
    max_event: int = cloud_group.attrs["max_event"]
    num_events = max_event - min_event + 1

    # Result storage
    pad_tb = np.zeros((NUM_CHANNELS, num_events), dtype=np.int64)
    pad_hits = np.zeros((NUM_CHANNELS, num_events), dtype=np.int64)

    # Go through each event
    for idx in range(num_events):
        event = min_event + idx

        cloud_data: h5.Dataset | None = None
        try:
            cloud_data = cloud_group[f"cloud_{event}"]  # type: ignore
        except Exception:
            continue

        if cloud_data is None:
            continue

        pc: np.ndarray = cloud_data[:].copy()
        pads_event, hits_event = pad_times(pc)
        pad_tb[:, idx] = pads_event
        pad_hits[:, idx] = hits_event

    # Make mask array masking pads that have many zero hits in an event
    mask = np.sum(pad_hits, axis=1) < 0.1 * num_events
    mask = np.tile(mask, (num_events, 1)).T
    pad_tb_ma = np.ma.array(pad_tb, mask=mask)

    # Find pad's average tb
    pad_tb_avg = np.ma.mean(pad_tb_ma, axis=1)
    pad_tb_err = np.ma.std(pad_tb_ma, axis=1)

    # Find run's average tb
    pad_tb_weights = pad_tb_err**-2 / np.ma.sum(pad_tb_err**-2)
    run_tb_avg = np.ma.average(pad_tb_avg, weights=pad_tb_weights)
    run_tb_err = np.ma.sqrt(1 / np.ma.sum(pad_tb_err**-2))

    # Fint run's time bucket correction factors
    tb_factors = run_tb_avg - pad_tb_avg
    tb_factors_err = np.ma.sqrt(run_tb_err**2 + pad_tb_err**2)
    tb_factors = tb_factors.filled(0.0)
    tb_factors_err = tb_factors_err.filled(0.0)

    np.savetxt(
        "/Users/attpc/Desktop/e20009_analysis/e20009_analysis/e20009_parameters/new2000mv.csv",
        np.transpose([tb_factors, tb_factors_err]),
        fmt="%.4f",
    )


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
    pads_event = np.zeros(NUM_CHANNELS, dtype=np.int64)
    hits_event = np.zeros(NUM_CHANNELS, dtype=np.int64)

    for point in pc:

        pad_id = int(point[5])
        tb = int(point[6])

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
