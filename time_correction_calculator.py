from spyral.core.run_stacks import form_run_string

from pathlib import Path
from numba import njit
from scipy.stats import sem

import numpy as np
import h5py as h5
import lmfit

NUM_CHANNELS = 10240
THRESHOLD = 0.1

# Configuration parameters
workspace_path = Path("D:\\tcorr")
write_path = Path("C:\\Users\\zachs\\Desktop")
write_raw = True
run_min = 372
run_max = 376

def time_correction_calculator(
    workspace_path: Path,
    run_min: int,
    run_max: int,
    write_raw: bool,
    write_path: Path,
):
    """
    Determines the time correction (in time buckets) of each pad using pulser runs. Calculates
    the time correction factors from each pulser run and applies a linear fit to them. The
    intercept of the fit for each pad is written to disk as the time correction factor.

    WARNING: The PointcloudLegacyPhase must be run on the data before the time correction
    factors are found. Also, two things must be turned off in the PointCloudLegacyPhase.
    First, turn the condition off that if a pad has more than x points it is not added to the
    point cloud. Second, remove the time correction factor from being applied to the time
    bucket of the point cloud (because this function will find it).

    WARNING: The GetParameters needed to be tweaked when running the pulser runs through
    the PointcloudLegacyPhase of Spyral. Specifically, the peak threshold was set to 100 and 
    the max peak width to 300.

    Parameters
    ----------
    workspace_path: Path
        Path to workspace where attpc_spyral results are stored.
    run_min: int
        Minimum run to calculate the drift velocity of.
    run_max: int
        Maximum run to calculate the drift velocity of.
    write_raw: bool
        Whether to write the results array to disk. This has all the
        time correction factors from each pad for all pulser runs along
        with their errors and that pad's average signal amplitude (after
        background removal).
    write_path:
        Directory to write time correction factors to and the results array
        if write_raw is true.
    """

    num_runs = run_max - run_min + 1

    # Column schema is (time bucket, time bucket error, average amplitude)
    results = np.zeros((NUM_CHANNELS, 4, num_runs))
    fit_results = np.zeros((NUM_CHANNELS, 2))

    # Find time correction factors from each pulser run
    for idx, run in enumerate(range(run_min, run_max + 1)):
        run_results = pulser_results(workspace_path, run)
        results[:, :, idx] = run_results

    # Linearly fit time correction factor as function of pad amplitude for each pad
    for pad in range(NUM_CHANNELS):

        # Don't fit pads that have zero signal amplitude for all runs
        if np.sum(results[pad, 2, :], axis=0) == 0.0:
            continue

        linear_fit = lmfit.models.LinearModel()

        weights = np.sqrt(results[pad, 1, :])
        weights = np.divide(
            1, weights, out=np.zeros_like(weights), where=weights != 0.0
        )

        pars = linear_fit.guess(
            x=results[pad, 2, :],
            data=results[pad, 0, :],
            weights=weights,
        )
        fit = linear_fit.fit(
            params=pars,
            x=results[pad, 2, :],
            data=results[pad, 0, :],
            weights=weights,
        )

        fit_results[pad, 0] = fit.params["slope"]
        fit_results[pad, 1] = fit.params["intercept"]

    if write_raw is True:
        np.save(write_path / "time_correction_results.npy", results)

    np.savetxt(write_path / "pad_time_correction.csv", fit_results[:, 1], header='tcorr', comments='', newline=",\n", fmt="%.4f")

def pulser_results(workspace_path: Path, run: int):
    """
    Determines the time correction of each pad in a pulser run. This function analyzes the
    point clouds produced from running the pulser run through Spyral. For each pad in a pulser
    run, its earliest point from each event is recorded. The average of these is the time
    correction for that pad. Its error is the standard error of the mean.

    Parameters
    ----------
    workspace_path: Path
        Path to workspace where attpc_spyral results are stored.
    run: int
        Run number of pulser run to analyze.

    Returns
    -------
    np.ndarray
        NUM_CHANNELS x 3 array where the row index is the pad number.
        Column schema is (correction factor, correction factor error,
        average pad amplitude).
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
    run_results = np.zeros((NUM_CHANNELS, num_events, 3))

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
        evt_res = event_results(pc)
        run_results[:, idx, 0] = evt_res[:, 0]  # Time bucket
        run_results[:, idx, 1] = evt_res[:, 1]  # Amplitude
        run_results[:, idx, 2] = evt_res[:, 2]  # Hit

    # Make mask array masking pads that have number of hits below the threshold
    mask = np.sum(run_results[:, :, 2], axis=1) < 0.1 * num_events
    mask_array = np.tile(mask, (num_events, 1)).T
    pad_tb_ma = np.ma.array(run_results[:, :, 0], mask=mask_array)

    # Find pad's average tb
    pad_tb_avg = np.ma.mean(pad_tb_ma, axis=1)
    pad_tb_err = np.ma.array(sem(pad_tb_ma.filled(0.0), axis=1), mask=mask)

    # Find run's average tb
    pad_tb_weights = pad_tb_err**-2 / np.ma.sum(pad_tb_err**-2)
    run_tb_avg = np.ma.average(pad_tb_avg, weights=pad_tb_weights)
    run_tb_err = np.ma.sqrt(1 / np.ma.sum(pad_tb_err**-2))

    # Find run's time bucket correction factors
    tb_factors = run_tb_avg - pad_tb_avg
    tb_factors_err = np.ma.sqrt(run_tb_err**2 + pad_tb_err**2)
    tb_factors = tb_factors.filled(0.0)
    tb_factors_err = tb_factors_err.filled(0.0)

    # Find average amplitude of each pad
    pad_amp_avg = np.mean(run_results[:, :, 1], axis=1)
    pad_amp_avg[mask] = 0.0

    # Find average amplitude error of each pad
    pad_amp_err = sem(run_results[:, :, 1], axis=1)
    pad_amp_err[mask] = 0.0

    return np.column_stack((tb_factors, tb_factors_err, pad_amp_avg, pad_amp_err))


@njit
def event_results(pc: np.ndarray):
    """
    Auxiliary function to pulser_results. It finds the first point of each pad
    in a given event, recording its time bucket and amplitude. It also indicates
    which pad was hit.

    This function encompasses a large for-loop through all the points of an
    event's point cloud and is jitted to speed up the time it takes to run
    pulser_results.

    Note: this function turns the time bucket of the point into an integer,
    i.e. it rounds it down. We don't want our time bucket smearing to be 
    taken into consideration; we already know that the error in a time bucket 
    is one, we are searching for the error of the time bucket.

    Parameters
    ----------
    pc: np.ndarray
        Point cloud of an event produced by Spyral

    Returns
    -------
    np.ndarray
        NUM_CHANNELS x 3 array where the row index is the pad number.
        Column schema is (correction factor, pad amplitude, hit indicator).
    """

    # Column schema is (time bucket, amplitude, hit)
    event_results = np.zeros((NUM_CHANNELS, 3))

    for point in pc:

        pad_id = int(point[5])
        tb = int(point[6])
        amp = point[3]

        if (
            event_results[pad_id, 0] == 0
        ):  # Array is initialized with 0, so we always want to replace the first time a pad is encountered
            event_results[pad_id, 0] = tb
            event_results[pad_id, 1] = amp
            event_results[pad_id, 2] = 1

        elif (
            tb < event_results[pad_id, 0]
        ):  # Only take the smallest point from each trace in an event
            event_results[pad_id, 0] = tb
            event_results[pad_id, 1] = amp

    return event_results


def main():
    time_correction_calculator(workspace_path, run_min, run_max, write_raw, write_path)


if __name__ == "__main__":
    main()
